import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import yaml
import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import precision_recall_fscore_support
from data.utils import load_test_pairs
from data.dataset import InferenceDataset
from eval.model_wrapper import load_embedding_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from datetime import datetime

def compute_embeddings(model, image_paths, transform, device, batch_size):
    dataset = InferenceDataset(image_paths, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    embeddings = {}
    model.eval()
    with torch.no_grad():
        for imgs, paths in tqdm(loader, desc="Embedding images"):
            imgs = imgs.to(device)
            feats = model(imgs)
            feats = F.normalize(feats, dim=-1)
            for path, feat in zip(paths, feats):
                embeddings[path] = feat.cpu()
    return embeddings

def compute_similarity_scores(model, pairs, transform, device, batch_size):
    # Step 1: collect all unique image paths
    all_paths = set()
    for path1, path2, _ in pairs:
        all_paths.add(path1)
        all_paths.add(path2)

    # Step 2: compute all embeddings in batch
    embedding_map = compute_embeddings(model, list(all_paths), transform, device, batch_size)

    # Step 3: score pairs
    scores = []
    labels = []
    for path1, path2, label in tqdm(pairs, desc="Scoring pairs"):
        emb1 = embedding_map[path1]
        emb2 = embedding_map[path2]
        sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        scores.append(sim)
        labels.append(label)
    return scores, labels

def find_best_threshold(scores, labels):
    best_f1 = 0
    best_thresh = 0
    for thresh in torch.linspace(0, 1, steps=100):
        preds = [1 if s >= thresh else 0 for s in scores]
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh.item()
    return best_thresh, best_f1

def evaluate(model, pairs, transform, threshold, device):
    scores, labels = compute_similarity_scores(model, pairs, transform, device)
    preds = [1 if s >= threshold else 0 for s in scores]
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return precision, recall, f1

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def log_results(config, model_name, dataset_name, precision, recall, f1, best_thresh):
    base_log_dir = config.get("log_dir", "logs")
    os.makedirs(base_log_dir, exist_ok=True)
    log_folder = os.path.join(base_log_dir, f"{model_name}_projection{config['use_projection']}_{config['head_type']}_vitmode{config['vit_mode']}")
    os.makedirs(log_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_folder, f"eval_{timestamp}.txt")

    with open(log_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Use Projection Head: {config.get('use_projection', False)}\n")
        f.write(f"ViT Mode (if applicable): {config.get('vit_mode', 'N/A')}\n")
        f.write(f"Best Threshold: {best_thresh:.4f}\n")
        f.write(f"Test Precision: {precision:.4f}\n")
        f.write(f"Test Recall: {recall:.4f}\n")
        f.write(f"Test F1 Score: {f1:.4f}\n")

    print(f"\n📁 Results logged to: {log_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_name = config.get("dataset_name", "deepfakeart")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = load_embedding_model(config)
    model.eval()

    os.chdir(config["data_root"])

    print("\nFinding best threshold on train split...")
    train_pairs = load_test_pairs(config["train_similar_json"], config["train_dissimilar_json"])
    train_scores, train_labels = compute_similarity_scores(model, train_pairs, transform, config["device"], config['batch_size'])
    best_thresh, best_f1 = find_best_threshold(train_scores, train_labels)
    print(f"Best threshold: {best_thresh:.4f}, Train F1: {best_f1:.4f}")

    print("\nEvaluating on test split...")
    test_pairs = load_test_pairs(config["test_similar_json"], config["test_dissimilar_json"])
    precision, recall, f1 = evaluate(model, test_pairs, transform, best_thresh, config["device"])
    print(f"Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    log_results(config, config["model_name"], dataset_name, precision, recall, f1, best_thresh)

if __name__ == "__main__":
    main()
