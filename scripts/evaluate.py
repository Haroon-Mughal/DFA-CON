import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



import sys
import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import precision_recall_fscore_support
from data.utils import load_test_pairs
from data.dataset import InferenceDataset
from eval.model_wrapper import load_embedding_model
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from collections import defaultdict

def compute_embeddings(model, image_paths, transform, device, batch_size):
    dataset = InferenceDataset(image_paths, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    embedding_map = {}

    model.eval()
    with torch.no_grad():
        for imgs, paths in tqdm(loader, desc="Embedding images"):
            imgs = imgs.to(device)
            feats = model(imgs)
            feats = F.normalize(feats, dim=-1)
            for path, feat in zip(paths, feats):
                embedding_map[path] = feat.cpu()

    return embedding_map

def compute_similarity_scores(model, pairs, transform, device, batch_size):
    all_paths = set()
    for path1, path2, _, _ in pairs:
        all_paths.add(path1)
        all_paths.add(path2)

    embedding_map = compute_embeddings(model, list(all_paths), transform, device, batch_size)

    scores, labels, types_ = [], [], []
    for path1, path2, label, attack_type in tqdm(pairs, desc="Scoring pairs"):
        emb1 = embedding_map[path1]
        emb2 = embedding_map[path2]
        sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        scores.append(sim)
        labels.append(label)
        types_.append(attack_type)

    return scores, labels, types_

def evaluate_by_type(model, pairs, transform, threshold, device, batch_size):
    scores, labels, types_ = compute_similarity_scores(model, pairs, transform, device, batch_size)
    preds = [1 if s >= threshold else 0 for s in scores]

    all_results = {}
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    all_results["whole_set"] = (precision, recall, f1)

    attack_types = ["inpainting", "style_transfer", "adversarial", "cutmix"]
    for attack in attack_types:
        sub_scores, sub_labels = [], []
        for s, l, t in zip(scores, labels, types_):
            if t == attack or t == "original":
                sub_scores.append(s)
                sub_labels.append(l)
        sub_preds = [1 if s >= threshold else 0 for s in sub_scores]
        precision, recall, f1, _ = precision_recall_fscore_support(sub_labels, sub_preds, average="binary")
        all_results[attack] = (precision, recall, f1)

    return all_results

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def log_results(config, model_name, dataset_name, best_thresh, all_results):
    base_log_dir = config.get("log_dir", "logs")
    os.makedirs(base_log_dir, exist_ok=True)
    log_folder = os.path.join(base_log_dir, f"{model_name}_{dataset_name}")
    os.makedirs(log_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_folder, f"eval_{timestamp}.txt")

    with open(log_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Use Projection Head: {config.get('use_projection', False)}\n")
        f.write(f"ViT Mode (if applicable): {config.get('vit_mode', 'N/A')}\n")
        f.write(f"Best Threshold: {best_thresh:.4f}\n")
        f.write("\nPerformance by category:\n")
        for key, (p, r, f1) in all_results.items():
            f.write(f"{key}: Precision={p:.4f}, Recall={r:.4f}, F1={f1:.4f}\n")

    print(f"\n📁 Results logged to: {log_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_name = config.get("dataset_name", "deepfakeart")
    batch_size = config.get("eval_batch_size", 64)

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
    train_scores, train_labels, train_types = compute_similarity_scores(model, train_pairs, transform, config["device"], batch_size)
    best_thresh = 0.0
    best_f1 = 0.0
    for thresh in torch.linspace(0, 1, steps=100):
        preds = [1 if s >= thresh else 0 for s in train_scores]
        _, _, f1, _ = precision_recall_fscore_support(train_labels, preds, average="binary")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh.item()

    print(f"Best threshold: {best_thresh:.4f}, Train F1: {best_f1:.4f}")

    print("\nEvaluating on test split...")
    test_pairs = load_test_pairs(config["test_similar_json"], config["test_dissimilar_json"])
    all_results = evaluate_by_type(model, test_pairs, transform, best_thresh, config["device"], batch_size)

    for key, (p, r, f1) in all_results.items():
        print(f"{key}: Precision={p:.4f}, Recall={r:.4f}, F1={f1:.4f}")

    log_results(config, config["model_name"], dataset_name, best_thresh, all_results)

if __name__ == "__main__":
    main()
