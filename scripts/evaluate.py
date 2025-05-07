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
from eval.model_wrapper import load_embedding_model
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from collections import defaultdict

def extract_embedding(model, path, transform, device):
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    feat = model(img)
    return F.normalize(feat, dim=-1)

def compute_similarity_scores(model, pairs, transform, device):
    model.eval()
    scores = []
    labels = []
    types_ = []
    for path1, path2, label, attack_type in tqdm(pairs, desc="Scoring pairs"):
        emb1 = extract_embedding(model, path1, transform, device)
        emb2 = extract_embedding(model, path2, transform, device)
        sim = F.cosine_similarity(emb1, emb2).item()
        scores.append(sim)
        labels.append(label)
        types_ = types_ + [attack_type]
    return scores, labels, types_

def evaluate_by_type(model, pairs, transform, threshold, device):
    scores, labels, types_ = compute_similarity_scores(model, pairs, transform, device)
    preds = [1 if s >= threshold else 0 for s in scores]

    all_results = {}

    # Whole set evaluation
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    all_results["whole_set"] = (precision, recall, f1)

    # Per attack type
    attack_types = ["inpainting", "style_transfer", "adversarial", "cutmix"]
    for attack in attack_types:
        sub_scores = []
        sub_labels = []

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
    train_scores, train_labels, train_types = compute_similarity_scores(model, train_pairs, transform, config["device"])
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
    all_results = evaluate_by_type(model, test_pairs, transform, best_thresh, config["device"])

    for key, (p, r, f1) in all_results.items():
        print(f"{key}: Precision={p:.4f}, Recall={r:.4f}, F1={f1:.4f}")

    log_results(config, config["model_name"], dataset_name, best_thresh, all_results)

if __name__ == "__main__":
    main()
