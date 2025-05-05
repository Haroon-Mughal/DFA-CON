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
from models.resnet import ResNetWithHead
from tqdm import tqdm
import os
from PIL import Image

@torch.no_grad()
def extract_embedding(model, path, transform, device):
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    feat = model(img)
    return F.normalize(feat, dim=1)

def compute_similarity_scores(model, pairs, transform, device):
    model.eval()
    scores = []
    labels = []
    for path1, path2, label in tqdm(pairs, desc="Scoring pairs"):
        emb1 = extract_embedding(model, path1, transform, device)
        emb2 = extract_embedding(model, path2, transform, device)
        sim = F.cosine_similarity(emb1, emb2).item()
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

def load_model(model_name, feature_dim, model_path, device):
    if model_name.lower() == "resnet50":
        model = ResNetWithHead(head_type='mlp', feature_dim=feature_dim).to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = load_model(config["model_name"], config["feature_dim"], config["model_path"], config["device"])
    model.eval()

    os.chdir(config["data_root"])

    print("\nFinding best threshold on train split...")
    train_pairs = load_test_pairs(config["train_similar_json"], config["train_dissimilar_json"])
    train_scores, train_labels = compute_similarity_scores(model, train_pairs, transform, config["device"])
    best_thresh, best_f1 = find_best_threshold(train_scores, train_labels)
    print(f"Best threshold: {best_thresh:.4f}, Train F1: {best_f1:.4f}")

    print("\nEvaluating on test split...")
    test_pairs = load_test_pairs(config["test_similar_json"], config["test_dissimilar_json"])
    precision, recall, f1 = evaluate(model, test_pairs, transform, best_thresh, config["device"])
    print(f"Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    main()
