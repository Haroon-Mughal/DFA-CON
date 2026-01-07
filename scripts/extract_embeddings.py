
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.dataset import InferenceDataset
from eval.model_wrapper import load_embedding_model


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_image_paths(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "image_paths" in data:
        return data["image_paths"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("image_paths_json must be a list or contain key 'image_paths'")


def make_run_name(cfg):
    return "_".join([
        cfg.get("model_name", "model"),
        cfg.get("dataset_name", "dataset"),
        f"proj{int(bool(cfg.get('use_projection', False)))}",
        str(cfg.get("head_type", "head")),
        str(cfg.get("vit_mode", "none")),
    ])


@torch.inference_mode()
def extract_embeddings(model, dataset, device, batch_size, num_workers, processor=None):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model.eval()
    model.to(device)

    all_embeddings = []
    all_paths = []

    for imgs, paths in tqdm(loader, desc="Extracting embeddings"):
        if processor is not None:
            pixel_values = processor(images=list(imgs), return_tensors="pt").pixel_values.to(device)
            feats = model(pixel_values)
        else:
            imgs = imgs.to(device)
            feats = model(imgs)

        feats = F.normalize(feats, dim=-1)
        all_embeddings.append(feats.cpu().numpy())
        all_paths.extend(paths)

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    return embeddings, all_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # --------------------
    # Load model
    # --------------------
    result = load_embedding_model(cfg)
    if isinstance(result, tuple):
        model, processor = result
        transform = None
    else:
        model = result
        processor = None
        transform = transforms.Compose([
            transforms.Resize((cfg.get("image_size", 224), cfg.get("image_size", 224))),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.get("normalize_mean", [0.485, 0.456, 0.406]),
                std=cfg.get("normalize_std", [0.229, 0.224, 0.225]),
            ),
        ])

    # --------------------
    # Load image paths
    # --------------------
    image_paths = load_image_paths(cfg["image_paths_json"])

    # Resolve paths relative to data_root (same logic as evaluate.py)
    os.chdir(cfg["data_root"])

    dataset = InferenceDataset(image_paths, transform=transform)

    # --------------------
    # Output directory
    # --------------------
    run_name = make_run_name(cfg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.get("output_dir", "embeddings_out")) / f"{run_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------
    # Extract embeddings
    # --------------------
    embeddings, paths = extract_embeddings(
        model=model,
        dataset=dataset,
        device=cfg.get("device", "cpu"),
        batch_size=cfg.get("eval_batch_size", 64),
        num_workers=cfg.get("num_workers", 2),
        processor=processor,
    )

    # --------------------
    # Save outputs
    # --------------------
    np.save(out_dir / "embeddings.npy", embeddings)

    with open(out_dir / "paths.json", "w", encoding="utf-8") as f:
        json.dump({"paths": paths, "num_images": len(paths)}, f, indent=2)

    meta = {
        "model_name": cfg.get("model_name"),
        "dataset_name": cfg.get("dataset_name"),
        "num_images": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "device": cfg.get("device"),
        "config": cfg,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # --------------------
    # Optional FAISS index
    # --------------------
    if cfg.get("save_faiss", False):
        try:
            import faiss
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            faiss.write_index(index, str(out_dir / "faiss.index"))
            print("FAISS index saved")
        except Exception as e:
            print(f"[WARN] FAISS not saved: {e}")

    print(f"\nEmbeddings saved to: {out_dir}")
    print(f"Shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
