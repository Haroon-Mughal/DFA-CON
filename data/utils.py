import os
import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

def load_anchor_positive_map(json_path: str) -> Dict[str, List[str]]:
    """Parses a DeepfakeArt-style JSON file and builds a mapping from anchor to its forged versions."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    anchor_to_positives = defaultdict(list)

    for attack_type, entries in data.items():
        for _, entry in entries.items():
            anchor = entry["original"]
            positive = entry["generated"]
            anchor_to_positives[anchor].append(positive)

    return anchor_to_positives

def split_train_val(anchor_to_positives: Dict[str, List[str]], val_ratio: float = 0.2,
    seed: int = 42) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Splits the anchor-to-positives mapping into training and validation subsets."""
    anchors = list(anchor_to_positives.keys())
    random.seed(seed)
    random.shuffle(anchors)

    split_idx = int(len(anchors) * (1 - val_ratio))
    train_anchors = anchors[:split_idx]
    val_anchors = anchors[split_idx:]

    train_map = {a: anchor_to_positives[a] for a in train_anchors}
    val_map = {a: anchor_to_positives[a] for a in val_anchors}

    return train_map, val_map
    

def prepare_data_splits(train_json: str, test_json: str, val_ratio: float = 0.2):
    """
    Loads and splits the dataset JSON files into train/val/test mappings.
    Returns:
        train_map, val_map, test_map: Dict[str, List[str]]
    """
    train_full_map = load_anchor_positive_map(train_json)
    test_map = load_anchor_positive_map(test_json)
    train_map, val_map = split_train_val(train_full_map, val_ratio)
    return train_map, val_map, test_map

def load_test_pairs(similar_json_path: str, dissimilar_json_path: str) -> List[Tuple[str, str, int, str]]:
    """
    Loads image pairs and labels from the DeepfakeArt test set format.

    Returns:
        List of (img_path1, img_path2, label, attack_type)
        where label = 1 for similar (forged), 0 for dissimilar (unrelated)
    """
    pairs = []

    # Load similar pairs (label = 1)
    with open(similar_json_path, 'r') as f:
        similar_data = json.load(f)
        for attack_type, entries in similar_data.items():
            for _, entry in entries.items():
                img1 = entry["original"]
                img2 = entry["generated"]
                pairs.append((img1, img2, 1, attack_type))

    # Load dissimilar pairs (label = 0)
    with open(dissimilar_json_path, 'r') as f:
        dissimilar_data = json.load(f)
        for attack_type, entries in dissimilar_data.items():
            for _, entry in entries.items():
                img1 = entry["image_0"]
                img2 = entry["image_1"]
                pairs.append((img1, img2, 0, attack_type))

    return pairs



def contrastive_collate_fn(batch):
    """
    Custom collate function for SupCon training.
    Flattens anchor + positive images into a single list.
    Assigns group_id to each image so SupCon can compute similarities.
    """
    all_images = []
    all_group_ids = []

    for item in batch:
        anchor = item["anchor"]
        positives = item["positives"]
        group_id = item["group_id"]

        # Add anchor and each positive to the list
        all_images.append(anchor)
        all_group_ids.append(group_id)

        for pos in positives:
            all_images.append(pos)
            all_group_ids.append(group_id)

    images_tensor = torch.stack(all_images, dim=0)  # [N_total, C, H, W]
    group_ids_tensor = torch.tensor(all_group_ids, dtype=torch.long)  # [N_total]

    return images_tensor, group_ids_tensor






