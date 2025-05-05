import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml

from data.dataset import ContrastiveTrainDataset
from data.utils import contrastive_collate_fn
from data.utils import prepare_data_splits
from models.resnet import ResNetWithHead
from loss.supcon import SupConLoss
from train.trainer import train_model

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train SupCon on DeepfakeArt")
    """ 
     # === General training arguments ===
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of dataset")
    parser.add_argument("--train_sim_json", type=str, required=True, help="Path to train_similar.json")
    parser.add_argument("--train_dis_json", type=str, required=True, help="Path to train_dissimilar.json")
    parser.add_argument("--test_sim_json", type=str, required=True, help="Path to test_similar.json")
    parser.add_argument("--test_dis_json", type=str, required=True, help="Path to test_dissimilar.json")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")

    # === Hyperparameters ===
    parser.add_argument("--batch_size", type=int, default=32, help="Number of anchor groups per batch")
    parser.add_argument("--epochs", type=int, default=100, help="Total number of training epochs")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Number of warm-up epochs")
    parser.add_argument("--lr", type=float, default=0.5, help="Initial learning rate")
    parser.add_argument("--feature_dim", type=int, default=128, help="Projection head output dimension")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    # === Model and checkpoint ===
    parser.add_argument("--model_name", type=str, default="resnet50", help="Model name (e.g., resnet50)")
    parser.add_argument("--loss_name", type=str, default="supcon", help="Loss function name (e.g., supcon)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")

    # === Device ===
    parser.add_argument("--device", type=str, default="cuda", help="Training device: cuda or cpu")
    """
    parser.add_argument("--config", type=str, required=True, help="Path to training YAML config file")
    
    args = parser.parse_args()
    
    config = load_config(args.config)


    # === Load and split data ===
    #train_map, val_map, _ = prepare_data_splits(args.train_sim_json, args.test_sim_json, val_ratio=args.val_split)
    train_map, val_map, _ = prepare_data_splits(
        config["train_sim_json"], config["test_sim_json"], val_ratio=config["val_split"])

    imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
    transforms.Resize((224, 224)),    
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    imagenet_norm])
    
    transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    imagenet_norm])


    train_dataset = ContrastiveTrainDataset(train_map, root_dir=config["data_root"], transform=transform_train)
    val_dataset = ContrastiveTrainDataset(val_map, root_dir=config["data_root"], transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                              collate_fn=contrastive_collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False,
                            collate_fn=contrastive_collate_fn, num_workers=2)

     # === Model selection ===
    if config["model_name"].lower() == "resnet50":
        model = ResNetWithHead(config['head_type'], feature_dim=config["feature_dim"]).to(config["device"])
    else:
        raise ValueError(f"Unsupported model: {config['model_name']}")

    # === Loss function selection ===
    if config["loss_name"].lower() == "supcon":
        loss_fn = SupConLoss()
    else:
        raise ValueError(f"Unsupported loss function: {config['loss_name']}")

    # === Optimizer ===
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=1e-4)


    # === Train ===
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=config["device"],
        num_epochs=config["epochs"],
        early_stopping_patience=config["patience"],
        warmup_epochs=config["warmup_epochs"],
        model_name=config["model_name"],
        batch_size=config["batch_size"],
        lr=config["lr"],
        loss_name=config["loss_name"],
        checkpoint_dir=config["checkpoint_dir"],
        log_root=config['log_root'],
        head_type=config['head_type']
    )

if __name__ == "__main__":
    main()
