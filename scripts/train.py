import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.dataset import ContrastiveTrainDataset
from data.utils import contrastive_collate_fn
from data.utils import prepare_data_splits
from models.resnet import ResNetWithHead
from loss.supcon import SupConLoss
from train.trainer import train_model

def main():
    parser = argparse.ArgumentParser(description="Train SupCon on DeepfakeArt")

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

    args = parser.parse_args()

    # === Load and split data ===
    train_map, val_map, _ = prepare_data_splits(args.train_sim_json, args.test_sim_json, val_ratio=args.val_split)

    imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
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


    train_dataset = ContrastiveTrainDataset(train_map, root_dir=args.data_root, transform=transform_train)
    val_dataset = ContrastiveTrainDataset(val_map, root_dir=args.data_root, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=contrastive_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=contrastive_collate_fn, num_workers=4)

    # === Model selection ===
    if args.model_name.lower() == "resnet50":
        model = ResNetWithHead(head_type='mlp', feature_dim=args.feature_dim).to(args.device)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    # === Loss function selection ===
    if args.loss_name.lower() == "supcon":
        loss_fn = SupConLoss()
    else:
        raise ValueError(f"Unsupported loss function: {args.loss_name}")

    # === Optimizer ===
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # === Train ===
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=args.device,
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        model_name=args.model_name,
        batch_size=args.batch_size,
        lr=args.lr,
        loss_name=args.loss_name,
        checkpoint_dir=args.checkpoint_dir
    )

if __name__ == "__main__":
    main()
