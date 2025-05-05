import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import math

def cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
    """
    Builds a learning rate scheduler with warm-up followed by cosine decay.
    """
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        progress = (current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1. + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def train_supcon_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    all_embeddings = []

    for images, group_ids in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        group_ids = group_ids.to(device)

        optimizer.zero_grad()
        features = model(images)

        all_embeddings.append(features.detach().cpu())
        loss = loss_fn(features, group_ids)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    feature_variance = torch.cat(all_embeddings).var(dim=0).mean().item()

    return avg_loss, feature_variance


def validate_supcon_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_embeddings = []

    with torch.no_grad():
        for images, group_ids in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device)
            group_ids = group_ids.to(device)

            features = model(images)
            all_embeddings.append(features.cpu())

            loss = loss_fn(features, group_ids)
            total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    feature_variance = torch.cat(all_embeddings).var(dim=0).mean().item()

    return avg_loss, feature_variance


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    num_epochs=100,
    checkpoint_path="checkpoints/best_model.pt",
    early_stopping_patience=10,
    warmup_epochs=10
):
    """
    Full training loop for SupCon with early stopping and LR scheduling.
    """
    best_val_loss = float("inf")
    patience_counter = 0
    scheduler = cosine_warmup_scheduler(optimizer, warmup_epochs, num_epochs)

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}")

        train_loss, train_var = train_supcon_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        print(f"Train Loss: {train_loss:.4f}, Feature Variance: {train_var:.4f}")

        val_loss, val_var = validate_supcon_epoch(
            model, val_loader, loss_fn, device
        )
        print(f"Val Loss: {val_loss:.4f}, Feature Variance: {val_var:.4f}")

        scheduler.step()

        if val_loss < best_val_loss:
            print("✅ Saving new best model...")
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")
            if patience_counter >= early_stopping_patience:
                print("⏹️ Early stopping triggered.")
                break
