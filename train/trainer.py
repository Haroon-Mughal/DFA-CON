import torch 
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
import math
import os
import sys
from datetime import datetime
import builtins

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
    early_stopping_patience=10,
    warmup_epochs=10,
    warmup_patience=3,
    model_name="resnet50",
    batch_size=32,
    lr=0.5,
    loss_name="supcon",
    checkpoint_dir="checkpoints",
    log_root="logs",
    head_type="mlp"
):
    exp_name = f"{model_name}_bs{batch_size}_lr{lr}_{loss_name}_head_type_{head_type}_warmupDynamic"
    exp_dir = os.path.join(log_root, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_file = open(os.path.join(log_dir, "train_log.txt"), "w")
    builtin_print = builtins.print
    def log_print(*args, **kwargs):
        builtin_print(*args, **kwargs)
        kwargs.pop("file", None)
        builtin_print(*args, file=log_file, **kwargs)
        log_file.flush()

    builtins.print = log_print

    try:
        best_val_loss = float("inf")
        best_epoch = -1
        patience_counter = 0
        warmup_counter = 0
        in_warmup = True

        warmup_scheduler = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-5)

        print("\n🔍 Pre-training evaluation...")
        val_loss, val_var = validate_supcon_epoch(model, val_loader, loss_fn, device)
        print(f"Val Loss (pre-train): {val_loss:.4f}, Feature Variance: {val_var:.4f}")

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

            if in_warmup:
                warmup_scheduler.step()
                if val_loss >= best_val_loss:
                    warmup_counter += 1
                    print(f"🔁 Warmup patience counter: {warmup_counter}/{warmup_patience}")
                    if warmup_counter >= warmup_patience:
                        print("🚀 Switching to Cosine Annealing Scheduler")
                        in_warmup = False
                else:
                    warmup_counter = 0
            else:
                cosine_scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            print(f"📉 Learning Rate: {current_lr:.6f}")

            if val_loss < best_val_loss:
                print("✅ Saving new best model...")
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                filename = f"{model_name}_bs{batch_size}_lr{lr}_{loss_name}_ep{best_epoch}_loss{val_loss:.4f}.pt"
                checkpoint_path = os.path.join(ckpt_dir, filename)
                torch.save(model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")
                if patience_counter >= early_stopping_patience:
                    print("⏹️ Early stopping triggered.")
                    break
    finally:
        log_file.close()
        builtins.print = builtin_print
