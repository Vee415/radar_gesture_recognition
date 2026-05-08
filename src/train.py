"""Training loop with MLflow experiment tracking."""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from dataset import get_dataloaders
from model import get_model


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += inputs.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Validate model. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += inputs.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def get_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
    """Linear warmup + cosine annealing scheduler."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1 + __import__("math").cos(__import__("math").pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def train(config: dict) -> None:
    """Main training function with MLflow logging."""
    import mlflow

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        data_dir="data/processed",
        batch_size=config["train"]["batch_size"],
        source=config["data"]["source"],
    )

    # Model
    model = get_model(
        name=config["model"]["name"],
        n_classes=config["model"]["n_classes"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config['model']['name']}, Parameters: {n_params:,}")

    # Training config
    weight_decay = config["train"].get("weight_decay", 1e-4)
    label_smoothing = config["train"].get("label_smoothing", 0.1)
    warmup_epochs = config["train"].get("warmup_epochs", 5)
    patience = config["train"].get("early_stopping_patience", 10)
    epochs = config["train"]["epochs"]

    optimizer = optim.AdamW(model.parameters(), lr=config["train"]["lr"], weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scheduler = get_warmup_scheduler(optimizer, warmup_epochs, epochs)

    # MLflow logging
    mlflow.set_experiment("radar-gesture")

    with mlflow.start_run():
        mlflow.log_params({
            "lr": config["train"]["lr"],
            "batch_size": config["train"]["batch_size"],
            "epochs": epochs,
            "model": config["model"]["name"],
            "n_classes": config["model"]["n_classes"],
            "n_params": n_params,
            "data_source": config["data"]["source"],
            "weight_decay": weight_decay,
            "label_smoothing": label_smoothing,
            "warmup_epochs": warmup_epochs,
        })

        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step()

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch)

            print(
                f"Epoch {epoch+1}/{epochs} "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "models/gesture_model.pth")
                print(f"  -> Saved best model (val_acc: {val_acc:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

        mlflow.log_metric("best_val_acc", best_val_acc)

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to models/gesture_model.pth")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    train(config)