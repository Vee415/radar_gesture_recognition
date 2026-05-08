"""Model evaluation: metrics, confusion matrix, and report generation."""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import get_dataloaders, load_split
from model import get_model


# Soli gesture labels (12 classes)
GESTURE_NAMES = [
    "pinch_index", "pinch_pinky", "pinch_middle", "pinch_ring",
    "swipe_left", "swipe_right", "swipe_up", "swipe_down",
    "slide_left", "slide_right", "slide_up", "slide_down",
]

# Simulated gesture labels (5 classes)
SIMULATED_NAMES = ["swipe_left", "swipe_right", "tap", "hold", "dismiss"]


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Run evaluation on test set. Returns metrics dict."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "per_class_accuracy": per_class_acc,
        "predictions": all_preds,
        "labels": all_labels,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: str,
) -> None:
    """Plot and save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title="Confusion Matrix",
           ylabel="True label",
           xlabel="Predicted label")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def generate_report(
    metrics: dict,
    model_info: dict,
    output_dir: str = "reports",
    class_names: list[str] | None = None,
) -> None:
    """Write evaluation report as markdown."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = GESTURE_NAMES

    cm = metrics["confusion_matrix"]
    report = f"""# Evaluation Report

## Model
- Architecture: {model_info.get('name', 'RadarGestureCNN')}
- Parameters: {model_info.get('n_params', 'N/A'):,}
- Input: (batch, {model_info.get('in_channels', 4)}, 32, 32)
- Classes: {model_info.get('n_classes', len(class_names))}

## Metrics

| Metric | Value |
|--------|-------|
| Accuracy | {metrics['accuracy']:.4f} |
| Macro F1 | {metrics['macro_f1']:.4f} |

## Per-Class Accuracy

| Class | Name | Accuracy |
|-------|------|----------|
"""
    for i, name in enumerate(class_names):
        if i < len(metrics["per_class_accuracy"]):
            acc = metrics["per_class_accuracy"][i]
            report += f"| {i} | {name} | {acc:.4f} |\n"

    report += f"""
## Confusion Matrix

See `confusion_matrix.png` in this directory.

## Classification Report

```
{classification_report(metrics['labels'], metrics['predictions'], target_names=class_names[:len(np.unique(metrics['labels']))])}
```
"""
    report_path = Path(output_dir) / "evaluation_report.md"
    report_path.write_text(report)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    import yaml

    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(
        data_dir="data/processed",
        batch_size=config["train"]["batch_size"],
        source=config["data"]["source"],
    )

    # Determine class names
    n_classes = config["model"]["n_classes"]
    class_names = SIMULATED_NAMES if config["data"]["source"] == "simulated" else GESTURE_NAMES

    model = get_model(
        name=config["model"]["name"],
        n_classes=n_classes,
    ).to(device)
    model.load_state_dict(torch.load("models/gesture_model.pth", map_location=device))

    metrics = evaluate_model(model, test_loader, device)

    n_params = sum(p.numel() for p in model.parameters())
    model_info = {
        "name": config["model"]["name"],
        "n_params": n_params,
        "in_channels": 4,
        "n_classes": n_classes,
    }

    plot_confusion_matrix(metrics["confusion_matrix"], class_names, "reports/confusion_matrix.png")
    generate_report(metrics, model_info, class_names=class_names)

    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")