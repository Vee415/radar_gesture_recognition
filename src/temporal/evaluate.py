"""Evaluation for temporal CNN+LSTM model.

Self-contained — does not import from src.evaluate to avoid
module name collisions between src/dataset.py and src/temporal/dataset.py.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import yaml
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.temporal.dataset import get_temporal_dataloaders
from src.temporal.model import build_cnn_lstm


GESTURE_NAMES = [
    "pinch_index", "pinch_pinky", "pinch_middle", "pinch_ring",
    "swipe_left", "swipe_right", "swipe_up", "swipe_down",
    "slide_left", "slide_right", "slide_up", "slide_down",
]


def evaluate_model(model, test_loader, device):
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


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title="Confusion Matrix (CNN+LSTM)",
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


if __name__ == "__main__":
    with open("params_lstm.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    _, _, test_loader = get_temporal_dataloaders(
        data_dir="data/processed",
        batch_size=config["train"]["batch_size"],
        source=config["data"]["source"],
    )

    # Model
    model = build_cnn_lstm(
        cnn_backbone=config["model"]["cnn_backbone"],
        n_classes=config["model"]["n_classes"],
        lstm_hidden_size=config["model"]["lstm_hidden_size"],
        lstm_num_layers=config["model"]["lstm_num_layers"],
        lstm_dropout=config["model"]["lstm_dropout"],
        fc_hidden_size=config["model"]["fc_hidden_size"],
    ).to(device)

    model.load_state_dict(torch.load("models/gesture_model_lstm.pth", map_location=device, weights_only=True))
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())

    # Evaluate
    metrics = evaluate_model(model, test_loader, device)

    # Confusion matrix
    os.makedirs("reports", exist_ok=True)
    plot_confusion_matrix(metrics["confusion_matrix"], GESTURE_NAMES, "reports/confusion_matrix_lstm.png")

    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    print("-" * 40)
    for i, name in enumerate(GESTURE_NAMES):
        if i < len(metrics["per_class_accuracy"]):
            print(f"  {name:20s}: {metrics['per_class_accuracy'][i]:.4f}")

    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        metrics["labels"], metrics["predictions"],
        target_names=GESTURE_NAMES[:len(np.unique(metrics["labels"]))]
    ))