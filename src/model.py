"""Radar gesture classification model architectures."""

import torch
import torch.nn as nn


class RadarGestureCNN(nn.Module):
    """2D CNN for radar gesture classification on range-doppler maps.

    Input:  (batch, 4, 32, 32)  — 4 channels of 32x32 range-doppler
    Output: (batch, n_classes)   — class logits
    """

    def __init__(self, n_classes: int = 12, in_channels: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class RadarGestureCNNV2(nn.Module):
    """CNN V2: original architecture + BatchNorm + global avg pool.

    Keeps the proven 3-conv structure but adds:
    - BatchNorm after every conv (stabilizes training)
    - Global average pooling (reduces FC params)
    - Smaller dropout (0.4 vs 0.5)

    Input:  (batch, 4, 32, 32)
    Output: (batch, n_classes)
    """

    def __init__(self, n_classes: int = 12, in_channels: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def get_model(name: str = "cnn", n_classes: int = 12) -> nn.Module:
    """Factory function to create a model by name."""
    if name == "cnn":
        return RadarGestureCNN(n_classes=n_classes)
    if name == "cnn_v2":
        return RadarGestureCNNV2(n_classes=n_classes)
    raise ValueError(f"Unknown model: {name}. Available: cnn, cnn_v2")