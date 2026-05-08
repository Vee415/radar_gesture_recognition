"""PyTorch Dataset and DataLoader for temporal radar gesture data.

Each sample is a sequence of range-doppler maps of shape (seq_len, 4, 32, 32)
with an integer label. Augmentation applies consistent transforms across all
frames in a sequence (e.g., same horizontal flip for every frame).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class RadarGestureTemporalDataset(Dataset):
    """Radar gesture dataset with frame sequences.

    Each sample is a sequence of range-doppler maps of shape
    (seq_len, 4, 32, 32) with an integer label.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
    ):
        self.data = data
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx].copy()  # (seq_len, 4, 32, 32)
        y = self.labels[idx]

        if self.augment:
            # Random horizontal flip — same flip applied to all frames
            if np.random.rand() > 0.5:
                x = x[:, :, :, ::-1].copy()
            # Gaussian noise per frame (same sigma as single-frame model)
            noise = np.random.normal(0, 0.03, x.shape).astype(np.float32)
            x = np.clip(x + noise, 0, 1)

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def load_temporal_split(
    data_dir: str,
    split: str,
    source: str = "soli",
) -> tuple[np.ndarray, np.ndarray]:
    """Load a temporal data split from .npz file.

    Args:
        data_dir: Directory containing processed .npz files.
        split: One of 'train', 'val', 'test'.
        source: 'soli' or 'simulated'.

    Returns:
        (data, labels) where data has shape (N, seq_len, 4, 32, 32).
    """
    prefix = source if source != "simulated" else "simulated"
    path = Path(data_dir) / f"{prefix}_lstm_{split}.npz"

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    npz = np.load(path)
    return npz["data"], npz["labels"]


def get_temporal_dataloaders(
    data_dir: str = "data/processed",
    batch_size: int = 16,
    source: str = "soli",
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders for temporal data.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_data, train_labels = load_temporal_split(data_dir, "train", source)
    val_data, val_labels = load_temporal_split(data_dir, "val", source)
    test_data, test_labels = load_temporal_split(data_dir, "test", source)

    train_ds = RadarGestureTemporalDataset(train_data, train_labels, augment=True)
    val_ds = RadarGestureTemporalDataset(val_data, val_labels, augment=False)
    test_ds = RadarGestureTemporalDataset(test_data, test_labels, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader