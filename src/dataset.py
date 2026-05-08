"""PyTorch Dataset and DataLoader for radar gesture data."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class RadarGestureDataset(Dataset):
    """Radar gesture dataset from preprocessed .npz files.

    Each sample is a range-doppler map of shape (4, 32, 32)
    with an integer label in [0, n_classes).
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
        x = self.data[idx].copy()
        y = self.labels[idx]

        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                x = x[:, :, ::-1].copy()
            # Random Gaussian noise
            noise = np.random.normal(0, 0.03, x.shape).astype(np.float32)
            x = np.clip(x + noise, 0, 1)

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def load_split(data_dir: str, split: str, source: str = "soli") -> tuple[np.ndarray, np.ndarray]:
    """Load a data split from .npz file.

    Args:
        data_dir: Directory containing processed .npz files.
        split: One of 'train', 'val', 'test'.
        source: 'soli' or 'simulated'.

    Returns:
        (data, labels) arrays.
    """
    from pathlib import Path

    prefix = source if source != "simulated" else "simulated"
    path = Path(data_dir) / f"{prefix}_{split}.npz"

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    npz = np.load(path)
    return npz["data"], npz["labels"]


def get_dataloaders(
    data_dir: str = "data/processed",
    batch_size: int = 32,
    source: str = "soli",
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_data, train_labels = load_split(data_dir, "train", source)
    val_data, val_labels = load_split(data_dir, "val", source)
    test_data, test_labels = load_split(data_dir, "test", source)

    train_ds = RadarGestureDataset(train_data, train_labels, augment=True)
    val_ds = RadarGestureDataset(val_data, val_labels, augment=False)
    test_ds = RadarGestureDataset(test_data, test_labels, augment=False)

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