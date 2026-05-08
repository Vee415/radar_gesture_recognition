"""Synthetic radar gesture data generator.

Creates simulated range-doppler maps for development/testing
when real Soli data is unavailable.
"""

import numpy as np
from pathlib import Path

GESTURE_PATTERNS = {
    "swipe_left": {"center_x": 10, "center_y": 16, "spread": 6},
    "swipe_right": {"center_x": 22, "center_y": 16, "spread": 6},
    "tap": {"center_x": 16, "center_y": 16, "spread": 3},
    "hold": {"center_x": 16, "center_y": 16, "spread": 5},
    "dismiss": {"center_x": 16, "center_y": 6, "spread": 4},
}

LABEL_MAP = {
    "swipe_left": 0,
    "swipe_right": 1,
    "tap": 2,
    "hold": 3,
    "dismiss": 4,
}


def _generate_range_doppler(
    center_x: int,
    center_y: int,
    spread: int,
    size: int = 32,
    noise_std: float = 0.05,
) -> np.ndarray:
    """Generate a single 32x32 range-doppler map with a Gaussian blob."""
    y, x = np.mgrid[0:size, 0:size]
    blob = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * spread ** 2))
    noise = np.random.normal(0, noise_std, (size, size)).astype(np.float32)
    return np.clip(blob + noise, 0, 1).astype(np.float32)


def simulate_gesture(
    gesture_class: str, n_samples: int = 200, n_channels: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic range-doppler maps for a gesture class.

    Returns:
        (data, labels) where data has shape (n_samples, n_channels, 32, 32).
    """
    pattern = GESTURE_PATTERNS[gesture_class]
    samples = []
    for _ in range(n_samples):
        channels = []
        for ch in range(n_channels):
            # Add slight channel variation
            cx = pattern["center_x"] + np.random.randint(-2, 3)
            cy = pattern["center_y"] + np.random.randint(-2, 3)
            sp = pattern["spread"] + np.random.uniform(-1, 1)
            rd = _generate_range_doppler(cx, cy, sp)
            channels.append(rd)
        sample = np.stack(channels, axis=0)
        samples.append(sample)

    data = np.array(samples, dtype=np.float32)
    label = LABEL_MAP[gesture_class]
    labels = np.full(n_samples, label, dtype=np.int64)
    return data, labels


def generate_all_gestures(
    n_samples_per_class: int = 200, output_dir: str = "data/raw"
) -> None:
    """Generate synthetic data for all gesture classes and save to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_data = []
    all_labels = []

    for gesture_name in GESTURE_PATTERNS:
        print(f"Generating {gesture_name}...")
        data, labels = simulate_gesture(gesture_name, n_samples_per_class)
        all_data.append(data)
        all_labels.append(labels)

    data_arr = np.concatenate(all_data, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)

    # Shuffle
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(data_arr))
    data_arr = data_arr[idx]
    labels_arr = labels_arr[idx]

    # Split 80/10/10
    n = len(data_arr)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    processed_path = Path("data/processed")
    processed_path.mkdir(parents=True, exist_ok=True)

    for split_name, indices in [
        ("train", slice(0, train_end)),
        ("val", slice(train_end, val_end)),
        ("test", slice(val_end, None)),
    ]:
        save_path = processed_path / f"simulated_{split_name}.npz"
        np.savez(
            save_path,
            data=data_arr[indices],
            labels=labels_arr[indices],
        )
        print(f"Saved {split_name}: {len(data_arr[indices])} samples -> {save_path}")

    print(f"Total: {n} samples across {len(GESTURE_PATTERNS)} classes")


if __name__ == "__main__":
    generate_all_gestures()