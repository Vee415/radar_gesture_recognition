"""Data loading and preprocessing for Deep-Soli radar gesture dataset.

Downloads SoliData from ETH Zurich, loads HDF5 files containing 4-channel
range-doppler maps, and saves as .npz for training.

Each Soli HDF5 file has:
  - ch0..ch3: shape (n_frames, 1024) where 1024 = 32x32 range-doppler
  - label: shape (n_frames, 1) — per-frame gesture label (0-11)
"""

import os
import glob
import urllib.request
import zipfile
from pathlib import Path

import h5py
import numpy as np


SOLI_URL = "https://polybox.ethz.ch/index.php/s/wG93iTUdvRU8EaT/download/SoliData.zip"
SOLI_GESTURE_LABELS = {
    0: "pinch_index_finger",
    1: "pinch_pinky",
    2: "pinch_middle",
    3: "pinch_ring",
    4: "swipe_left",
    5: "swipe_right",
    6: "swipe_up",
    7: "swipe_down",
    8: "finger_slide_left",
    9: "finger_slide_right",
    10: "finger_slide_up",
    11: "finger_slide_down",
}


def download_soli(data_dir: str = "data/raw") -> str:
    """Download and extract Deep-Soli dataset.

    Returns path to extracted data directory.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / "SoliData.zip"
    if not zip_path.exists():
        print(f"Downloading SoliData from {SOLI_URL}...")
        urllib.request.urlretrieve(SOLI_URL, zip_path)
        print(f"Downloaded to {zip_path}")

    extract_dir = data_dir / "SoliData"
    if not extract_dir.exists():
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
        print(f"Extracted to {extract_dir}")

    return str(extract_dir)


def load_soli_h5(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a single Soli HDF5 file.

    Each file has variable-length frames. We reshape (n_frames, 1024)
    to (n_frames, 32, 32) per channel, stack 4 channels, and take
    the majority label across frames as the file-level label.

    Returns:
        (range_doppler, labels) where range_doppler has shape
        (4, 32, 32) and labels has shape (n_frames,).
    """
    with h5py.File(filepath, "r") as f:
        channels = []
        for ch in ["ch0", "ch1", "ch2", "ch3"]:
            # (n_frames, 1024) -> (n_frames, 32, 32)
            data = f[ch][:].astype(np.float32).reshape(-1, 32, 32)
            channels.append(data)

        # (4, n_frames, 32, 32)
        range_doppler = np.stack(channels, axis=0)
        frame_labels = f["label"][:].flatten().astype(np.int64)

    return range_doppler, frame_labels


def normalize(feature_map: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1] per channel.

    Args:
        feature_map: shape (4, 32, 32)
    """
    result = np.empty_like(feature_map)
    for c in range(feature_map.shape[0]):
        ch = feature_map[c]
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max - ch_min > 1e-8:
            result[c] = (ch - ch_min) / (ch_max - ch_min)
        else:
            result[c] = np.zeros_like(ch)
    return result


def preprocess_pipeline(
    raw_dir: str = "data/raw/SoliData",
    processed_dir: str = "data/processed",
) -> None:
    """Load raw Soli .h5 files, extract representative frames, and save as .npz.

    For each file, we take the middle frame and use the majority label
    as the sample label. This gives one (4, 32, 32) sample per file.
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(glob.glob(str(raw_path / "**" / "*.h5"), recursive=True))
    if not h5_files:
        print(f"No .h5 files found in {raw_path}. Run download_soli() first.")
        return

    print(f"Found {len(h5_files)} HDF5 files")

    all_data = []
    all_labels = []
    skipped = 0

    for fpath in h5_files:
        try:
            rd, frame_labels = load_soli_h5(fpath)
            # rd shape: (4, n_frames, 32, 32)

            # Use the majority label as the file label
            majority_label = int(np.bincount(frame_labels).argmax())

            # Take the middle frame as representative
            mid_frame = rd.shape[1] // 2
            sample = rd[:, mid_frame, :, :]  # (4, 32, 32)

            all_data.append(sample)
            all_labels.append(majority_label)
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"Skipping {fpath}: {e}")

    if skipped > 5:
        print(f"... skipped {skipped} files total")

    data_arr = np.array(all_data, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.int64)

    # Normalize each sample
    data_arr = np.array([normalize(d) for d in data_arr], dtype=np.float32)

    print(f"Data shape: {data_arr.shape}, Labels shape: {labels_arr.shape}")
    print(f"Classes: {sorted(np.unique(labels_arr))}")
    print(f"Samples per class:")
    for c in sorted(np.unique(labels_arr)):
        name = SOLI_GESTURE_LABELS.get(c, f"class_{c}")
        count = (labels_arr == c).sum()
        print(f"  {c}: {name} = {count}")

    # Split: 80/10/10
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(data_arr))
    n = len(data_arr)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        save_path = processed_path / f"soli_{split_name}.npz"
        np.savez(
            save_path,
            data=data_arr[idx],
            labels=labels_arr[idx],
        )
        print(f"Saved {split_name}: {len(idx)} samples -> {save_path}")


if __name__ == "__main__":
    raw_dir = download_soli()
    preprocess_pipeline(raw_dir=raw_dir)