"""Temporal preprocessing: extract frame sequences from Soli HDF5 files.

Unlike the original preprocess.py which takes only the middle frame,
this script keeps all frames from each recording, resamples to a fixed
sequence length, and saves as (N, seq_len, 4, 32, 32) .npz files.

Resampling strategy (matches Deep-Soli UIST 2016 paper):
  - Short sequences (<seq_len): right-pad with zero frames
  - Long sequences (>seq_len): uniformly downsample
  - Exact length: use as-is
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import glob

import numpy as np

from src.preprocess import download_soli, load_soli_h5, normalize


def resample_sequence(rd: np.ndarray, target_len: int = 40) -> np.ndarray:
    """Resample a frame sequence to target_len frames.

    Args:
        rd: shape (4, n_frames, 32, 32) — full frame sequence from one HDF5 file.
        target_len: desired sequence length.

    Returns:
        np.ndarray of shape (4, target_len, 32, 32).
    """
    n_frames = rd.shape[1]

    if n_frames == target_len:
        return rd

    if n_frames > target_len:
        # Uniform downsample: pick evenly-spaced frame indices
        indices = np.linspace(0, n_frames - 1, target_len, dtype=int)
        return rd[:, indices, :, :]

    # Short sequence: right-pad with zero frames
    # Gesture action is at the end (right-aligned), matching Deep-Soli paper
    pad_len = target_len - n_frames
    padding = np.zeros((4, pad_len, 32, 32), dtype=rd.dtype)
    return np.concatenate([padding, rd], axis=1)


def preprocess_temporal(
    raw_dir: str = "data/raw/SoliData",
    processed_dir: str = "data/processed",
    seq_len: int = 40,
) -> None:
    """Load raw Soli .h5 files, extract frame sequences, and save as .npz.

    For each file, all frames are kept and resampled to seq_len frames.
    Saves three files:
      data/processed/soli_lstm_train.npz
      data/processed/soli_lstm_val.npz
      data/processed/soli_lstm_test.npz

    Where 'data' has shape (N, seq_len, 4, 32, 32).
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

            # Majority label as file-level label
            majority_label = int(np.bincount(frame_labels).argmax())

            # Resample to fixed sequence length
            rd_resampled = resample_sequence(rd, target_len=seq_len)
            # Shape: (4, seq_len, 32, 32)

            # Normalize each frame in the sequence
            normalized_frames = np.stack([
                normalize(rd_resampled[:, t, :, :]) for t in range(seq_len)
            ], axis=0)
            # Shape: (seq_len, 4, 32, 32)

            all_data.append(normalized_frames)
            all_labels.append(majority_label)
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"Skipping {fpath}: {e}")

    if skipped > 5:
        print(f"... skipped {skipped} files total")

    data_arr = np.array(all_data, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.int64)

    print(f"Data shape: {data_arr.shape}, Labels shape: {labels_arr.shape}")
    print(f"Classes: {sorted(np.unique(labels_arr))}")

    # 80/10/10 split — same RandomState(42) as original for identical splits
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(data_arr))
    n = len(data_arr)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        save_path = processed_path / f"soli_lstm_{split_name}.npz"
        np.savez(
            save_path,
            data=data_arr[idx],
            labels=labels_arr[idx],
        )
        print(f"Saved {split_name}: {len(idx)} samples -> {save_path}")


if __name__ == "__main__":
    raw_dir = download_soli()
    preprocess_temporal(raw_dir=raw_dir)