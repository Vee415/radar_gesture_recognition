# Radar Gesture Pipeline — Technical Reference

Everything a future session needs to know about this project. No fluff, no narrative — just specs, gotchas, and hard-won facts.

---

## Project Overview

End-to-end ML pipeline for 12-class radar gesture recognition on the Google Soli (60 GHz FMCW) dataset. Two models:

1. **Single-frame CNN** — 82.2% accuracy, 0.089 ms/frame (C++ ORT), 621K params
2. **Streaming CNN+LSTM** — 98.0% accuracy, 0.27 ms/frame (C++ ORT), ~4.2M params

The LSTM model is the one worth showcasing. The single-frame CNN exists as the baseline that motivates why temporal modeling matters — pinch gestures look identical in one frame.

---

## Dataset

**Deep-Soli** (ETH Zurich): https://polybox.ethz.ch/index.php/s/wG93iTUdvRU8EaT/download/SoliData.zip (77 MB)

- 5,500 HDF5 files, 12 gesture classes
- Variable-length sequences: 28–145 frames per recording
- 4 channels per frame: range-doppler maps reshaped to (4, 32, 32)
- Per-frame labels (0–11), majority-voted to file-level label
- Split: 80/10/10 with RandomState(42) — same split for both models

Label map:
```
0: pinch_index_finger    4: swipe_left       8: finger_slide_left
1: pinch_pinky            5: swipe_right       9: finger_slide_right
2: pinch_middle           6: swipe_up        10: finger_slide_up
3: pinch_ring             7: swipe_down      11: finger_slide_down
```

Short names used in reports: pinch_index, pinch_pinky, pinch_middle, pinch_ring, swipe_left/right/up/down, slide_left/right/up/down.

---

## Single-Frame CNN

### Architecture

```
Input: (B, 4, 32, 32)
  Conv2d(4→32, 3x3, pad=1, no bias) → BN(32) → ReLU → MaxPool(2)       → (B, 32, 16, 16)
  Conv2d(32→64, 3x3, pad=1, no bias) → BN(64) → ReLU → MaxPool(2)       → (B, 64, 8, 8)
  Conv2d(64→128, 3x3, pad=1, no bias) → BN(128) → ReLU → AdaptiveAvgPool(4,4) → (B, 128, 4, 4)
  Flatten → 2048
  Linear(2048→256) → ReLU → Dropout(0.5) → Linear(256→12)
```

Total params: 621,388. All convs use `bias=False` (BatchNorm provides bias).

### Preprocessing

- Extract middle frame from variable-length HDF5 sequence
- Reshape (n_frames, 1024) → (n_frames, 32, 32) per channel
- Min-max normalize per channel (constant channels → zeros)
- 80/10/10 split, RandomState(42)

### Training (params.yaml)

- LR: 1e-3, batch: 32, epochs: 80 (early stopped at 42)
- AdamW, weight_decay 1e-4, label_smoothing 0.05
- Cosine annealing with 5-epoch linear warmup
- Gradient clipping norm 1.0
- Augmentation: random horizontal flip, Gaussian noise σ=0.03

### Key Findings

- AdaptiveAvgPool(4,4) preserves spatial info → 82%. Global avg pool → 78%. Don't use GAP on this task.
- BatchNorm improved pinch_index from 57% → 77%.
- Label smoothing 0.1 is too aggressive for 12 classes → 78%. Use 0.05.
- 4-conv deeper model: training instability on 4,400 training samples.

---

## Streaming CNN+LSTM

### Architecture

```
Input: (B, 40, 4, 32, 32)
  Per-frame CNN (shared RadarGestureCNN.features backbone)
    (B*40, 4, 32, 32) → CNN features → (B*40, 2048)
  Frame FC compression
    (B*40, 2048) → Linear(2048, 256) → ReLU → (B*40, 256)
  Reshape to sequence
    (B, 40, 256)
  LSTM (unidirectional, 1 layer, hidden=512)
    (B, 40, 256) → (B, 40, 512)
  Per-frame classifier
    Dropout(0.5) → Linear(512, 12) → (B, 40, 12)
  Sequence prediction
    Softmax per frame → mean across time → log → (B, 12)
```

Params: ~4.2M total. CNN backbone alone is 621K.

### Preprocessing (src/temporal/preprocess.py)

- Resample variable-length sequences to exactly 40 frames:
  - Exact length: return as-is
  - Too long (>40): uniform downsample via `np.linspace(0, n-1, 40, dtype=int)`
  - Too short (<40): right-aligned zero-pad (zeros prepended, gesture at end)
- Same per-channel min-max normalization as single-frame
- Same 80/10/10 split, RandomState(42)
- Augmentation: consistent horizontal flip across all frames, Gaussian noise σ=0.03
- **No time reversal** — swipe_left↔swipe_right are directional

### Training (params_lstm.yaml)

- LR: 5e-4 (lower than CNN — LSTM sensitive to LR), batch: 16 (sequences use ~40x more memory)
- Epochs: 100, early stopped at 35 (best val: 98.36%)
- CNN backbone frozen for first 10 epochs, then unfrozen
- Same AdamW, cosine+warmup, label smoothing 0.05, gradient clipping 1.0
- MLflow experiment: `radar-gesture-lstm`
- Saves to `models/gesture_model_lstm.pth`

### Key Findings

- 98.0% test accuracy, 0.9772 macro F1
- Pinch_middle: 56.4% → 94.9% (+38.5pp). Pinch_ring: 63.4% → 87.8% (+24.4pp)
- At 15 frames (375ms at 40fps), already exceeds single-frame CNN's 82.2%
- Unidirectional LSTM is required for real-time streaming (bidirectional needs future frames)

---

## ONNX Export

### Single-Frame CNN

- Opset 17, dynamic batch axis
- Legacy exporter (dynamo=False) — PyTorch 2.3.1 dynamo had Unicode issues on Windows
- Static INT8 quantization with calibration on training data → 620 KB (3.9x compression)
- Dynamic INT8 is 17x slower than FP32 — don't use dynamic quant for small models

### Streaming Split Export (src/temporal/export_streaming.py)

Two separate ONNX models for real-time frame-by-frame inference:

**Feature extractor** (`models/feature_extractor.onnx`, 2,417 KB):
- Input: `frame_input` (1, 4, 32, 32)
- Output: `feature_output` (1, 256)
- Wraps CNN backbone + frame FC compression

**LSTM step** (`models/lstm_step.onnx`, 6,185 KB):
- Inputs: `feature_input` (1, 1, 256), `h0` (1, 1, 512), `c0` (1, 1, 512)
- Outputs: `logits_output` (1, 12), `h1` (1, 1, 512), `c1` (1, 1, 512)
- Wraps LSTM + classifier with explicit hidden state I/O

Verification: streaming predictions match batch model exactly (cosine similarity 1.0).

No dynamic axes on spatial dims — AdaptiveAvgPool2d requires static shapes for ONNX trace.

---

## C++ Inference

### Single-Frame (src/inference_cpp/inference.h/cpp, main.cpp)

- ORT session with `ORT_ENABLE_ALL` graph optimization
- Default thread count (single thread was 2.7x slower)
- FP32 model: 0.089 ms median, 0.111 ms P95
- Static INT8: 0.131 ms P95
- Dynamic INT8: 0.852 ms P95 (don't use this)

### Streaming (src/inference_cpp/streaming_inference.h/cpp, streaming_main.cpp)

- Two ORT sessions: feature_extractor + lstm_step
- Hidden state vectors: `h_state_` and `c_state_`, each 512 floats, reset between gestures
- Accumulated softmax: running average across frames
- Per-frame latency: 0.27 ms median (0.13 ms feature + 0.14 ms LSTM)
- 40-frame sequence: ~11 ms total
- Windows wchar_t path handling via `MultiByteToWideChar`

### Build

```bash
cd src/inference_cpp
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

ORT SDK path: `C:/Users/varun/Projects/radar-gesture/radar_gesture_pipeline/onnxruntime_cpp/onnxruntime-win-x64-1.25.1`

---

## Gotchas & Lessons Learned

### Import collisions

`src/dataset.py` and `src/temporal/dataset.py` both exist. When `src/` is on `sys.path`, `from dataset import ...` resolves to whichever is found first — usually wrong. Fix: use `sys.path.insert(0, project_root)` and `from src.temporal.dataset import ...` style imports. The evaluate.py in temporal is self-contained (copies needed functions) to avoid this entirely.

### NumPy 2.x incompatibility

PyTorch 2.3.1+cu118 was built against NumPy 1.x. `pip install "numpy<2"` to avoid segfaults.

### MLflow schema version mismatch

If switching between venv and venv_gpu (different MLflow versions), delete `mlflow.db` and let it recreate. Schema mismatch causes crashes.

### ONNX export issues

- `dynamo=False` parameter doesn't exist in PyTorch 2.3.1 — just remove it, legacy exporter is default.
- Dynamic axes on spatial dims cause `adaptive_avg_pool2d input size not accessible` — remove dynamic axes, only batch is dynamic.
- Conditional `return_frame_logits` can't be traced — use `ONNXWrapper` class that removes the conditional.

### `np.softmax` doesn't exist

NumPy 1.26.4 doesn't have `np.softmax`. Use `scipy.special.softmax` or implement manually.

### C++ missing includes

`std::setw` and `std::setprecision` need `#include <iomanip>`.

### CMake generator mismatch

If `CMakeCache.txt` has the wrong generator, delete it and re-run. Don't mix VS 18 2026 and VS 17 2022.

### UTF-8 on Windows

`open("file.md", "w")` defaults to cp1252 on Windows. Use `open("file.md", "w", encoding="utf-8")` for any file with Unicode characters (arrows, etc.).

### Right-aligned zero-padding convention

Short sequences are padded with zeros **prepended** (left-pad), not appended. This means the gesture action is right-aligned. `np.concatenate([padding, rd], axis=1)` — padding first, data second. Matches Deep-Soli paper convention.

---

## File Map

```
params.yaml                          # CNN config
params_lstm.yaml                     # CNN+LSTM config

src/preprocess.py                    # Soli download + single-frame .npz
src/dataset.py                       # CNN dataset + augmentation
src/model.py                         # RadarGestureCNN, RadarGestureCNNV2
src/train.py                         # CNN training loop + MLflow
src/evaluate.py                      # CNN evaluation + confusion matrix
src/export.py                        # CNN ONNX export + quantization
src/benchmark_latency.py             # Python latency comparison
src/generate_report.py               # Report generator
src/simulate_data.py                 # Synthetic data for testing

src/temporal/__init__.py
src/temporal/preprocess.py           # Frame sequence extraction from HDF5
src/temporal/dataset.py              # Temporal dataset + sequence augmentation
src/temporal/model.py                # RadarGestureCNNLSTM + build_cnn_lstm()
src/temporal/train.py                # CNN+LSTM training (freeze/unfreeze CNN)
src/temporal/evaluate.py             # CNN+LSTM evaluation (self-contained)
src/temporal/export.py               # Batch ONNX export
src/temporal/export_streaming.py     # Split ONNX: feature_extractor + lstm_step
src/temporal/benchmark_streaming.py  # Python streaming latency benchmark
src/temporal/evaluate_streaming.py   # Streaming ONNX eval on test data

src/inference_cpp/CMakeLists.txt     # Builds both targets
src/inference_cpp/main.cpp           # Single-frame CNN CLI
src/inference_cpp/inference.h/cpp    # Single-frame ORT inference
src/inference_cpp/streaming_main.cpp          # Streaming CNN+LSTM CLI
src/inference_cpp/streaming_inference.h/cpp   # Streaming ORT inference
src/inference_cpp/preprocessor.h/cpp          # Radar data preprocessing
```

---

## Model Files

```
models/gesture_model.pth                # CNN checkpoint (621K params)
models/gesture_model.onnx               # CNN ONNX FP32 (2,430 KB)
models/gesture_model_optimized.onnx     # CNN ONNX graph-fused
models/gesture_model_quant.onnx         # CNN ONNX INT8 dynamic (don't use)
models/gesture_model_static_quant.onnx  # CNN ONNX INT8 static (620 KB)
models/gesture_model_lstm.pth           # CNN+LSTM checkpoint (~4.2M params)
models/gesture_model_lstm.onnx          # CNN+LSTM batch ONNX (8,602 KB)
models/feature_extractor.onnx          # Streaming: CNN→256 (2,417 KB)
models/lstm_step.onnx                  # Streaming: LSTM step (6,185 KB)
```

---

## Data Files

```
data/raw/                              # Soli HDF5 files (5500 files)
data/processed/soli_train.npz          # Single-frame train split
data/processed/soli_val.npz            # Single-frame val split
data/processed/soli_test.npz           # Single-frame test split
data/processed/soli_lstm_train.npz     # Sequence train (N, 40, 4, 32, 32)
data/processed/soli_lstm_val.npz       # Sequence val
data/processed/soli_lstm_test.npz      # Sequence test
```

---

## Reports

```
reports/evaluation_report.md           # CNN per-class accuracy, F1
reports/evaluation_report_lstm.md      # CNN+LSTM evaluation
reports/streaming_evaluation_report.md  # Streaming ONNX evaluation
reports/model_card.md                  # Cross-variant comparison
reports/confusion_matrix.png           # CNN confusion matrix
reports/confusion_matrix_lstm.png      # CNN+LSTM confusion matrix
```

---

## Benchmarking Environment

| Component | Spec |
|-----------|------|
| CPU | Intel Core i9-13900H (14 cores, 24 threads) |
| GPU | NVIDIA GeForce RTX 4060 Laptop 8GB |
| OS | Windows 11 Home |
| ONNX Runtime | 1.25.1 (CPU Execution Provider) |
| C++ Compiler | MSVC 2022 (v143) |
| PyTorch | 2.3.1+cu118 |

All latency numbers (0.089 ms, 0.27 ms, etc.) are measured on this hardware with ORT CPU EP. Numbers on ARM ECUs will differ.

---

## Key Numbers Quick Reference

| Metric | Single-Frame CNN | Streaming CNN+LSTM |
|--------|-----------------|---------------------|
| Accuracy | 82.2% | 98.0% |
| Macro F1 | 0.82 | 0.98 |
| Parameters | 621K | ~4.2M |
| ONNX size | 2,430 KB | 8,602 KB (split: 2,417 + 6,185) |
| C++ per-frame | 0.089 ms | 0.27 ms |
| Full sequence | 0.089 ms | ~11 ms |
| ECU budget | 20-50 ms | 20-50 ms |

Pinch accuracy delta: pinch_middle +38.5pp, pinch_ring +24.4pp, pinch_index +17.2pp.

Streaming accuracy buildup: 1 frame → 57.8%, 10 frames → 84.7%, 20 frames → 93.8%, 40 frames → 98.0%.