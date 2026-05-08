# Radar Gesture Recognition Pipeline

End-to-end ML pipeline for gesture classification on radar sensor data, targeting automotive ECU edge deployment via ONNX Runtime.

```
Raw Radar Data (Soli HDF5)
        ↓
  Preprocessing
  - Middle-frame extraction
  - 4-channel range-doppler reshape (32x32)
  - Min-max normalization per channel
        ↓
  PyTorch CNN (BatchNorm, 621K params)
        ↓
  MLflow Experiment Tracking
        ↓
  ONNX Export + Static INT8 Quantization
        ↓
  C++ ONNX Runtime Inference (0.089ms)
        ↓
  Gesture Label + Confidence
```

---

## Results

### Accuracy

| Metric | Value |
|--------|-------|
| Overall accuracy | **82.2%** |
| Macro F1 | **0.82** |
| Parameters | 621,388 |

**Per-class accuracy:**

| Gesture | Accuracy | Type |
|---------|----------|------|
| slide_down | 100.0% | Swipe/Slide |
| swipe_down | 92.5% | Swipe/Slide |
| swipe_right | 91.2% | Swipe/Slide |
| swipe_up | 91.3% | Swipe/Slide |
| slide_up | 85.4% | Swipe/Slide |
| slide_right | 84.9% | Swipe/Slide |
| pinch_pinky | 84.4% | Pinch |
| swipe_left | 79.7% | Swipe/Slide |
| slide_left | 76.1% | Swipe/Slide |
| pinch_index | 77.1% | Pinch |
| pinch_ring | 63.4% | Pinch |
| pinch_middle | 56.4% | Pinch |

**Swipe/slide gestures average ~90%; pinch gestures average ~70%.** Pinch classes are harder because they involve subtle finger motions that produce similar range-doppler signatures.

### Model Size & Latency Comparison

| Model | Accuracy | Size (KB) | Compression | C++ P95 Latency |
|-------|----------|-----------|-------------|-----------------|
| PyTorch FP32 | 82.2% | 2,437 | 1.0x | — |
| ONNX FP32 | 82.2% | 2,430 | 1.0x | 0.111 ms |
| **ONNX INT8 (static)** | **82.4%** | **620** | **3.9x** | **0.131 ms** |
| ONNX INT8 (dynamic) | 81.5% | 619 | 3.9x | 0.852 ms |

All C++ latencies are well within a typical ECU real-time budget of 10 ms.

---

## Engineering Decisions

### Data Preprocessing

| Decision | Rationale |
|----------|-----------|
| **Middle-frame extraction** from variable-length HDF5 | Start/end frames contain gesture transitions, not stable gesture signatures |
| **Majority-vote label per file** | Handles per-frame label noise within a single recording |
| **Reshape (n_frames, 1024) → (n_frames, 32, 32)** | 1024 = 32×32 range-doppler map as defined by Soli sensor |
| **Min-max normalization per channel** | Different radar channels have different scales; per-channel normalization handles this; constant channels zeroed out |
| **80/10/10 split with RandomState(42)** | Reproducible experiments across runs |

### Model Architecture

| Decision | Rationale |
|----------|-----------|
| **3-conv CNN** (4→32→64→128 channels) | Sufficient capacity for 32×32 range-doppler maps; deeper models didn't improve accuracy on this small dataset |
| **BatchNorm after every conv** | Stabilizes training, allows higher learning rates, improved pinch_index from 57% → 77% |
| **`bias=False` on all convolutions** | BatchNorm already provides bias; removing it reduces params and avoids redundancy |
| **AdaptiveAvgPool2d(4,4)** instead of global avg pool | Preserves spatial information for the classifier; global avg pool lost too much detail and dropped accuracy to 78% |
| **Dropout 0.5** in classifier | Strong regularization needed for 4,400 training sample dataset |
| **FC head: 2048→256→12** | Larger FC than V2's 128→12; the 2048→256 hidden layer was necessary to maintain accuracy |

**What we tried and rejected:**
- **Global average pooling (V2)**: Reduced FC from 526K → 1.5K params, but accuracy dropped from 82% → 78%. The spatial information in the 4×4 feature map matters for distinguishing pinch gestures.
- **4-conv deeper model**: Added a 4th conv block (128→128→128→256), but the small dataset (4,400 samples) couldn't support the extra capacity — training became unstable.

### Training Recipe

| Decision | Rationale |
|----------|-----------|
| **AdamW** over Adam | Proper decoupled weight decay (1e-4) prevents overfitting on small dataset |
| **Linear warmup (5 epochs)** | Prevents early training instability when cosine schedule starts at high LR |
| **Cosine annealing** over step decay | Smoother LR decay, better final accuracy |
| **Label smoothing (0.05)** | Prevents overconfident predictions; helps hard classes (pinch). Tested 0.1 first — too aggressive for 12-class problem |
| **Gradient clipping (norm=1.0)** | Prevents exploding gradients during early training |
| **Early stopping (patience=10)** | Avoids overfitting; actual training stopped at epoch 42 of 80 max |
| **MLflow tracking** | Full experiment reproducibility; logs hyperparams, per-epoch metrics, best model |

**What we tried and rejected:**
- **Label smoothing 0.1**: Too aggressive for 12-class problem; val accuracy dropped to 78%
- **Vertical flip augmentation**: Doesn't make physical sense for range-doppler maps (range axis is not symmetric)
- **Random erasing augmentation**: Too destructive for small dataset; caused training instability
- **Strong noise (σ=0.05)**: Hurt more than helped; σ=0.03 was the sweet spot

### ONNX Export & Quantization

| Decision | Rationale |
|----------|-----------|
| **Opset 17** | Modern opset with full operator support |
| **Dynamic batch axis** | Enables variable batch sizes at inference without re-export |
| **Legacy exporter (`dynamo=False`)** | PyTorch 2.x dynamo exporter had Unicode encoding issues on Windows |
| **Static INT8 quantization** over dynamic | Dynamic quant dequantizes at runtime — 17× slower on small models. Static pre-computes scales with calibration data, no runtime overhead |
| **QDQ quant format** (static) | Explicit Quantize/DeQuantize nodes enable better ORT graph optimization |
| **Calibration with real training data** | Produces accurate activation scale/zero-point for static quantization |
| **ONNX graph optimization** (ORT_ENABLE_ALL) | Fuses BatchNorm→Conv, removes identity nodes, folds constants. Reduced graph from 20 → 10 nodes |

**Key finding: Dynamic INT8 quantization is the wrong choice for small models.** Despite identical 3.9× compression, dynamic quant is 17× slower than FP32 in Python (0.701ms vs 0.041ms P95) because runtime dequantization overhead dominates. Static quantization eliminates this overhead and is the only viable INT8 option for deployment.

### C++ Inference

| Decision | Rationale |
|----------|-----------|
| **ORT_ENABLE_ALL** session option | Applies graph optimizations at session creation (one-time cost), fuses Conv+BN+ReLU into single ops |
| **Default thread count** (not pinned to 1) | Small model benefits from ORT's default multi-thread pipeline; single thread was 2.7× slower |
| **Sequential execution NOT used** | PARALLEL (default) was faster for this model — pipeline parallelism helps even on small graphs |

**What we tried and rejected:**
- **SetIntraOpNumThreads(1)**: Made FP32 latency 2.7× worse (0.256ms vs 0.094ms). Thread scheduling overhead is less than the parallelism benefit
- **ORT_SEQUENTIAL execution**: Slower than default PARALLEL mode for single-request latency on this model
- **NCHW16c layout optimization** (ORT_ENABLE_ALL on pre-optimized file): Optimized for AVX512 but slower on our CPU

---

## Dataset

**Deep-Soli** (ETH Zurich) — Google Soli radar chip, 60 GHz

- 5,500 samples across 12 gesture classes
- HDF5 format: 4 channels of range-doppler maps per sample
- Each channel: (n_frames, 1024) reshaped to (n_frames, 32, 32)
- Per-frame labels (0-11), majority-voted to file-level label
- Classes: 4 pinch + 4 swipe + 4 slide gestures

Download: [SoliData.zip](https://polybox.ethz.ch/index.php/s/wG93iTUdvRU8EaT/download/SoliData.zip) (77 MB)

---

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Download and preprocess Soli data
python src/preprocess.py

# Or use simulated data for quick testing
python src/simulate_data.py

# Train
python src/train.py

# Evaluate
python src/evaluate.py

# Export ONNX + quantize
python src/export.py

# Latency benchmark (Python)
python src/benchmark_latency.py

# Full metrics report
python src/generate_report.py
```

### C++ Inference

Requires [ONNX Runtime C++ SDK](https://github.com/microsoft/onnxruntime/releases):

```bash
cd src/inference_cpp
mkdir build && cd build
cmake .. -G "Visual Studio 18 2026" -A x64
cmake --build . --config Release

# Run benchmark
./Release/gesture_inference.exe ../../models/gesture_model.onnx 1000
```

---

## Project Structure

```
├── data/
│   ├── raw/                        # Soli HDF5 files
│   └── processed/                  # .npz train/val/test splits
├── models/
│   ├── gesture_model.pth           # PyTorch checkpoint
│   ├── gesture_model.onnx          # ONNX FP32
│   ├── gesture_model_optimized.onnx # ONNX FP32 (graph-fused)
│   ├── gesture_model_quant.onnx    # ONNX INT8 dynamic
│   └── gesture_model_static_quant.onnx  # ONNX INT8 static
├── reports/
│   ├── evaluation_report.md        # Per-class accuracy, F1
│   ├── model_card.md               # Cross-variant comparison
│   └── confusion_matrix.png
├── src/
│   ├── preprocess.py               # Soli download + HDF5 → .npz
│   ├── simulate_data.py            # Synthetic data generator
│   ├── dataset.py                  # PyTorch Dataset + augmentation
│   ├── model.py                    # RadarGestureCNN + V2 variant
│   ├── train.py                    # Training loop + MLflow
│   ├── evaluate.py                 # Metrics + confusion matrix
│   ├── export.py                   # ONNX export + quantization
│   ├── benchmark_latency.py        # Python latency comparison
│   ├── generate_report.py          # Metrics report generator
│   └── inference_cpp/
│       ├── CMakeLists.txt
│       ├── main.cpp
│       ├── inference.h/cpp
│       └── preprocessor.h/cpp
├── params.yaml
├── requirements.txt
└── Dockerfile
```

---

## Stack

| Component | Tool |
|-----------|------|
| Model training | PyTorch |
| Experiment tracking | MLflow |
| Model export | ONNX (opset 17) |
| Quantization | ONNX Runtime (static INT8) |
| Edge inference | ONNX Runtime C++ |
| Containerization | Docker |
| Evaluation | scikit-learn |