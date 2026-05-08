# Radar Gesture Recognition Pipeline

End-to-end ML pipeline for gesture classification on radar sensor data, targeting automotive ECU edge deployment via ONNX Runtime. Two models compared: a single-frame CNN baseline and a streaming CNN+LSTM that achieves **98.0% accuracy** by capturing temporal motion signatures.

```
Raw Radar Data (Soli HDF5)
        ↓
  Preprocessing
  - Single-frame: middle-frame extraction
  - Temporal: resample to 40-frame sequences
  - 4-channel range-doppler reshape (32x32)
  - Min-max normalization per channel
        ↓
  ┌─────────────────────┐    ┌──────────────────────────┐
  │  Single-Frame CNN   │    │  Streaming CNN+LSTM       │
  │  621K params        │    │  Per-frame CNN (shared)   │
  │  82.2% accuracy      │    │  → LSTM (512 hidden)      │
  │  0.089 ms/frame     │    │  → Accumulated softmax     │
  │                     │    │  98.0% accuracy             │
  │                     │    │  0.27 ms/frame (C++)       │
  └─────────────────────┘    └──────────────────────────┘
        ↓                            ↓
  ONNX Export + Quantization   ONNX Split Export
        ↓                       (feature_extractor + lstm_step)
  C++ ONNX Runtime                   ↓
        ↓                      C++ Streaming Inference
  Gesture Label                (hidden state across frames)
```

---

## Results

### Single-Frame CNN Baseline

| Metric | Value |
|--------|-------|
| Overall accuracy | 82.2% |
| Macro F1 | 0.82 |
| Parameters | 621,388 |

### Streaming CNN+LSTM

| Metric | Value |
|--------|-------|
| Overall accuracy | **98.0%** |
| Macro F1 | **0.98** |
| Parameters | 4.2M (CNN backbone + LSTM + classifier) |

**Per-class comparison:**

| Gesture | Single-Frame CNN | Streaming CNN+LSTM | Delta |
|---------|-----------------|---------------------|-------|
| pinch_middle | 56.4% | **94.9%** | +38.5pp |
| pinch_ring | 63.4% | **87.8%** | +24.4pp |
| slide_left | 76.1% | **100.0%** | +23.9pp |
| pinch_index | 77.1% | **94.3%** | +17.2pp |
| pinch_pinky | 84.4% | **100.0%** | +15.6pp |
| slide_right | 84.9% | **100.0%** | +15.1pp |
| swipe_left | 79.7% | **100.0%** | +20.3pp |
| slide_up | 85.4% | **95.8%** | +10.4pp |
| swipe_right | 91.2% | **100.0%** | +8.8pp |
| swipe_up | 91.3% | **100.0%** | +8.7pp |
| swipe_down | 92.5% | **100.0%** | +7.5pp |
| slide_down | 100.0% | **100.0%** | +0.0pp |

**Pinch gestures improve by +24pp on average** — the LSTM captures the temporal dynamics of finger motion that a single frame cannot distinguish.

### Streaming Accuracy Buildup

The LSTM accumulates evidence over frames. Accuracy improves rapidly:

| Frames | Accuracy | Macro F1 |
|--------|----------|----------|
| 1 | 57.8% | 0.57 |
| 5 | 76.6% | 0.76 |
| 10 | 84.7% | 0.84 |
| 15 | 90.4% | 0.90 |
| 20 | 93.8% | 0.93 |
| 30 | 96.9% | 0.96 |
| 40 | 98.0% | 0.98 |

At 15 frames (375ms at 40fps), the model already exceeds the single-frame CNN's final accuracy.

### Model Size & Latency Comparison

**Single-Frame CNN:**

| Model | Accuracy | Size (KB) | Compression | C++ P95 Latency |
|-------|----------|-----------|-------------|-----------------|
| PyTorch FP32 | 82.2% | 2,437 | 1.0x | — |
| ONNX FP32 | 82.2% | 2,430 | 1.0x | 0.111 ms |
| **ONNX INT8 (static)** | **82.4%** | **620** | **3.9x** | **0.131 ms** |
| ONNX INT8 (dynamic) | 81.5% | 619 | 3.9x | 0.852 ms |

**Streaming CNN+LSTM:**

| Component | Size (KB) | C++ Latency |
|-----------|-----------|-------------|
| Feature extractor ONNX | 2,417 | 0.13 ms |
| LSTM step ONNX | 6,185 | 0.14 ms |
| **Total per frame** | **8,602** | **0.27 ms** |
| Full sequence (40 frames) | — | ~11 ms |

All C++ latencies are well within a typical ECU real-time budget of 20-50 ms.

### Benchmarking Environment

| Component | Spec |
|-----------|------|
| CPU | Intel Core i9-13900H (14 cores, 24 threads) |
| GPU | NVIDIA GeForce RTX 4060 Laptop 8GB |
| OS | Windows 11 Home |
| ONNX Runtime | 1.25.1 (CPU Execution Provider) |
| C++ Compiler | MSVC 2022 (v143) |
| PyTorch | 2.3.1+cu118 |

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

**Temporal preprocessing (CNN+LSTM):**

| Decision | Rationale |
|----------|-----------|
| **Resample to 40 frames** (right-aligned zero-pad for short, uniform downsample for long) | Matches Deep-Soli paper; right-padding preserves gesture endpoint which carries discriminative motion info |
| **Consistent horizontal flip across all frames** | A flip applied to one frame must apply to all frames in the sequence; otherwise the temporal coherence is destroyed |
| **No time reversal augmentation** | swipe_left ↔ swipe_right are directional; reversing frames would change the gesture label |

### Model Architecture

**Single-Frame CNN:**

| Decision | Rationale |
|----------|-----------|
| **3-conv CNN** (4→32→64→128 channels) | Sufficient capacity for 32×32 range-doppler maps; deeper models didn't improve accuracy on this small dataset |
| **BatchNorm after every conv** | Stabilizes training, allows higher learning rates, improved pinch_index from 57% → 77% |
| **`bias=False` on all convolutions** | BatchNorm already provides bias; removing it reduces params and avoids redundancy |
| **AdaptiveAvgPool2d(4,4)** instead of global avg pool | Preserves spatial information for the classifier; global avg pool lost too much detail and dropped accuracy to 78% |
| **Dropout 0.5** in classifier | Strong regularization needed for 4,400 training sample dataset |
| **FC head: 2048→256→12** | Larger FC than V2's 128→12; the 2048→256 hidden layer was necessary to maintain accuracy |

**Streaming CNN+LSTM:**

| Decision | Rationale |
|----------|-----------|
| **CNN backbone reuse** | Same RadarGestureCNN.features backbone; shared feature extraction ensures consistent input to LSTM |
| **Frame FC: 2048→256 + ReLU** | Compresses CNN features before LSTM; 2048-dim input to LSTM would be wasteful |
| **LSTM (1 layer, hidden=512)** | Matches Deep-Soli paper architecture; 1 layer sufficient for 40-frame sequences |
| **Unidirectional LSTM** | Real-time streaming requires causal processing; bidirectional would need future frames |
| **Per-frame classifier** | Linear(512→12) on every LSTM output; softmax then averaged across time for sequence prediction |
| **CNN frozen for first 10 epochs** | Pretrained CNN features stabilize early LSTM training; unfreezing after epoch 10 allows fine-tuning |

**What we tried and rejected:**
- **Global average pooling (V2)**: Reduced FC from 526K → 1.5K params, but accuracy dropped from 82% → 78%. The spatial information in the 4×4 feature map matters for distinguishing pinch gestures.
- **4-conv deeper model**: Added a 4th conv block (128→128→128→256), but the small dataset (4,400 samples) couldn't support the extra capacity — training became unstable.
- **Bidirectional LSTM**: Would require future frames, incompatible with streaming/real-time deployment.

### Training Recipe

**Single-Frame CNN:**

| Decision | Rationale |
|----------|-----------|
| **AdamW** over Adam | Proper decoupled weight decay (1e-4) prevents overfitting on small dataset |
| **Linear warmup (5 epochs)** | Prevents early training instability when cosine schedule starts at high LR |
| **Cosine annealing** over step decay | Smoother LR decay, better final accuracy |
| **Label smoothing (0.05)** | Prevents overconfident predictions; helps hard classes (pinch). Tested 0.1 first — too aggressive for 12-class problem |
| **Gradient clipping (norm=1.0)** | Prevents exploding gradients during early training |
| **Early stopping (patience=10)** | Avoids overfitting; actual training stopped at epoch 42 of 80 max |
| **MLflow tracking** | Full experiment reproducibility; logs hyperparams, per-epoch metrics, best model |

**CNN+LSTM:**

| Decision | Rationale |
|----------|-----------|
| **Lower LR (5e-4)** than CNN (1e-3) | LSTM is more sensitive to learning rate; 1e-3 caused training divergence |
| **Batch size 16** (vs 32 for CNN) | Sequences use ~40× more memory per sample; smaller batch avoids OOM |
| **CNN frozen for first 10 epochs** | Stabilizes early LSTM training with pretrained CNN features; unfreezing allows fine-tuning |
| **Cosine annealing with warmup (5 epochs)** | Same recipe as CNN; proven effective for this dataset |
| **Label smoothing (0.05)** | Consistent with CNN training |
| **Early stopping (patience=15)** | Slightly more patient than CNN (15 vs 10); temporal model benefits from longer training |
| **Early stopped at epoch 35** | Best val accuracy 98.36% |

**What we tried and rejected:**
- **Label smoothing 0.1**: Too aggressive for 12-class problem; val accuracy dropped to 78%
- **Vertical flip augmentation**: Doesn't make physical sense for range-doppler maps (range axis is not symmetric)
- **Random erasing augmentation**: Too destructive for small dataset; caused training instability
- **Strong noise (σ=0.05)**: Hurt more than helped; σ=0.03 was the sweet spot

### ONNX Export & Quantization

| Decision | Rationale |
|----------|-----------|
| **Opset 17** | Modern opset with full operator support |
| **Dynamic batch axis** (single-frame CNN) | Enables variable batch sizes at inference without re-export |
| **Legacy exporter (`dynamo=False`)** | PyTorch 2.x dynamo exporter had Unicode encoding issues on Windows |
| **Static INT8 quantization** over dynamic | Dynamic quant dequantizes at runtime — 17× slower on small models. Static pre-computes scales with calibration data, no runtime overhead |
| **QDQ quant format** (static) | Explicit Quantize/DeQuantize nodes enable better ORT graph optimization |
| **Calibration with real training data** | Produces accurate activation scale/zero-point for static quantization |
| **ONNX graph optimization** (ORT_ENABLE_ALL) | Fuses BatchNorm→Conv, removes identity nodes, folds constants. Reduced graph from 20 → 10 nodes |

### Streaming ONNX Export

| Decision | Rationale |
|----------|-----------|
| **Two-model split** (feature_extractor + lstm_step) | Enables frame-by-frame processing with explicit hidden state management; no dynamic sequence length needed |
| **Feature extractor**: (1,4,32,32) → (1,256) | CNN + FC compression; runs once per frame |
| **LSTM step**: (1,1,256) + h0(1,1,512) + c0(1,1,512) → logits(1,12) + h1 + c1 | Explicit hidden state I/O for streaming; reset between gestures |
| **Accumulated softmax averaging** | Running average of per-frame softmax probabilities; provides early predictions and stable confidence estimates |
| **No dynamic axes** | Fixed shapes enable ORT_ENABLE_ALL graph optimizations; AdaptiveAvgPool2d requires static spatial dims |

**Key finding: Dynamic INT8 quantization is the wrong choice for small models.** Despite identical 3.9× compression, dynamic quant is 17× slower than FP32 in Python (0.701ms vs 0.041ms P95) because runtime dequantization overhead dominates. Static quantization eliminates this overhead and is the only viable INT8 option for deployment.

### C++ Inference

| Decision | Rationale |
|----------|-----------|
| **ORT_ENABLE_ALL** session option | Applies graph optimizations at session creation (one-time cost), fuses Conv+BN+ReLU into single ops |
| **Default thread count** (not pinned to 1) | Small model benefits from ORT's default multi-thread pipeline; single thread was 2.7× slower |
| **Sequential execution NOT used** | PARALLEL (default) was faster for this model — pipeline parallelism helps even on small graphs |
| **Streaming architecture** | Two ORT sessions (feature_extractor + lstm_step); hidden state vectors maintained across frames, reset between gestures |
| **Windows wchar_t path handling** | ORT requires wide strings on Windows; `MultiByteToWideChar` conversion for model paths |

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

# Download and preprocess Soli data (single-frame)
python src/preprocess.py

# Or use simulated data for quick testing
python src/simulate_data.py

# --- Single-Frame CNN ---
python src/train.py          # Train CNN
python src/evaluate.py       # Evaluate
python src/export.py          # Export ONNX + quantize

# --- Streaming CNN+LSTM ---
python src/temporal/preprocess.py    # Preprocess frame sequences
python src/temporal/train.py         # Train CNN+LSTM
python src/temporal/evaluate.py      # Evaluate
python src/temporal/export.py        # Export batch ONNX
python src/temporal/export_streaming.py  # Export split ONNX (feature_extractor + lstm_step)
python src/temporal/evaluate_streaming.py # Evaluate streaming on test data
```

### C++ Inference

Requires [ONNX Runtime C++ SDK](https://github.com/microsoft/onnxruntime/releases):

```bash
cd src/inference_cpp
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release

# Single-frame CNN benchmark
./Release/gesture_inference.exe ../../models/gesture_model.onnx 1000

# Streaming CNN+LSTM benchmark
./Release/streaming_inference.exe ../../models/feature_extractor.onnx ../../models/lstm_step.onnx 1000
```

---

## Project Structure

```
├── data/
│   ├── raw/                        # Soli HDF5 files
│   └── processed/                  # .npz train/val/test splits
├── models/
│   ├── gesture_model.pth           # CNN checkpoint
│   ├── gesture_model_lstm.pth      # CNN+LSTM checkpoint
│   ├── gesture_model.onnx          # CNN ONNX FP32
│   ├── gesture_model_static_quant.onnx  # CNN ONNX INT8 static
│   ├── gesture_model_lstm.onnx     # CNN+LSTM batch ONNX
│   ├── feature_extractor.onnx      # Streaming: CNN feature extractor
│   └── lstm_step.onnx              # Streaming: LSTM step with hidden state
├── reports/
│   ├── evaluation_report.md        # CNN per-class accuracy, F1
│   ├── evaluation_report_lstm.md   # CNN+LSTM evaluation
│   ├── streaming_evaluation_report.md  # Streaming ONNX evaluation
│   ├── model_card.md               # Cross-variant comparison
│   └── confusion_matrix*.png
├── src/
│   ├── preprocess.py               # Soli download + single-frame HDF5 → .npz
│   ├── simulate_data.py            # Synthetic data generator
│   ├── dataset.py                  # PyTorch Dataset + augmentation
│   ├── model.py                    # RadarGestureCNN + V2 variant
│   ├── train.py                    # CNN training loop + MLflow
│   ├── evaluate.py                 # CNN metrics + confusion matrix
│   ├── export.py                   # CNN ONNX export + quantization
│   ├── benchmark_latency.py        # Python latency comparison
│   ├── generate_report.py          # Metrics report generator
│   ├── temporal/                   # CNN+LSTM temporal model
│   │   ├── preprocess.py           # Frame sequence extraction from HDF5
│   │   ├── dataset.py              # Temporal dataset + sequence augmentation
│   │   ├── model.py                # RadarGestureCNNLSTM model
│   │   ├── train.py                # CNN+LSTM training loop
│   │   ├── evaluate.py             # CNN+LSTM evaluation
│   │   ├── export.py               # Batch ONNX export
│   │   ├── export_streaming.py     # Split ONNX export (feature_extractor + lstm_step)
│   │   ├── benchmark_streaming.py  # Python streaming latency benchmark
│   │   └── evaluate_streaming.py   # Streaming ONNX evaluation on test data
│   └── inference_cpp/
│       ├── CMakeLists.txt          # Builds both targets
│       ├── main.cpp                # Single-frame CNN benchmark
│       ├── inference.h/cpp         # Single-frame ORT inference
│       ├── streaming_main.cpp      # Streaming CNN+LSTM benchmark
│       ├── streaming_inference.h/cpp  # Streaming ORT inference + hidden state
│       └── preprocessor.h/cpp
├── params.yaml                     # CNN config
├── params_lstm.yaml                # CNN+LSTM config
├── requirements.txt
└── Dockerfile
```

---

## Stack

| Component | Tool |
|-----------|------|
| Model training | PyTorch |
| Temporal model | CNN+LSTM (PyTorch) |
| Experiment tracking | MLflow |
| Model export | ONNX (opset 17) |
| Quantization | ONNX Runtime (static INT8) |
| Edge inference | ONNX Runtime C++ |
| Streaming inference | Split ONNX + hidden state management |
| Containerization | Docker |
| Evaluation | scikit-learn |