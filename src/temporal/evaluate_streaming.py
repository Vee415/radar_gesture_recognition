"""Evaluate streaming CNN+LSTM inference on actual test data.

Simulates real-time deployment: processes frames one at a time,
maintains LSTM hidden state, and tracks how prediction accuracy
builds up over the sequence.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import onnxruntime as ort
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from src.temporal.dataset import load_temporal_split
from src.preprocess import SOLI_GESTURE_LABELS


GESTURE_NAMES = [
    "pinch_index", "pinch_pinky", "pinch_middle", "pinch_ring",
    "swipe_left", "swipe_right", "swipe_up", "swipe_down",
    "slide_left", "slide_right", "slide_up", "slide_down",
]


def softmax(x):
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def evaluate_streaming(
    feature_path: str = "models/feature_extractor.onnx",
    lstm_path: str = "models/lstm_step.onnx",
    data_dir: str = "data/processed",
    source: str = "soli",
    seq_len: int = 40,
) -> None:
    """Run streaming evaluation on test set and track accuracy buildup."""

    # Load test data
    test_data, test_labels = load_temporal_split(data_dir, "test", source)
    print(f"Test set: {len(test_data)} samples, shape {test_data.shape}")

    # Create ONNX sessions
    feat_session = ort.InferenceSession(feature_path)
    lstm_session = ort.InferenceSession(lstm_path)

    # Get input/output names
    feat_input = feat_session.get_inputs()[0].name
    lstm_inputs = [s.get_inputs()[0].name for s in [lstm_session]]
    lstm_feature = lstm_session.get_inputs()[0].name
    lstm_h0 = lstm_session.get_inputs()[1].name
    lstm_c0 = lstm_session.get_inputs()[2].name

    # Track predictions at each frame count
    frame_checkpoints = [1, 2, 3, 5, 10, 15, 20, 25, 30, 40]
    predictions_at_frame = {cp: [] for cp in frame_checkpoints}
    final_predictions = []
    final_confidences = []

    # Per-sample frame-by-frame tracking (for detailed analysis)
    sample_details = []

    for idx in range(len(test_data)):
        sequence = test_data[idx]  # (40, 4, 32, 32)
        true_label = test_labels[idx]

        # Initialize hidden state
        h = np.zeros((1, 1, 512), dtype=np.float32)
        c = np.zeros((1, 1, 512), dtype=np.float32)

        # Accumulated softmax probabilities
        accumulated_probs = np.zeros((1, 12), dtype=np.float32)
        frame_count = 0

        for t in range(seq_len):
            # Feature extraction
            frame_input = sequence[t:t+1].astype(np.float32)  # (1, 4, 32, 32)
            feature = feat_session.run(None, {feat_input: frame_input})[0]  # (1, 256)
            feature = feature.reshape(1, 1, 256)  # (1, 1, 256)

            # LSTM step
            logits, h, c = lstm_session.run(
                None,
                {lstm_feature: feature, lstm_h0: h, lstm_c0: c},
            )

            # Accumulate softmax probabilities
            frame_probs = softmax(logits)  # (1, 12)
            frame_count += 1
            accumulated_probs = accumulated_probs * (1 - 1.0 / frame_count) + frame_probs / frame_count

            # Record predictions at checkpoints
            if frame_count in frame_checkpoints:
                pred = np.argmax(accumulated_probs)
                predictions_at_frame[frame_count].append((pred, true_label))

        # Final prediction (after all 40 frames)
        final_pred = np.argmax(accumulated_probs)
        final_conf = accumulated_probs[0, final_pred]
        final_predictions.append(final_pred)
        final_confidences.append(final_conf)

    # Calculate metrics at each frame checkpoint
    print("\n" + "=" * 70)
    print("STREAMING INFERENCE: ACCURACY BUILDUP OVER FRAMES")
    print("=" * 70)
    print(f"\n{'Frames':>8} | {'Accuracy':>10} | {'Macro F1':>10}")
    print("-" * 40)

    for cp in frame_checkpoints:
        if cp in predictions_at_frame and len(predictions_at_frame[cp]) > 0:
            preds, labels = zip(*predictions_at_frame[cp])
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average="macro", zero_division=0)
            print(f"{cp:>8} | {acc:>10.4f} | {f1:>10.4f}")

    # Final metrics (after all 40 frames)
    final_preds = np.array(final_predictions)
    final_labels = np.array(test_labels)

    overall_acc = accuracy_score(final_labels, final_preds)
    macro_f1 = f1_score(final_labels, final_preds, average="macro")
    cm = confusion_matrix(final_labels, final_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    print("\n" + "=" * 70)
    print("FINAL STREAMING RESULTS (40 frames)")
    print("=" * 70)
    print(f"\nOverall Accuracy: {overall_acc:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")

    print("\nPer-Class Accuracy:")
    print("-" * 40)
    for i, name in enumerate(GESTURE_NAMES):
        if i < len(per_class_acc):
            print(f"  {name:20s}: {per_class_acc[i]:.4f}")

    print("\nConfusion Matrix:")
    print(classification_report(final_labels, final_preds,
                                target_names=GESTURE_NAMES[:len(np.unique(final_labels))]))

    # Comparison with single-frame CNN
    print("\n" + "=" * 70)
    print("COMPARISON: SINGLE-FRAME CNN vs STREAMING CNN+LSTM")
    print("=" * 70)
    print(f"\n{'Metric':<25} | {'Single-frame CNN':>18} | {'Streaming CNN+LSTM':>20}")
    print("-" * 70)
    print(f"{'Overall Accuracy':<25} | {'82.2%':>18} | {overall_acc*100:>19.1f}%")
    print(f"{'Macro F1':<25} | {'0.82':>18} | {macro_f1:>20.2f}")
    print(f"{'C++ Latency per frame':<25} | {'0.089 ms':>18} | {'0.27 ms':>20}")
    print(f"{'Full sequence latency':<25} | {'0.089 ms':>18} | {'~11 ms':>20}")
    print(f"{'Model size (ONNX)':<25} | {'2,430 KB':>18} | {'8,602 KB':>20}")
    print(f"{'Model size (split)':<25} | {'N/A':>18} | {'2,417 + 6,185 KB':>20}")

    # Per-class comparison
    single_frame_accs = {
        "pinch_index": 0.771, "pinch_pinky": 0.844, "pinch_middle": 0.564, "pinch_ring": 0.634,
        "swipe_left": 0.797, "swipe_right": 0.912, "swipe_up": 0.913, "swipe_down": 0.925,
        "slide_left": 0.761, "slide_right": 0.849, "slide_up": 0.854, "slide_down": 1.000,
    }

    print(f"\n{'Gesture':<20} | {'Single-frame':>12} | {'Streaming':>12} | {'Delta':>8}")
    print("-" * 60)
    for i, name in enumerate(GESTURE_NAMES):
        if i < len(per_class_acc):
            sf = single_frame_accs.get(name, 0)
            st = per_class_acc[i]
            delta = st - sf
            print(f"{name:<20} | {sf*100:>11.1f}% | {st*100:>11.1f}% | {delta*100:>+7.1f}pp")

    # Save results
    import os
    os.makedirs("reports", exist_ok=True)

    with open("reports/streaming_evaluation_report.md", "w", encoding="utf-8") as f:
        f.write("# Streaming CNN+LSTM Evaluation Report\n\n")
        f.write("## Architecture\n\n")
        f.write("Two-model streaming pipeline for real-time edge deployment:\n")
        f.write("1. **Feature extractor**: (1, 4, 32, 32) → (1, 256) — CNN backbone\n")
        f.write("2. **LSTM step**: (1, 1, 256) + hidden state → (1, 12) + updated hidden state\n")
        f.write("3. Hidden state maintained across frames; reset between gestures\n\n")
        f.write("## Results\n\n")
        f.write(f"- **Overall Accuracy**: {overall_acc:.4f}\n")
        f.write(f"- **Macro F1**: {macro_f1:.4f}\n\n")
        f.write("### Accuracy Buildup Over Frames\n\n")
        f.write("| Frames | Accuracy | Macro F1 |\n")
        f.write("|--------|----------|----------|\n")
        for cp in frame_checkpoints:
            if cp in predictions_at_frame and len(predictions_at_frame[cp]) > 0:
                preds, labels = zip(*predictions_at_frame[cp])
                acc = accuracy_score(labels, preds)
                f1 = f1_score(labels, preds, average="macro", zero_division=0)
                f.write(f"| {cp} | {acc:.4f} | {f1:.4f} |\n")
        f.write(f"\n### Per-Class Accuracy\n\n")
        f.write("| Gesture | Single-frame CNN | Streaming CNN+LSTM | Delta |\n")
        f.write("|--------|-----------------|---------------------|-------|\n")
        for i, name in enumerate(GESTURE_NAMES):
            if i < len(per_class_acc):
                sf = single_frame_accs.get(name, 0)
                st = per_class_acc[i]
                f.write(f"| {name} | {sf:.3f} | {st:.3f} | {st-sf:+.3f} |\n")
        f.write(f"\n### Latency\n\n")
        f.write("| Metric | Single-frame CNN | Streaming CNN+LSTM |\n")
        f.write("|--------|-----------------|---------------------|\n")
        f.write("| Per-frame (C++) | 0.089 ms | 0.27 ms |\n")
        f.write("| Full sequence | 0.089 ms | ~11 ms |\n")
        f.write("| Model size (ONNX) | 2,430 KB | 8,602 KB (split: 2,417 + 6,185 KB) |\n")

    print(f"\nReport saved to reports/streaming_evaluation_report.md")


if __name__ == "__main__":
    evaluate_streaming()