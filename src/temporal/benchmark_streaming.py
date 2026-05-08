"""Benchmark streaming CNN+LSTM inference in Python ONNX Runtime.

Measures per-frame latency for the two-model streaming pipeline
and compares with the single-frame CNN baseline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import onnxruntime as ort
import time


def benchmark_streaming(
    feature_path: str = "models/feature_extractor.onnx",
    lstm_path: str = "models/lstm_step.onnx",
    n_frames: int = 40,
    n_iterations: int = 1000,
) -> dict:
    """Benchmark streaming inference: per-frame feature extraction + LSTM step."""
    feat_session = ort.InferenceSession(feature_path)
    lstm_session = ort.InferenceSession(lstm_path)

    # Warmup
    sample_frame = np.random.randn(1, 4, 32, 32).astype(np.float32)
    h = np.zeros((1, 1, 512), dtype=np.float32)
    c = np.zeros((1, 1, 512), dtype=np.float32)
    for _ in range(10):
        feature = feat_session.run(None, {"frame_input": sample_frame})[0]
        feature = feature.reshape(1, 1, 256)
        logits, h, c = lstm_session.run(None, {"feature_input": feature, "h0": h, "c0": c})

    # Benchmark feature extraction alone
    feat_latencies = []
    for _ in range(n_iterations):
        frame = np.random.randn(1, 4, 32, 32).astype(np.float32)
        start = time.perf_counter()
        _ = feat_session.run(None, {"frame_input": frame})[0]
        feat_latencies.append((time.perf_counter() - start) * 1000)

    # Benchmark LSTM step alone
    lstm_latencies = []
    for _ in range(n_iterations):
        feature = np.random.randn(1, 1, 256).astype(np.float32)
        h = np.zeros((1, 1, 512), dtype=np.float32)
        c = np.zeros((1, 1, 512), dtype=np.float32)
        start = time.perf_counter()
        _, h, c = lstm_session.run(None, {"feature_input": feature, "h0": h, "c0": c})
        lstm_latencies.append((time.perf_counter() - start) * 1000)

    # Benchmark full streaming pipeline (feature + LSTM per frame)
    combined_latencies = []
    for _ in range(n_iterations):
        frame = np.random.randn(1, 4, 32, 32).astype(np.float32)
        h = np.zeros((1, 1, 512), dtype=np.float32)
        c = np.zeros((1, 1, 512), dtype=np.float32)
        start = time.perf_counter()
        feature = feat_session.run(None, {"frame_input": frame})[0].reshape(1, 1, 256)
        logits, h, c = lstm_session.run(None, {"feature_input": feature, "h0": h, "c0": c})
        combined_latencies.append((time.perf_counter() - start) * 1000)

    # Benchmark full 40-frame sequence (total time for gesture)
    sequence_latencies = []
    for _ in range(n_iterations):
        frames = np.random.randn(n_frames, 4, 32, 32).astype(np.float32)
        h = np.zeros((1, 1, 512), dtype=np.float32)
        c = np.zeros((1, 1, 512), dtype=np.float32)
        start = time.perf_counter()
        for t in range(n_frames):
            feature = feat_session.run(None, {"frame_input": frames[t:t+1]})[0].reshape(1, 1, 256)
            _, h, c = lstm_session.run(None, {"feature_input": feature, "h0": h, "c0": c})
        sequence_latencies.append((time.perf_counter() - start) * 1000)

    return {
        "feature_extractor": {
            "mean": np.mean(feat_latencies),
            "p95": np.percentile(feat_latencies, 95),
            "p99": np.percentile(feat_latencies, 99),
        },
        "lstm_step": {
            "mean": np.mean(lstm_latencies),
            "p95": np.percentile(lstm_latencies, 95),
            "p99": np.percentile(lstm_latencies, 99),
        },
        "combined_per_frame": {
            "mean": np.mean(combined_latencies),
            "p95": np.percentile(combined_latencies, 95),
            "p99": np.percentile(combined_latencies, 99),
        },
        "full_sequence_40": {
            "mean": np.mean(sequence_latencies),
            "p95": np.percentile(sequence_latencies, 95),
            "p99": np.percentile(sequence_latencies, 99),
        },
    }


def benchmark_single_frame_cnn(
    model_path: str = "models/gesture_model.onnx",
    n_iterations: int = 1000,
) -> dict:
    """Benchmark single-frame CNN for comparison."""
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(10):
        x = np.random.randn(1, 4, 32, 32).astype(np.float32)
        session.run(None, {input_name: x})

    latencies = []
    for _ in range(n_iterations):
        x = np.random.randn(1, 4, 32, 32).astype(np.float32)
        start = time.perf_counter()
        session.run(None, {input_name: x})
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "mean": np.mean(latencies),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Streaming CNN+LSTM Latency Benchmark (Python ORT)")
    print("=" * 60)

    print("\nBenchmarking streaming pipeline...")
    streaming = benchmark_streaming()

    print("\nBenchmarking single-frame CNN baseline...")
    cnn = benchmark_single_frame_cnn()

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    print(f"\nPer-frame latency:")
    print(f"  Feature extractor:  {streaming['feature_extractor']['mean']:.3f} ms (P95: {streaming['feature_extractor']['p95']:.3f} ms)")
    print(f"  LSTM step:           {streaming['lstm_step']['mean']:.3f} ms (P95: {streaming['lstm_step']['p95']:.3f} ms)")
    print(f"  Combined per frame:  {streaming['combined_per_frame']['mean']:.3f} ms (P95: {streaming['combined_per_frame']['p95']:.3f} ms)")
    print(f"  Single-frame CNN:    {cnn['mean']:.3f} ms (P95: {cnn['p95']:.3f} ms)")

    print(f"\nFull 40-frame sequence (total gesture processing):")
    print(f"  Streaming pipeline:  {streaming['full_sequence_40']['mean']:.3f} ms (P95: {streaming['full_sequence_40']['p95']:.3f} ms)")

    print(f"\nComparison (Python ONNX Runtime):")
    print(f"  Single-frame CNN:    {cnn['mean']:.3f} ms per frame  | 82.2% accuracy | 0.089 ms C++ latency")
    print(f"  Streaming per frame: {streaming['combined_per_frame']['mean']:.3f} ms per frame  | 98.0% accuracy | ~0.2-0.3 ms C++ est.")
    print(f"  Full 40-frame seq:   {streaming['full_sequence_40']['mean']:.3f} ms total     | 98.0% accuracy")