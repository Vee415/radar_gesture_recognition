"""Latency benchmark: PyTorch vs ONNX FP32 vs ONNX INT8 dynamic vs ONNX INT8 static.

Runs inference on the same test data and compares:
- Mean, std, p95, p99 latency
- Prediction consistency across runtimes
- Model size comparison
"""

import os
import time
import numpy as np
import torch
import onnxruntime as ort
from dataset import get_dataloaders
from model import get_model
import yaml


def benchmark_pytorch(model, test_loader, device, n_runs=5):
    """Benchmark PyTorch model inference latency."""
    model.eval()
    # Warmup
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            model(inputs)
            break

    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                start = time.perf_counter()
                outputs = model(inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                latencies.append(elapsed * 1000 / inputs.size(0))  # ms per sample

    return np.array(latencies)


def benchmark_onnx(model_path, test_loader, n_runs=5):
    """Benchmark ONNX Runtime inference latency."""
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    # Warmup
    for inputs, _ in test_loader:
        session.run(None, {input_name: inputs.numpy()})
        break

    latencies = []
    for _ in range(n_runs):
        for inputs, _ in test_loader:
            inp = inputs.numpy()
            start = time.perf_counter()
            session.run(None, {input_name: inp})
            elapsed = time.perf_counter() - start
            latencies.append(elapsed * 1000 / inp.shape[0])  # ms per sample

    return np.array(latencies)


def print_stats(name, latencies):
    """Print latency statistics."""
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Mean:   {latencies.mean():.3f} ms")
    print(f"  Std:    {latencies.std():.3f} ms")
    print(f"  P95:    {np.percentile(latencies, 95):.3f} ms")
    print(f"  P99:    {np.percentile(latencies, 99):.3f} ms")
    print(f"  Min:    {latencies.min():.3f} ms")
    print(f"  Max:    {latencies.max():.3f} ms")
    return {
        "name": name,
        "mean": latencies.mean(),
        "std": latencies.std(),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99),
        "min": latencies.min(),
        "max": latencies.max(),
    }


def compare_predictions(pytorch_model, model_paths, test_loader, device):
    """Check prediction consistency across all ONNX runtimes vs PyTorch."""
    pytorch_model.eval()

    sessions = {}
    for name, path in model_paths.items():
        if os.path.exists(path):
            sessions[name] = ort.InferenceSession(path)

    results = {}
    with torch.no_grad():
        for inputs, _ in test_loader:
            pt_preds = pytorch_model(inputs.to(device)).argmax(dim=1).cpu().numpy()
            for name, session in sessions.items():
                input_name = session.get_inputs()[0].name
                preds = session.run(None, {input_name: inputs.numpy()})[0].argmax(axis=1)
                if name not in results:
                    results[name] = {"match": 0, "total": 0}
                results[name]["match"] += (pt_preds == preds).sum()
                results[name]["total"] += len(pt_preds)
            break  # single batch is enough for consistency check

    print(f"\nPrediction consistency (vs PyTorch):")
    for name, r in results.items():
        print(f"  {name}: {r['match']}/{r['total']} ({100*r['match']/r['total']:.1f}% match)")


if __name__ == "__main__":
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, _, test_loader = get_dataloaders(
        data_dir="data/processed",
        batch_size=32,
        source=config["data"]["source"],
    )

    # Load PyTorch model
    model = get_model(
        name=config["model"]["name"],
        n_classes=config["model"]["n_classes"],
    ).to(device)
    model.load_state_dict(torch.load("models/gesture_model.pth", map_location=device))

    print("\nBenchmarking... (5 runs over test set)")

    pt_latencies = benchmark_pytorch(model, test_loader, device)
    fp32_latencies = benchmark_onnx("models/gesture_model.onnx", test_loader)

    stats = []
    stats.append(print_stats("PyTorch (CPU)", pt_latencies))
    stats.append(print_stats("ONNX FP32", fp32_latencies))

    dynamic_path = "models/gesture_model_quant.onnx"
    static_path = "models/gesture_model_static_quant.onnx"

    if os.path.exists(dynamic_path):
        dyn_latencies = benchmark_onnx(dynamic_path, test_loader)
        stats.append(print_stats("ONNX INT8 (dynamic)", dyn_latencies))

    if os.path.exists(static_path):
        static_latencies = benchmark_onnx(static_path, test_loader)
        stats.append(print_stats("ONNX INT8 (static)", static_latencies))

    # Speedup comparison
    pt_mean = stats[0]["mean"]
    print(f"\n{'='*50}")
    print(f"  Speedup vs PyTorch")
    print(f"{'='*50}")
    for s in stats[1:]:
        speedup = pt_mean / s["mean"]
        print(f"  {s['name']}: {speedup:.2f}x faster")

    # Model sizes
    print(f"\n{'='*50}")
    print(f"  Model Sizes")
    print(f"{'='*50}")
    for path, label in [
        ("models/gesture_model.onnx", "ONNX FP32"),
        (dynamic_path, "ONNX INT8 (dynamic)"),
        (static_path, "ONNX INT8 (static)"),
    ]:
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"  {label}: {size_kb:.1f} KB")

    # Prediction consistency
    model_paths = {}
    if os.path.exists(dynamic_path):
        model_paths["ONNX INT8 (dynamic)"] = dynamic_path
    if os.path.exists(static_path):
        model_paths["ONNX INT8 (static)"] = static_path

    if model_paths:
        compare_predictions(model, model_paths, test_loader, device)