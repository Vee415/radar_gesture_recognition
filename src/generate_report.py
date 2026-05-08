"""Generate comprehensive metrics comparison report.

Combines model evaluation results (accuracy, F1), Python latency benchmarks,
C++ latency benchmarks, and model sizes into a single report.
"""

import os
import numpy as np
import torch
import onnxruntime as ort
from dataset import get_dataloaders
from model import get_model
import yaml


def get_model_sizes():
    """Get file sizes for all model variants."""
    sizes = {}
    for name, path in [
        ("PyTorch (.pth)", "models/gesture_model.pth"),
        ("ONNX FP32", "models/gesture_model.onnx"),
        ("ONNX INT8 (dynamic)", "models/gesture_model_quant.onnx"),
        ("ONNX INT8 (static)", "models/gesture_model_static_quant.onnx"),
    ]:
        if os.path.exists(path):
            kb = os.path.getsize(path) / 1024
            sizes[name] = kb
    return sizes


def benchmark_onnx(model_path, test_loader, n_runs=5):
    """Benchmark ONNX model latency."""
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
            start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

            import time
            t0 = time.perf_counter()
            session.run(None, {input_name: inp})
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed * 1000 / inp.shape[0])

    return np.array(latencies)


def evaluate_accuracy(model_path, test_loader, device, n_classes):
    """Evaluate ONNX model accuracy on test set."""
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    all_preds = []
    all_labels = []

    for inputs, labels in test_loader:
        outputs = session.run(None, {input_name: inputs.numpy()})[0]
        preds = outputs.argmax(axis=1)
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()

    # Per-class accuracy
    per_class = {}
    for c in range(n_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class[c] = (all_preds[mask] == c).mean()

    return accuracy, per_class


def generate_report(config):
    device = torch.device("cpu")

    _, _, test_loader = get_dataloaders(
        data_dir="data/processed",
        batch_size=config["train"]["batch_size"],
        source=config["data"]["source"],
    )

    # Load pytorch model for reference accuracy
    model = get_model(name=config["model"]["name"], n_classes=config["model"]["n_classes"])
    model.load_state_dict(torch.load("models/gesture_model.pth", map_location=device))
    model.eval()

    # PyTorch accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    pytorch_acc = correct / total

    # ONNX model paths
    model_variants = []
    for name, path in [
        ("ONNX FP32", "models/gesture_model.onnx"),
        ("ONNX INT8 (dynamic)", "models/gesture_model_quant.onnx"),
        ("ONNX INT8 (static)", "models/gesture_model_static_quant.onnx"),
    ]:
        if os.path.exists(path):
            model_variants.append((name, path))

    # Evaluate accuracy for each variant
    print("Evaluating accuracy...")
    accuracy_results = {"PyTorch": pytorch_acc}
    for name, path in model_variants:
        acc, per_class = evaluate_accuracy(path, test_loader, device, config["model"]["n_classes"])
        accuracy_results[name] = acc
        print(f"  {name}: {acc:.4f}")

    # Benchmark latency (Python)
    print("\nBenchmarking latency (Python)...")
    latency_results = {}
    for name, path in model_variants:
        lat = benchmark_onnx(path, test_loader)
        latency_results[name] = {
            "mean": lat.mean(),
            "std": lat.std(),
            "p95": np.percentile(lat, 95),
            "p99": np.percentile(lat, 99),
        }
        print(f"  {name}: mean={lat.mean():.3f}ms, p95={np.percentile(lat, 95):.3f}ms")

    # C++ latency results (from benchmark runs)
    cpp_benchmarks = {
        "ONNX FP32": {"mean": 0.094, "p95": 0.119, "p99": 0.230},
        "ONNX INT8 (static)": {"mean": 0.101, "p95": 0.129, "p99": 0.188},
        "ONNX INT8 (dynamic)": {"mean": 0.717, "p95": 0.852, "p99": 0.976},
    }

    # Model sizes
    sizes = get_model_sizes()

    # Write report
    report = """# Metrics Comparison Report

## Model

- Architecture: RadarGestureCNN (2D CNN)
- Parameters: {:,}
- Input: (batch, 4, 32, 32) — 4 range-doppler channels
- Classes: {} (Soli gesture dataset)

## Accuracy Comparison

| Model | Accuracy | Accuracy Drop |
|-------|----------|-------------- |
""".format(
        sum(p.numel() for p in model.parameters()),
        config["model"]["n_classes"],
    )

    pytorch_acc = accuracy_results["PyTorch"]
    for name, acc in accuracy_results.items():
        drop = acc - pytorch_acc if name != "PyTorch" else 0
        drop_str = f"{drop:+.4f}" if name != "PyTorch" else "baseline"
        report += f"| {name} | {acc:.4f} | {drop_str} |\n"

    report += """
## Model Size Comparison

| Model | Size (KB) | Compression |
|-------|----------|-------------|
"""

    baseline_size = sizes.get("PyTorch (.pth)", sizes.get("ONNX FP32", 1))
    for name, kb in sizes.items():
        ratio = baseline_size / kb if kb > 0 else 0
        report += f"| {name} | {kb:.1f} | {ratio:.1f}x |\n"

    report += """
## Latency Comparison (Python, per sample)

| Model | Mean (ms) | Std (ms) | P95 (ms) | P99 (ms) |
|-------|-----------|----------|----------|----------|
"""

    for name, lat in latency_results.items():
        report += f"| {name} | {lat['mean']:.3f} | {lat['std']:.3f} | {lat['p95']:.3f} | {lat['p99']:.3f} |\n"

    report += """
## Latency Comparison (C++ ONNX Runtime, per sample)

| Model | Mean (ms) | P95 (ms) | P99 (ms) |
|-------|-----------|----------|----------|
"""

    for name, lat in cpp_benchmarks.items():
        report += f"| {name} | {lat['mean']:.3f} | {lat['p95']:.3f} | {lat['p99']:.3f} |\n"

    report += """
## Summary

| Metric | PyTorch | ONNX FP32 | ONNX INT8 (static) | ONNX INT8 (dynamic) |
|--------|---------|-----------|---------------------|--------------------- |
"""

    # Accuracy row
    report += "| Accuracy | "
    for name in ["PyTorch", "ONNX FP32", "ONNX INT8 (static)", "ONNX INT8 (dynamic)"]:
        if name in accuracy_results:
            report += f"{accuracy_results[name]:.4f} | "
        else:
            report += "— | "

    # Size row
    report += "\n| Size (KB) | "
    for name in ["PyTorch (.pth)", "ONNX FP32", "ONNX INT8 (static)", "ONNX INT8 (dynamic)"]:
        if name in sizes:
            report += f"{sizes[name]:.1f} | "
        else:
            report += "— | "

    # C++ latency row
    report += "\n| C++ Latency (ms) | — | "
    for name in ["ONNX FP32", "ONNX INT8 (static)", "ONNX INT8 (dynamic)"]:
        if name in cpp_benchmarks:
            report += f"{cpp_benchmarks[name]['mean']:.3f} | "
        else:
            report += "— | "

    report += """
"""

    # Write
    with open("reports/model_card.md", "w") as f:
        f.write(report)
    print(f"\nReport saved to reports/model_card.md")


if __name__ == "__main__":
    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
    generate_report(config)