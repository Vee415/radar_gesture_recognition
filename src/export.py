"""ONNX model export and quantization."""

import os
from pathlib import Path

import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType, quantize_static, CalibrationDataReader, QuantFormat

from model import get_model
from dataset import get_dataloaders


def optimize_onnx(
    input_path: str = "models/gesture_model.onnx",
    output_path: str = "models/gesture_model_optimized.onnx",
) -> None:
    """Apply ONNX graph optimization passes.

    Fuses BatchNorm into Conv, eliminates identity/transpose nodes,
    folds constants, and merges consecutive operations. These are
    the same passes ORT applies at session creation with ORT_ENABLE_ALL,
    but pre-applying them produces a smaller/cleaner ONNX file.
    """
    from onnxruntime.transformers import optimizer as ort_optimizer

    # Use ORT's built-in optimization via InferenceSession with ORT_ENABLE_ALL
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = output_path

    _ = ort.InferenceSession(input_path, sess_options)
    print(f"Optimized ONNX model saved to {output_path}")

    # Print node count comparison
    orig_model = onnx.load(input_path)
    opt_model = onnx.load(output_path)
    orig_nodes = len(orig_model.graph.node)
    opt_nodes = len(opt_model.graph.node)
    print(f"  Nodes: {orig_nodes} -> {opt_nodes} ({orig_nodes - opt_nodes} fused/removed)")


def export_onnx(
    model: torch.nn.Module,
    save_path: str = "models/gesture_model.onnx",
    input_shape: tuple[int, ...] = (1, 4, 32, 32),
) -> None:
    """Export PyTorch model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=["radar_input"],
        output_names=["gesture_logits"],
        dynamic_axes={"radar_input": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported ONNX model to {save_path}")

    # Verify
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully")


def quantize_onnx(
    input_path: str = "models/gesture_model.onnx",
    output_path: str = "models/gesture_model_quant.onnx",
) -> None:
    """Apply dynamic INT8 quantization to ONNX model."""
    quantize_dynamic(
        input_path,
        output_path,
        weight_type=QuantType.QInt8,
    )
    print(f"Quantized model saved to {output_path}")


class GestureCalibrationReader(CalibrationDataReader):
    """Feeds real training data to static quantization for calibration."""

    def __init__(self, data_dir: str = "data/processed", source: str = "soli", batch_size: int = 32):
        train_loader, _, _ = get_dataloaders(data_dir=data_dir, batch_size=batch_size, source=source)
        self.data = iter(train_loader)
        self.input_name = "radar_input"

    def get_next(self) -> dict | None:
        try:
            inputs, _ = next(self.data)
            return {self.input_name: inputs.numpy()}
        except StopIteration:
            return None


def quantize_static_onnx(
    input_path: str = "models/gesture_model.onnx",
    output_path: str = "models/gesture_model_static_quant.onnx",
    data_dir: str = "data/processed",
    source: str = "soli",
) -> None:
    """Apply static INT8 quantization with calibration data.

    Static quantization pre-computes scale/zero-point for activations
    using real data, eliminating the runtime dequant overhead that makes
    dynamic quantization slower on small models.
    """
    calibration_reader = GestureCalibrationReader(data_dir=data_dir, source=source)

    quantize_static(
        input_path,
        output_path,
        calibration_reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
    )
    print(f"Static quantized model saved to {output_path}")


def verify_onnx(
    model_path: str,
    sample_input: np.ndarray | None = None,
    pytorch_model: torch.nn.Module | None = None,
    input_shape: tuple[int, ...] = (1, 4, 32, 32),
) -> None:
    """Verify ONNX model output matches PyTorch output."""
    session = ort.InferenceSession(model_path)

    if sample_input is None:
        sample_input = np.random.randn(*input_shape).astype(np.float32)

    # ONNX inference
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: sample_input})[0]

    # PyTorch inference for comparison
    if pytorch_model is not None:
        pytorch_model.eval()
        with torch.no_grad():
            torch_input = torch.from_numpy(sample_input)
            torch_output = pytorch_model(torch_input).numpy()

        # Cosine similarity
        similarity = np.dot(onnx_output.flatten(), torch_output.flatten()) / (
            np.linalg.norm(onnx_output.flatten()) * np.linalg.norm(torch_output.flatten())
        )
        print(f"ONNX vs PyTorch cosine similarity: {similarity:.6f}")
        if similarity > 0.999:
            print("Verification PASSED: outputs match closely")
        else:
            print(f"WARNING: outputs differ (similarity={similarity:.4f})")


def compare_model_sizes(
    float_path: str = "models/gesture_model.onnx",
    quant_path: str = "models/gesture_model_quant.onnx",
    static_quant_path: str = "models/gesture_model_static_quant.onnx",
    optimized_path: str = "models/gesture_model_optimized.onnx",
) -> dict:
    """Print and return model size comparison across all variants."""
    float_size = os.path.getsize(float_path)
    quant_size = os.path.getsize(quant_path)

    print(f"Float32 model:        {float_size / 1024:.1f} KB")
    print(f"INT8 dynamic quant:   {quant_size / 1024:.1f} KB  ({float_size / quant_size:.1f}x smaller)")

    result = {"float32_kb": float_size / 1024, "dynamic_kb": quant_size / 1024}

    if os.path.exists(static_quant_path):
        static_size = os.path.getsize(static_quant_path)
        print(f"INT8 static quant:    {static_size / 1024:.1f} KB  ({float_size / static_size:.1f}x smaller)")
        result["static_kb"] = static_size / 1024

    if os.path.exists(optimized_path):
        opt_size = os.path.getsize(optimized_path)
        print(f"Optimized FP32:       {opt_size / 1024:.1f} KB  ({float_size / opt_size:.1f}x smaller)")
        result["optimized_kb"] = opt_size / 1024

    return result


if __name__ == "__main__":
    import yaml

    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cpu")

    model = get_model(
        name=config["model"]["name"],
        n_classes=config["model"]["n_classes"],
    )
    model.load_state_dict(torch.load("models/gesture_model.pth", map_location=device))
    model.to(device)

    # Export float32 ONNX
    export_onnx(model)

    # Optimize ONNX graph (fuse BN into Conv, etc.)
    optimize_onnx()

    # Dynamic INT8 quantization
    quantize_onnx()

    # Static INT8 quantization (with calibration)
    quantize_static_onnx(source=config["data"]["source"])

    # Verify outputs match
    sample = np.random.randn(1, 4, 32, 32).astype(np.float32)
    verify_onnx("models/gesture_model.onnx", sample, model)
    verify_onnx("models/gesture_model_quant.onnx", sample)
    verify_onnx("models/gesture_model_static_quant.onnx", sample)

    # Compare sizes
    compare_model_sizes()