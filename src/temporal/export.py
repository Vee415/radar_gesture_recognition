"""ONNX export for CNN+LSTM temporal model."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

from src.temporal.model import build_cnn_lstm


class ONNXWrapper(nn.Module):
    """Wrapper that removes conditional logic for clean ONNX tracing."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x, return_frame_logits=False)


def export_onnx(
    model: torch.nn.Module,
    save_path: str = "models/gesture_model_lstm.onnx",
    input_shape: tuple[int, ...] = (1, 40, 4, 32, 32),
) -> None:
    """Export CNN+LSTM model to ONNX format."""
    model.eval()
    wrapper = ONNXWrapper(model)
    wrapper.eval()
    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        wrapper,
        dummy_input,
        save_path,
        input_names=["sequence_input"],
        output_names=["gesture_logits"],
        opset_version=17,
    )
    print(f"Exported ONNX model to {save_path}")

    # Verify
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully")


def verify_onnx(
    model_path: str,
    pytorch_model: torch.nn.Module | None = None,
    input_shape: tuple[int, ...] = (1, 40, 4, 32, 32),
) -> None:
    """Verify ONNX model output matches PyTorch output."""
    session = ort.InferenceSession(model_path)
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


if __name__ == "__main__":
    with open("params_lstm.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cpu")

    model = build_cnn_lstm(
        cnn_backbone=config["model"]["cnn_backbone"],
        n_classes=config["model"]["n_classes"],
        lstm_hidden_size=config["model"]["lstm_hidden_size"],
        lstm_num_layers=config["model"]["lstm_num_layers"],
        lstm_dropout=config["model"]["lstm_dropout"],
        fc_hidden_size=config["model"]["fc_hidden_size"],
    )
    model.load_state_dict(torch.load("models/gesture_model_lstm.pth", map_location=device, weights_only=True))
    model.to(device)

    # Export
    export_onnx(model)

    # Verify
    verify_onnx("models/gesture_model_lstm.onnx", model)