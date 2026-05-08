"""Export streaming ONNX models for real-time inference.

Splits the CNN+LSTM into two models for edge deployment:
  1. Feature extractor: (1, 4, 32, 32) -> (1, 256)
     CNN backbone + frame_fc compress to 256-dim feature vector
  2. LSTM step: (1, 1, 256) + (h0, c0) -> (1, 12) + (h1, c1)
     LSTM + classifier, runs per frame with hidden state

In real-time deployment:
  - Each radar frame goes through the feature extractor
  - The 256-dim feature + previous hidden state goes through LSTM step
  - Hidden state is maintained between frames
  - After enough frames, the prediction stabilizes
  - Reset hidden state when a new gesture starts
"""

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


class FeatureExtractor(nn.Module):
    """CNN backbone + frame FC, producing 256-dim feature vectors per frame."""

    def __init__(self, cnn_lstm_model):
        super().__init__()
        self.cnn_features = cnn_lstm_model.cnn_features
        self.frame_fc = cnn_lstm_model.frame_fc
        self.cnn_feature_dim = cnn_lstm_model.cnn_feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: (1, 4, 32, 32) -> Output: (1, 256)"""
        x = self.cnn_features(x)
        x = x.view(x.size(0), -1)
        x = self.frame_fc(x)
        return x


class LSTMStep(nn.Module):
    """Single LSTM timestep + classifier, with explicit hidden state.

    Takes a 256-dim feature vector and previous hidden state,
    returns 12-dim logits and updated hidden state.
    """

    def __init__(self, cnn_lstm_model):
        super().__init__()
        self.lstm = cnn_lstm_model.lstm
        self.classifier = cnn_lstm_model.classifier
        self.hidden_size = cnn_lstm_model._lstm_hidden_size

    def forward(
        self,
        feature: torch.Tensor,
        h0: torch.Tensor,
        c0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single LSTM step.

        Args:
            feature: (1, 1, 256) - one timestep feature vector
            h0: (1, 1, 512) - previous hidden state
            c0: (1, 1, 512) - previous cell state

        Returns:
            logits: (1, 12) - per-frame class logits
            h1: (1, 1, 512) - updated hidden state
            c1: (1, 1, 512) - updated cell state
        """
        output, (h1, c1) = self.lstm(feature, (h0, c0))
        # output: (1, 1, 512) -> classifier -> (1, 1, 12)
        logits = self.classifier(output)
        # Squeeze the timestep dimension: (1, 1, 12) -> (1, 12)
        logits = logits.squeeze(1)
        return logits, h1, c1


def export_feature_extractor(
    model: nn.Module,
    save_path: str = "models/feature_extractor.onnx",
) -> None:
    """Export CNN feature extractor to ONNX."""
    feature_ext = FeatureExtractor(model)
    feature_ext.eval()

    dummy_input = torch.randn(1, 4, 32, 32)

    torch.onnx.export(
        feature_ext,
        dummy_input,
        save_path,
        input_names=["frame_input"],
        output_names=["feature_output"],
        opset_version=17,
    )
    print(f"Feature extractor exported to {save_path}")

    # Verify
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print(f"  Input names: {[i.name for i in onnx_model.graph.input]}")
    print(f"  Output names: {[o.name for o in onnx_model.graph.output]}")
    print("  ONNX verification passed")


def export_lstm_step(
    model: nn.Module,
    save_path: str = "models/lstm_step.onnx",
    hidden_size: int = 512,
    feature_size: int = 256,
) -> None:
    """Export single LSTM step with explicit hidden state to ONNX."""
    lstm_step = LSTMStep(model)
    lstm_step.eval()

    dummy_feature = torch.randn(1, 1, feature_size)
    dummy_h0 = torch.zeros(1, 1, hidden_size)
    dummy_c0 = torch.zeros(1, 1, hidden_size)

    torch.onnx.export(
        lstm_step,
        (dummy_feature, dummy_h0, dummy_c0),
        save_path,
        input_names=["feature_input", "h0", "c0"],
        output_names=["logits_output", "h1", "c1"],
        opset_version=17,
    )
    print(f"LSTM step exported to {save_path}")

    # Verify
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print(f"  Inputs: {[i.name for i in onnx_model.graph.input]}")
    print(f"  Outputs: {[o.name for o in onnx_model.graph.output]}")
    print("  ONNX verification passed")


def verify_streaming_pipeline(
    feature_path: str = "models/feature_extractor.onnx",
    lstm_path: str = "models/lstm_step.onnx",
    pytorch_model: nn.Module | None = None,
    seq_len: int = 40,
) -> None:
    """Verify streaming inference matches batch inference."""
    feat_session = ort.InferenceSession(feature_path)
    lstm_session = ort.InferenceSession(lstm_path)

    # Simulate streaming: process 40 frames one at a time
    rng = np.random.RandomState(42)
    frames = rng.randn(seq_len, 4, 32, 32).astype(np.float32)

    # Initialize hidden state
    h = np.zeros((1, 1, 512), dtype=np.float32)
    c = np.zeros((1, 1, 512), dtype=np.float32)

    frame_logits_list = []

    for t in range(seq_len):
        # Feature extraction
        frame_input = frames[t:t+1]  # (1, 4, 32, 32)
        feature = feat_session.run(None, {"frame_input": frame_input})[0]  # (1, 256)
        feature = feature.reshape(1, 1, 256)  # (1, 1, 256)

        # LSTM step
        logits, h, c = lstm_session.run(
            None,
            {"feature_input": feature, "h0": h, "c0": c},
        )
        frame_logits_list.append(logits)  # (1, 12)

    # Average softmax probabilities across all frames (same as batch model)
    from scipy.special import softmax as scipy_softmax
    frame_probs = scipy_softmax(np.array(frame_logits_list), axis=-1)  # (40, 1, 12)
    streaming_pred = frame_probs.mean(axis=0)  # (1, 12)
    streaming_class = np.argmax(streaming_pred)
    streaming_conf = streaming_pred[0, streaming_class]

    # Compare with batch PyTorch model
    if pytorch_model is not None:
        pytorch_model.eval()
        with torch.no_grad():
            batch_input = torch.from_numpy(frames).unsqueeze(0)  # (1, 40, 4, 32, 32)
            batch_output = pytorch_model(batch_input).numpy()
        batch_class = np.argmax(batch_output)
        batch_conf = np.exp(batch_output[0, batch_class]) / np.sum(np.exp(batch_output))

        print(f"\nStreaming prediction:  class {streaming_class} (conf: {streaming_conf:.4f})")
        print(f"Batch prediction:      class {batch_class} (conf: {batch_conf:.4f})")
        print(f"Predictions match: {streaming_class == batch_class}")

    # Show how accuracy builds over time
    print("\nAccuracy buildup over frames (softmax averaged from frame 0 to t):")
    for t in [1, 3, 5, 10, 20, 30, 40]:
        if t <= seq_len:
            probs = frame_probs[:t].mean(axis=0)  # (1, 12)
            pred = np.argmax(probs)
            conf = probs[0, pred]
            print(f"  After {t:2d} frames: class {pred:2d} (conf: {conf:.4f})")


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
    model.eval()

    # Export both models
    export_feature_extractor(model)
    export_lstm_step(model)

    # Verify streaming pipeline matches batch inference
    print("\nVerifying streaming pipeline...")
    verify_streaming_pipeline(pytorch_model=model)

    # Also verify feature extractor output shape
    feat_session = ort.InferenceSession("models/feature_extractor.onnx")
    sample = np.random.randn(1, 4, 32, 32).astype(np.float32)
    output = feat_session.run(None, {"frame_input": sample})[0]
    print(f"\nFeature extractor: input (1, 4, 32, 32) -> output {output.shape}")

    # Verify LSTM step output shapes
    lstm_session = ort.InferenceSession("models/lstm_step.onnx")
    feature = np.random.randn(1, 1, 256).astype(np.float32)
    h = np.zeros((1, 1, 512), dtype=np.float32)
    c = np.zeros((1, 1, 512), dtype=np.float32)
    logits, h_new, c_new = lstm_session.run(None, {"feature_input": feature, "h0": h, "c0": c})
    print(f"LSTM step: feature (1, 1, 256) + h0 (1, 1, 512) + c0 (1, 1, 512)")
    print(f"  -> logits {logits.shape}, h1 {h_new.shape}, c1 {c_new.shape}")

    # Print model sizes
    import os
    feat_size = os.path.getsize("models/feature_extractor.onnx")
    lstm_size = os.path.getsize("models/lstm_step.onnx")
    batch_size = os.path.getsize("models/gesture_model_lstm.onnx")
    print(f"\nModel sizes:")
    print(f"  Feature extractor: {feat_size / 1024:.1f} KB")
    print(f"  LSTM step:          {lstm_size / 1024:.1f} KB")
    print(f"  Combined:           {(feat_size + lstm_size) / 1024:.1f} KB")
    print(f"  Batch model:        {batch_size / 1024:.1f} KB")