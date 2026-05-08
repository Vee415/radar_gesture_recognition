"""CNN+LSTM temporal model for radar gesture classification.

Architecture: CNN backbone extracts per-frame features, LSTM models
temporal dynamics across the frame sequence, classifier produces
per-frame logits that are average-pooled into a sequence prediction.

Based on the Deep-Soli UIST 2016 paper (Wang et al.):
  - Single-layer LSTM with 512 hidden units
  - Per-frame classification with average-pooled softmax
  - Unidirectional (for real-time inference)
  - End-to-end training of CNN+LSTM jointly

Input:  (batch, seq_len, 4, 32, 32)
Output: (batch, n_classes) — sequence-level logits
"""

import torch
import torch.nn as nn


class RadarGestureCNNLSTM(nn.Module):
    """CNN+LSTM for temporal radar gesture classification.

    CNN backbone extracts per-frame features from (4, 32, 32) range-doppler
    maps. LSTM processes the sequence of frame features. Classifier produces
    per-frame logits, averaged across time for sequence-level prediction.
    """

    def __init__(
        self,
        n_classes: int = 12,
        in_channels: int = 4,
        cnn_feature_dim: int = 2048,
        lstm_hidden_size: int = 512,
        lstm_num_layers: int = 1,
        lstm_dropout: float = 0.5,
        fc_hidden_size: int = 256,
        cnn_features: nn.Module | None = None,
    ):
        super().__init__()

        # CNN backbone — provided externally (from RadarGestureCNN.features)
        self.cnn_features = cnn_features
        self.cnn_feature_dim = cnn_feature_dim

        # FC to compress CNN features before LSTM
        self.frame_fc = nn.Sequential(
            nn.Linear(cnn_feature_dim, fc_hidden_size),
            nn.ReLU(inplace=True),
        )

        # LSTM (unidirectional, matching Deep-Soli paper)
        self.lstm = nn.LSTM(
            input_size=fc_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=False,
        )

        # Per-frame classifier
        self.classifier = nn.Sequential(
            nn.Dropout(lstm_dropout),
            nn.Linear(lstm_hidden_size, n_classes),
        )

        self._lstm_hidden_size = lstm_hidden_size
        self._lstm_num_layers = lstm_num_layers
        self._fc_hidden_size = fc_hidden_size
        self._n_classes = n_classes

    def forward(
        self,
        x: torch.Tensor,
        return_frame_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, 4, 32, 32).
            return_frame_logits: If True, also return per-frame logits.

        Returns:
            If return_frame_logits is False:
                logits: (batch, n_classes) — sequence-level prediction
            If return_frame_logits is True:
                (seq_logits, frame_logits) where frame_logits is (batch, seq_len, n_classes)
        """
        batch_size, seq_len, c, h, w = x.shape

        # Process all frames through CNN at once
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.cnn_features(x)
        x = x.view(batch_size * seq_len, -1)
        x = self.frame_fc(x)

        # Reshape to sequence
        x = x.view(batch_size, seq_len, -1)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Per-frame classification
        frame_logits = self.classifier(lstm_out)  # (B, T, n_classes)

        # Sequence-level: average-pool softmax probabilities across time
        frame_probs = torch.softmax(frame_logits, dim=-1)
        seq_probs = frame_probs.mean(dim=1)
        seq_logits = torch.log(seq_probs + 1e-8)

        if return_frame_logits:
            return seq_logits, frame_logits
        return seq_logits

    def freeze_cnn(self) -> None:
        """Freeze CNN backbone parameters."""
        for param in self.cnn_features.parameters():
            param.requires_grad = False

    def unfreeze_cnn(self) -> None:
        """Unfreeze CNN backbone for fine-tuning."""
        for param in self.cnn_features.parameters():
            param.requires_grad = True


def build_cnn_lstm(
    cnn_backbone: str = "cnn",
    n_classes: int = 12,
    lstm_hidden_size: int = 512,
    lstm_num_layers: int = 1,
    lstm_dropout: float = 0.5,
    fc_hidden_size: int = 256,
    pretrained_cnn_path: str | None = None,
) -> RadarGestureCNNLSTM:
    """Build a RadarGestureCNNLSTM from config, importing original CNN.

    This function handles the import of RadarGestureCNN/RadarGestureCNNV2
    from the original src/model.py and wires up the CNN backbone.
    """
    from src.model import RadarGestureCNN, RadarGestureCNNV2

    if cnn_backbone == "cnn":
        cnn = RadarGestureCNN(n_classes=n_classes)
        cnn_feature_dim = 128 * 4 * 4  # 2048
    elif cnn_backbone == "cnn_v2":
        cnn = RadarGestureCNNV2(n_classes=n_classes)
        cnn_feature_dim = 128  # global avg pool
    else:
        raise ValueError(f"Unknown cnn_backbone: {cnn_backbone}. Available: cnn, cnn_v2")

    model = RadarGestureCNNLSTM(
        n_classes=n_classes,
        cnn_feature_dim=cnn_feature_dim,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        lstm_dropout=lstm_dropout,
        fc_hidden_size=fc_hidden_size,
        cnn_features=cnn.features,
    )

    # Load pretrained CNN weights if provided
    if pretrained_cnn_path is not None:
        state_dict = torch.load(pretrained_cnn_path, map_location="cpu", weights_only=True)
        cnn_keys = {k: v for k, v in state_dict.items() if k.startswith("features.")}
        model.cnn_features.load_state_dict(cnn_keys)
        print(f"Loaded {len(cnn_keys)} pretrained CNN parameter groups")

    return model