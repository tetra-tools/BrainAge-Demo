"""SFCN architecture in plain PyTorch (no Lightning dependency).

Layer names are identical to ``models/sfcn.py`` of the training repository, so
the ``state_dict`` of a Lightning checkpoint loads without any key remapping.
The network regresses the age directly in years (L1/Huber loss on raw ages):
no output transformation is applied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, maxpool: bool = True,
                 kernel_size: int = 3, padding: int = 0, maxpool_stride: int = 2):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.maxpool = nn.MaxPool3d(2, stride=maxpool_stride)
        self.relu = nn.ReLU(inplace=True)
        self.do_maxpool = maxpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if self.do_maxpool:
            x = self.maxpool(x)
        return self.relu(x)


class SFCN(nn.Module):
    """Simple Fully Convolutional Network (Peng et al., 2021) as trained in the source repo."""

    def __init__(self, channel_number: int = 32, dropout: float = 0.1):
        super().__init__()
        c = channel_number
        self.feature_extractor = nn.Sequential(
            ConvLayer(1, c, maxpool=True, kernel_size=3, padding=1),
            ConvLayer(c, c * 2, maxpool=True, kernel_size=3, padding=1),
            ConvLayer(c * 2, c * 4, maxpool=True, kernel_size=3, padding=1),
            ConvLayer(c * 4, c * 8, maxpool=True, kernel_size=3, padding=1),
            ConvLayer(c * 8, c * 8, maxpool=True, kernel_size=3, padding=1),
            ConvLayer(c * 8, c * 2, maxpool=False, kernel_size=1, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.AvgPool3d([5, 6, 5]),
            nn.Dropout(dropout),
            nn.Conv3d(c * 2, 1, kernel_size=1, padding=0),
            nn.Flatten(start_dim=1),
            nn.Linear(1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x.squeeze(-1)  # (N,) predicted age in years


def load_lightning_checkpoint(path: str | Path) -> Tuple[SFCN, Dict[str, Any]]:
    """Load an SFCN Lightning checkpoint (``model_best_score.ckpt``) into a plain ``SFCN``.

    Lightning checkpoints contain pickled hyper-parameters, hence
    ``weights_only=False``: only load checkpoints you produced yourself.
    """
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    hparams = dict(ckpt.get("hyper_parameters", {}))

    # Strip a potential "model." / torch.compile "_orig_mod." prefix
    cleaned = {}
    for k, v in state_dict.items():
        for prefix in ("_orig_mod.", "model."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        cleaned[k] = v

    if "channel_number" in hparams:
        channel_number = int(hparams["channel_number"])
    else:  # infer from the first conv layer
        channel_number = int(cleaned["feature_extractor.0.conv.weight"].shape[0])
    dropout = float(hparams.get("dropout", 0.1))

    model = SFCN(channel_number=channel_number, dropout=dropout)
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    unexpected = [k for k in unexpected if not k.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint does not match SFCN(channel_number={channel_number}): "
                           f"missing={missing} unexpected={unexpected}")
    model.eval()
    hparams.setdefault("channel_number", channel_number)
    return model, hparams
