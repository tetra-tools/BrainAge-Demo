"""Model runtimes. Production uses ONNX Runtime; the PyTorch runtime exists for
validation (parity checks when exporting a checkpoint)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)


class OnnxAgePredictor:
    def __init__(self, model_path: str | Path, providers: Optional[Sequence[str]] = None,
                 num_threads: Optional[int] = None):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError("ONNX Runtime is not installed: install `brainage[cpu]` (onnxruntime) "
                               "or `brainage[gpu]` (onnxruntime-gpu), not both") from e

        available = ort.get_available_providers()
        if providers is None:
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        else:
            unknown = [p for p in providers if p not in available]
            if unknown:
                raise ValueError(f"ONNX providers not available: {unknown} (available: {available})")
        opts = ort.SessionOptions()
        if num_threads:
            opts.intra_op_num_threads = int(num_threads)
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=list(providers))
        self.input_name = self.session.get_inputs()[0].name
        log.debug("ONNX session ready, providers=%s", self.session.get_providers())

    @property
    def providers(self):
        return self.session.get_providers()

    def predict(self, batch: np.ndarray) -> np.ndarray:
        """``batch``: (N, 1, 160, 192, 160) float32 -> (N,) ages in years."""
        batch = np.asarray(batch, dtype=np.float32)
        if batch.ndim == 4:
            batch = batch[:, None]
        (out,) = self.session.run(None, {self.input_name: batch})
        return np.asarray(out, dtype=np.float64).reshape(-1)


class TorchAgePredictor:
    def __init__(self, checkpoint_path: str | Path, device: str = "cpu"):
        import torch
        from .model import load_lightning_checkpoint

        self.torch = torch
        self.model, self.hparams = load_lightning_checkpoint(checkpoint_path)
        self.device = device
        self.model.to(device).eval()

    def predict(self, batch: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            x = self.torch.as_tensor(np.asarray(batch, dtype=np.float32), device=self.device)
            if x.ndim == 4:
                x = x.unsqueeze(1)
            return self.model(x).double().cpu().numpy().reshape(-1)
