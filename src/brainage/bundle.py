"""Model bundle: the self-contained, versioned artefact used in production.

A bundle is a directory::

    brainage-sfcn-1.0.0/
    ├── manifest.json        # metadata, preprocessing parameters, SHA-256 of every file
    ├── model.onnx           # SFCN exported to ONNX (float32, dynamic batch)
    ├── t1_landmarks.npy     # histogram-standardisation landmarks used in training
    ├── template.nii.gz      # registration target (avg152T1 1 mm)
    └── brain_mask.nii.gz    # brain mask source (ravg152T1_1mm_defaced, thresholded at runtime)

The model is only valid together with the exact landmarks / template / mask it
was trained with, which is why they travel together and are checksummed.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from . import constants as C


class BundleError(RuntimeError):
    pass


def sha256sum(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class ModelBundle:
    root: Path
    manifest: Dict[str, Any] = field(repr=False)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: Optional[str | Path] = None, verify: bool = True) -> "ModelBundle":
        """Load a bundle directory. ``path`` defaults to ``$BRAINAGE_BUNDLE`` then ``/opt/brainage/model``."""
        if path is None:
            path = os.environ.get(C.DEFAULT_BUNDLE_ENV, C.DEFAULT_BUNDLE_PATH)
        root = Path(path).expanduser().resolve()
        manifest_path = root / C.MANIFEST_NAME
        if not manifest_path.is_file():
            raise BundleError(
                f"No {C.MANIFEST_NAME} in '{root}'. Pass --bundle / BRAINAGE_BUNDLE pointing to a model "
                f"bundle (see docs/MODEL_FORMAT.md, created with `brainage export`).")
        manifest = json.loads(manifest_path.read_text())
        fmt = manifest.get("format_version")
        if fmt != C.BUNDLE_FORMAT_VERSION:
            raise BundleError(f"Unsupported bundle format_version={fmt} (expected {C.BUNDLE_FORMAT_VERSION})")
        bundle = cls(root=root, manifest=manifest)
        bundle._check_files(verify)
        return bundle

    def _check_files(self, verify: bool) -> None:
        for rel, expected in self.manifest.get("files", {}).items():
            p = self.root / rel
            if not p.is_file():
                raise BundleError(f"Bundle file missing: {p}")
            if verify and sha256sum(p) != expected:
                raise BundleError(f"Checksum mismatch for {p}: the bundle is corrupted or was modified")

    # --------------------------------------------------------------- accessors
    @property
    def name(self) -> str:
        return self.manifest.get("name", "brainage")

    @property
    def version(self) -> str:
        return self.manifest.get("version", "unknown")

    @property
    def model_path(self) -> Path:
        return self.root / self.manifest["model"]["file"]

    @property
    def preprocessing(self) -> Dict[str, Any]:
        return self.manifest.get("preprocessing", {})

    def _pp_file(self, key: str) -> Path:
        try:
            return self.root / self.preprocessing[key]
        except KeyError as e:
            raise BundleError(f"manifest.preprocessing.{key} is missing") from e

    @property
    def landmarks_path(self) -> Path:
        return self._pp_file("landmarks")

    @property
    def template_path(self) -> Path:
        return self._pp_file("template")

    @property
    def mask_path(self) -> Path:
        return self._pp_file("brain_mask")

    def preprocessor_kwargs(self) -> Dict[str, Any]:
        """Arguments for :class:`brainage.preprocessing.Preprocessor` taken from the manifest."""
        pp = self.preprocessing
        n4 = pp.get("n4", {})
        return dict(
            template_path=self.template_path,
            mask_path=self.mask_path,
            landmarks_path=self.landmarks_path,
            mask_threshold=float(pp.get("mask_threshold", C.MASK_THRESHOLD)),
            registration_transform=pp.get("registration", C.REGISTRATION_TRANSFORM),
            registration_seed=int(pp.get("registration_seed", C.REGISTRATION_RANDOM_SEED)),
            target_shape=tuple(pp.get("target_shape", C.TARGET_SHAPE)),
            rescale_percentiles=tuple(pp.get("rescale_percentiles", C.RESCALE_PERCENTILES)),
            n4_shrink_factor=int(n4.get("shrink_factor", C.N4_SHRINK_FACTOR)),
            n4_convergence=n4.get("convergence", C.N4_CONVERGENCE),
            n4_spline_param=n4.get("spline_param", C.N4_SPLINE_PARAM),
            fp16_roundtrip=bool(pp.get("fp16_roundtrip", C.FP16_ROUNDTRIP)),
        )

    def summary(self) -> str:
        m = self.manifest
        lines = [f"Bundle   : {self.root}",
                 f"Name     : {self.name}  version {self.version}",
                 f"Created  : {m.get('created_utc', '?')}",
                 f"Model    : {m['model'].get('architecture')} (channels={m['model'].get('channel_number')}), "
                 f"ONNX opset {m['model'].get('opset')}, output in {m['model'].get('output_unit', 'years')}",
                 f"Registr. : {self.preprocessing.get('registration')}",
                 "Files    :"]
        for rel, digest in m.get("files", {}).items():
            lines.append(f"  - {rel}  sha256={digest[:16]}…")
        if m.get("training"):
            lines.append(f"Training : {json.dumps(m['training'])}")
        return "\n".join(lines)
