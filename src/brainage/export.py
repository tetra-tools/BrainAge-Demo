"""Convert a trained Lightning checkpoint into a production model bundle.

    brainage export \
        --checkpoint checkpoints/sfcn/lossmae_h32_lr0.04_batch16_dr0.4/model_best_score.ckpt \
        --landmarks cached_data/t1_landmarks.npy \
        --template /ix1/hkarim/team/references/ravg152T1_1mm.nii \
        --mask /ix1/hkarim/team/references/ravg152T1_1mm_defaced.nii \
        --version 1.0.0 --out dist/brainage-sfcn-1.0.0
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from . import __version__
from . import constants as C
from .bundle import ModelBundle, sha256sum

log = logging.getLogger(__name__)


def _nifti_ext(path: Path) -> str:
    return ".nii.gz" if path.name.endswith(".nii.gz") else ".nii"


def export_onnx(model, out_path: Path, opset: int = 18) -> None:
    import torch

    # batch of 2: torch.export specialises a dimension whose example size is 1
    dummy = torch.zeros(2, 1, *C.TARGET_SHAPE, dtype=torch.float32)
    names = dict(input_names=[C.ONNX_INPUT_NAME], output_names=[C.ONNX_OUTPUT_NAME], opset_version=opset)
    with torch.no_grad():
        try:  # torch.export-based exporter (default since torch 2.9, needs onnxscript)
            batch = torch.export.Dim("batch", min=1, max=1024)
            torch.onnx.export(model, (dummy,), str(out_path), dynamo=True, external_data=False,
                              dynamic_shapes={"x": {0: batch}}, **names)
            return
        except Exception as e:  # noqa: BLE001 - fall back to the TorchScript exporter
            log.warning("torch.export ONNX exporter unavailable (%s); using the TorchScript exporter", e)
            for leftover in (out_path, Path(str(out_path) + ".data")):
                leftover.unlink(missing_ok=True)
        dynamic_axes = {C.ONNX_INPUT_NAME: {0: "batch"}, C.ONNX_OUTPUT_NAME: {0: "batch"}}
        try:
            torch.onnx.export(model, (dummy,), str(out_path), dynamo=False, dynamic_axes=dynamic_axes,
                              do_constant_folding=True, **names)
        except TypeError:  # torch < 2.5 has no `dynamo` argument
            torch.onnx.export(model, (dummy,), str(out_path), dynamic_axes=dynamic_axes,
                              do_constant_folding=True, **names)


def export_bundle(checkpoint: str | Path, landmarks: str | Path, template: str | Path, mask: str | Path,
                  out_dir: str | Path, version: str, name: str = "brainage-sfcn", opset: int = 18,
                  mask_threshold: float = C.MASK_THRESHOLD,
                  registration: str = C.REGISTRATION_TRANSFORM,
                  notes: str = "", force: bool = False, parity_tolerance: float = 1e-2) -> ModelBundle:
    checkpoint, landmarks, template, mask = map(Path, (checkpoint, landmarks, template, mask))
    for p in (checkpoint, landmarks, template, mask):
        if not p.is_file():
            raise FileNotFoundError(f"File not found: {p}")
    out_dir = Path(out_dir)
    if out_dir.exists():
        if not force:
            raise FileExistsError(f"{out_dir} exists (use --force to overwrite)")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    try:
        _populate_bundle(checkpoint, landmarks, template, mask, out_dir, version, name, opset,
                         mask_threshold, registration, notes, parity_tolerance)
    except BaseException:
        shutil.rmtree(out_dir, ignore_errors=True)  # never leave a half-written bundle behind
        raise
    log.info("Bundle written to %s", out_dir)
    return ModelBundle.load(out_dir)


def _populate_bundle(checkpoint: Path, landmarks: Path, template: Path, mask: Path, out_dir: Path,
                     version: str, name: str, opset: int, mask_threshold: float, registration: str,
                     notes: str, parity_tolerance: float) -> None:
    import torch
    import nibabel as nib
    from .inference import OnnxAgePredictor
    from .model import load_lightning_checkpoint

    # 1. model -> ONNX
    log.info("Loading checkpoint %s", checkpoint)
    model, hparams = load_lightning_checkpoint(checkpoint)
    onnx_path = out_dir / "model.onnx"
    log.info("Exporting ONNX (opset %d)", opset)
    export_onnx(model, onnx_path, opset=opset)

    # 2. parity check torch vs onnxruntime on a realistic-range random batch
    rng = np.random.default_rng(0)
    x = (rng.random((2, 1, *C.TARGET_SHAPE), dtype=np.float32) * 100.0)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).double().numpy()
    got = OnnxAgePredictor(onnx_path, providers=["CPUExecutionProvider"]).predict(x)
    if not (np.isfinite(ref).all() and np.isfinite(got).all()):
        raise RuntimeError("The model outputs NaN/Inf (non-finite weights in the checkpoint?): bundle not written")
    max_diff = float(np.max(np.abs(ref - got)))
    log.info("Parity torch vs ONNX: max |diff| = %.2e years", max_diff)
    if not np.isfinite(max_diff) or max_diff > parity_tolerance:
        raise RuntimeError(f"ONNX output differs from PyTorch by {max_diff:.3g} years")

    # 3. preprocessing assets
    lm = np.load(landmarks, allow_pickle=False)
    if lm.ndim != 1:
        raise ValueError(f"landmarks must be a 1D array, got {lm.shape}")
    shutil.copyfile(landmarks, out_dir / "t1_landmarks.npy")
    tpl_name = "template" + _nifti_ext(template)
    mask_name = "brain_mask" + _nifti_ext(mask)
    shutil.copyfile(template, out_dir / tpl_name)
    shutil.copyfile(mask, out_dir / mask_name)

    tpl_img, mask_img = nib.load(str(template)), nib.load(str(mask))
    if tpl_img.shape[:3] != mask_img.shape[:3] or not np.allclose(tpl_img.affine, mask_img.affine, atol=1e-2):
        raise ValueError(f"template {tpl_img.shape} and mask {mask_img.shape} are not on the same grid")

    # 4. manifest
    files = {p.name: sha256sum(p) for p in sorted(out_dir.iterdir())}
    manifest = {
        "format_version": C.BUNDLE_FORMAT_VERSION,
        "name": name,
        "version": version,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "created_with": f"brainage {__version__}, torch {torch.__version__}",
        "model": {
            "file": "model.onnx",
            "architecture": "SFCN",
            "channel_number": int(hparams["channel_number"]),
            "input_name": C.ONNX_INPUT_NAME,
            "output_name": C.ONNX_OUTPUT_NAME,
            "input_shape": ["batch", 1, *C.TARGET_SHAPE],
            "dtype": "float32",
            "output_unit": "years",
            "opset": opset,
            "onnx_parity_max_abs_diff_years": max_diff,
        },
        "training": {
            "source_checkpoint": checkpoint.name,
            "source_checkpoint_sha256": sha256sum(checkpoint),
            "hyper_parameters": {k: v for k, v in hparams.items() if isinstance(v, (int, float, str, bool))},
        },
        "preprocessing": {
            "landmarks": "t1_landmarks.npy",
            "template": tpl_name,
            "template_source": template.name,
            "brain_mask": mask_name,
            "brain_mask_source": mask.name,
            "mask_threshold": mask_threshold,
            "registration": registration,
            "registration_seed": C.REGISTRATION_RANDOM_SEED,
            "n4": {"shrink_factor": C.N4_SHRINK_FACTOR, "convergence": C.N4_CONVERGENCE,
                   "spline_param": C.N4_SPLINE_PARAM},
            "rescale_percentiles": list(C.RESCALE_PERCENTILES),
            "target_shape": list(C.TARGET_SHAPE),
            "fp16_roundtrip": C.FP16_ROUNDTRIP,
        },
        "files": files,
        "notes": notes,
    }
    (out_dir / C.MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")


def archive_bundle(bundle_dir: str | Path, out_file: Optional[str | Path] = None) -> Path:
    """Pack a bundle directory as ``<name>.tar.gz`` (+ ``.sha256``) for upload to a release / registry."""
    import tarfile

    bundle_dir = Path(bundle_dir)
    out_file = Path(out_file) if out_file else bundle_dir.with_name(bundle_dir.name + ".tar.gz")
    with tarfile.open(out_file, "w:gz") as tar:
        tar.add(bundle_dir, arcname=bundle_dir.name)
    Path(str(out_file) + ".sha256").write_text(f"{sha256sum(out_file)}  {out_file.name}\n")
    return out_file
