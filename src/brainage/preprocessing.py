"""Production preprocessing: raw T1w NIfTI -> model input tensor (1, 1, 160, 192, 160).

Steps (no data augmentation):

1. N4 bias-field correction (ANTsPy)                     -> "desc-biascorr"
2. Rigid + affine registration to avg152T1 1 mm (ANTsPy) -> "space-avg152T1"
3. The registered volume is written to NIfTI and re-read with
   ``tio.ScalarImage``, exactly like training read its NIfTI files
   (no in-memory ANTs->TorchIO conversion, which silently mixes LPS and RAS).
4. TorchIO transforms copied verbatim from ``generate_ptfiles.py``:
   NaN/Inf cleaning -> RescaleIntensity(0-1, p0.5-p99.5) ->
   HistogramStandardization(training landmarks) -> brain Mask -> CropOrPad(160,192,160)
5. float16 round-trip (training tensors were cached as float16).

Steps 1 and 2 can be skipped for inputs already bias-corrected and/or already
in the template grid (e.g. the "space-avg152T1_desc-biascorr" BIDS derivatives
used for training).
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import constants as C

log = logging.getLogger(__name__)


class PreprocessingError(RuntimeError):
    pass


def clean_image_data(data: torch.Tensor) -> torch.Tensor:
    """Replace NaN / +-Inf by 0 (identical to generate_ptfiles.clean_image_data)."""
    if data.dtype != torch.float32:
        data = data.float()
    data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if torch.isnan(data).any() or torch.isinf(data).any():
        data = torch.where(torch.isnan(data) | torch.isinf(data), torch.tensor(0.0), data)
    return data


def _stem(path: Path) -> str:
    name = path.name
    for ext in (".nii.gz", ".nii", ".mgz", ".nrrd", ".mha", ".mhd"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return path.stem


def syn_quick_affine_args(fixed: str, moving: str, outprefix: str, warped: str, inverse_warped: str,
                          random_seed: Optional[int] = C.REGISTRATION_RANDOM_SEED) -> List[str]:
    """``antsRegistration`` arguments of ``antsRegistrationSyNQuick.sh -d 3 -t a`` (rigid + affine).

    This is the command that produced the training derivatives (``space-avg152T1_desc-biascorr``,
    BIDS conversion scripts). ``ants.registration(type_of_transform="antsRegistrationSyNQuick[a]")``
    omits ``--winsorize-image-intensities``, hence this explicit version. Only ``--float 1``
    differs from the script (which runs in double precision): ANTsPy passes float images in memory.
    ``--random-seed`` is added so that runs with one ITK thread are bit-reproducible.
    """
    linear = "MI[%s,%s,1,32,Regular,0.25]" % (fixed, moving)
    stage = ["--convergence", "[1000x500x250x0,1e-6,10]", "--shrink-factors", "8x4x2x1",
             "--smoothing-sigmas", "3x2x1x0vox"]
    return [
        "--dimensionality", "3",
        "--float", "1",
        "--collapse-output-transforms", "1",
        "--output", "[%s,%s,%s]" % (outprefix, warped, inverse_warped),
        "--interpolation", "Linear",
        "--use-histogram-matching", "0",
        "--winsorize-image-intensities", "[0.005,0.995]",
        "--initial-moving-transform", "[%s,%s,1]" % (fixed, moving),
        "--transform", "Rigid[0.1]", "--metric", linear, *stage,
        "--transform", "Affine[0.1]", "--metric", linear, *stage,
    ] + ([] if random_seed is None else ["--random-seed", str(int(random_seed))])


def _syn_quick_affine(fixed_img, moving_img, random_seed: Optional[int] = C.REGISTRATION_RANDOM_SEED):
    from ants.internal import get_lib_fn, get_pointer_string, process_arguments

    fixed = fixed_img.clone("float")
    moving = moving_img.clone("float")
    warped = fixed.clone()          # receives the moving image resampled on the template grid
    inverse_warped = moving.clone()
    with tempfile.TemporaryDirectory(prefix="brainage_reg_") as tmp:
        args = syn_quick_affine_args(get_pointer_string(fixed), get_pointer_string(moving),
                                     str(Path(tmp) / "reg_"), get_pointer_string(warped),
                                     get_pointer_string(inverse_warped), random_seed=random_seed)
        code = get_lib_fn("antsRegistration")(process_arguments(args))
    if code != 0:
        raise PreprocessingError(f"antsRegistration failed with exit code {code}")
    return warped


def default_output_stem(path: Path) -> str:
    """File stem used for QC outputs: ``sub-01_ses-1_T1w.nii.gz`` -> ``sub-01_ses-1``."""
    stem = _stem(path)
    return stem[:-4] if stem.endswith("_T1w") else stem


@dataclass
class PreprocessedImage:
    tensor: np.ndarray                      # (1, 1, 160, 192, 160) float32
    steps: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    saved_files: Dict[str, str] = field(default_factory=dict)


class Preprocessor:
    def __init__(
        self,
        template_path: str | Path,
        mask_path: str | Path,
        landmarks_path: str | Path,
        mask_threshold: float = C.MASK_THRESHOLD,
        registration_transform: str = C.REGISTRATION_TRANSFORM,
        registration_seed: Optional[int] = C.REGISTRATION_RANDOM_SEED,
        target_shape: Sequence[int] = C.TARGET_SHAPE,
        rescale_percentiles: Tuple[float, float] = C.RESCALE_PERCENTILES,
        n4_shrink_factor: int = C.N4_SHRINK_FACTOR,
        n4_convergence: Optional[dict] = None,
        n4_spline_param=C.N4_SPLINE_PARAM,
        fp16_roundtrip: bool = C.FP16_ROUNDTRIP,
        num_threads: Optional[int] = None,
    ):
        import torchio as tio  # imported lazily: heavy

        self.template_path = Path(template_path)
        self.mask_path = Path(mask_path)
        self.landmarks_path = Path(landmarks_path)
        for p in (self.template_path, self.mask_path, self.landmarks_path):
            if not p.is_file():
                raise FileNotFoundError(f"File not found: {p}")

        self.registration_transform = registration_transform
        self.registration_seed = registration_seed
        self.target_shape = tuple(int(s) for s in target_shape)
        self.n4_shrink_factor = n4_shrink_factor
        self.n4_convergence = n4_convergence or dict(C.N4_CONVERGENCE)
        self.n4_spline_param = n4_spline_param
        self.fp16_roundtrip = fp16_roundtrip

        if num_threads:
            # ITK reads this once, when ANTs first runs a filter: it must be set before that.
            if "ants" in sys.modules and os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS") != str(num_threads):
                log.warning("ANTs is already loaded: num_threads=%s may not apply to N4/registration. "
                            "Set ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS before importing ants.", num_threads)
            os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(num_threads)
            torch.set_num_threads(num_threads)

        # Landmarks: a plain np.ndarray saved by np.save(tio.HistogramStandardization.train(...))
        landmarks = np.load(self.landmarks_path, allow_pickle=False)
        if landmarks.ndim != 1:
            raise PreprocessingError(f"Unexpected landmarks shape {landmarks.shape} in {self.landmarks_path}")
        self.landmarks = landmarks

        # Brain mask exactly as in training: ScalarImage(...).data > threshold
        mask_img = tio.ScalarImage(self.mask_path)
        self.brain_mask = tio.LabelMap(tensor=(mask_img.data > mask_threshold).float(), affine=mask_img.affine)

        self.transform = tio.Compose([
            tio.Lambda(clean_image_data, types_to_apply=[tio.INTENSITY]),
            tio.RescaleIntensity(out_min_max=C.RESCALE_OUT_MIN_MAX, percentiles=tuple(rescale_percentiles)),
            tio.HistogramStandardization({"t1": self.landmarks}),
            tio.Mask(masking_method="brain"),
            tio.CropOrPad(target_shape=self.target_shape, padding_mode=0),
        ], exclude=["brain"])

        self._template = None  # ANTs image, loaded on first registration

    # ----------------------------------------------------------------- ANTs
    @property
    def template(self):
        import ants
        if self._template is None:
            self._template = ants.image_read(str(self.template_path))
        return self._template

    def bias_correct(self, image):
        import ants
        return ants.n4_bias_field_correction(
            image,
            shrink_factor=self.n4_shrink_factor,
            convergence=self.n4_convergence,
            spline_param=self.n4_spline_param,
            verbose=False,
            return_bias_field=False,
        )

    def register(self, image):
        """Register ``image`` to the template and return it resampled on the template grid."""
        import ants
        if self.registration_transform == C.REGISTRATION_TRANSFORM:
            return _syn_quick_affine(self.template, image, random_seed=self.registration_seed)
        # any other value: an ANTsPy type_of_transform (e.g. "Affine", "antsRegistrationSyNQuick[a]")
        return ants.registration(fixed=self.template, moving=image,
                                 type_of_transform=self.registration_transform, verbose=False)["warpedmovout"]

    # ------------------------------------------------------------- pipeline
    def _check_grid(self, t1, source: Path) -> None:
        mask = self.brain_mask
        same_shape = tuple(t1.spatial_shape) == tuple(mask.spatial_shape)
        same_affine = same_shape and np.allclose(t1.affine, mask.affine, atol=1e-2)
        if not (same_shape and same_affine):
            raise PreprocessingError(
                f"{source}: image grid {tuple(t1.spatial_shape)} does not match the template/mask grid "
                f"{tuple(mask.spatial_shape)} (or their affines differ). Do not use --skip-registration "
                f"for images that are not already in template space.")

    def run(
        self,
        image_path: str | Path,
        skip_bias_correction: bool = False,
        skip_registration: bool = False,
        save_dir: Optional[str | Path] = None,
        output_stem: Optional[str] = None,
    ) -> PreprocessedImage:
        import ants
        import torchio as tio

        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"File not found: {image_path}")
        out = PreprocessedImage(tensor=np.empty(0))

        t0 = time.perf_counter()
        img = ants.image_read(str(image_path))
        if img.dimension == 4 and img.shape[3] == 1:
            img = ants.slice_image(img, axis=3, idx=0)
        if img.dimension != 3:
            raise PreprocessingError(f"{image_path}: expected a 3D T1w volume, got {img.dimension}D {img.shape}")
        out.timings["load"] = time.perf_counter() - t0

        if not skip_bias_correction:
            t0 = time.perf_counter()
            img = self.bias_correct(img)
            out.timings["n4_bias_correction"] = time.perf_counter() - t0
            out.steps.append("n4_bias_correction")

        if not skip_registration:
            t0 = time.perf_counter()
            img = self.register(img)
            out.timings["registration"] = time.perf_counter() - t0
            out.steps.append(f"registration:{self.registration_transform}")

        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="brainage_") as tmp:
            nii = Path(tmp) / "t1.nii.gz"
            ants.image_write(img, str(nii))
            t1 = tio.ScalarImage(nii)
            t1.load()  # read now, before the temp dir disappears
        self._check_grid(t1, image_path)

        subject = tio.Subject(t1=t1, brain=self.brain_mask)
        processed = self.transform(subject)
        tensor = processed.t1.data
        if self.fp16_roundtrip:
            tensor = tensor.to(torch.float16)
        tensor = tensor.to(torch.float32).unsqueeze(0).contiguous()  # (1, 1, D, H, W)
        if tuple(tensor.shape[2:]) != self.target_shape:
            raise PreprocessingError(f"Unexpected preprocessed shape {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise PreprocessingError("Preprocessed tensor contains NaN/Inf")
        out.timings["intensity_and_crop"] = time.perf_counter() - t0
        out.steps += ["rescale_intensity", "histogram_standardization", "brain_mask", "crop_or_pad"]

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            stem = output_stem or default_output_stem(image_path)
            registered = save_dir / f"{stem}_space-template_desc-biascorr_T1w.nii.gz"
            ants.image_write(img, str(registered))
            model_input = save_dir / f"{stem}_desc-modelinput_T1w.nii.gz"
            tio.ScalarImage(tensor=tensor[0], affine=processed.t1.affine).save(model_input)
            out.saved_files = {"registered": str(registered), "model_input": str(model_input)}

        out.tensor = tensor.numpy()
        return out
