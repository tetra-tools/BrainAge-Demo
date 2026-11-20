"""End-to-end pipeline: raw T1w NIfTI -> predicted brain age (years)."""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .bundle import ModelBundle
from .inference import OnnxAgePredictor
from .preprocessing import Preprocessor, default_output_stem

log = logging.getLogger(__name__)


def floor_age(age) -> Optional[float]:
    """Chronological age rounded down to whole years (63.9 -> 63.0).

    Design rule: the model was trained on ages rounded down to whole years, so every
    chronological age is rounded down before it is stored or used (brain-age gap included).
    The predicted age is the model output and is not rounded.

    A missing age (``None``, empty string, NaN as in pandas) gives ``None``: no brain-age gap.
    Raises ``ValueError`` for a value that is not a number, infinite, or negative.
    """
    if age is None:
        return None
    if isinstance(age, str):
        age = age.strip()
        if not age:
            return None
    try:
        value = float(age)
    except (TypeError, ValueError):
        raise ValueError(f"Chronological age must be a number, got {age!r}") from None
    if math.isnan(value):
        return None
    if math.isinf(value) or value < 0:
        raise ValueError(f"Chronological age must be a finite number >= 0, got {age!r}")
    return float(math.floor(value))


def _floor_age_or_none(age) -> Optional[float]:
    """``floor_age`` for error rows: an invalid age is reported as missing, not raised again."""
    try:
        return floor_age(age)
    except ValueError:
        return None


@dataclass
class PredictionResult:
    input_path: str
    predicted_age: Optional[float] = None
    chronological_age: Optional[float] = None   # rounded down to whole years (floor_age)
    brain_age_gap: Optional[float] = None       # predicted - chronological (rounded down)
    subject_id: Optional[str] = None
    status: str = "ok"                          # "ok" | "error"
    error: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    steps: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    saved_files: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class BrainAgePipeline:
    """Load once, predict many.

    >>> pipe = BrainAgePipeline("/opt/brainage/model")
    >>> pipe.predict("sub-01_T1w.nii.gz", chronological_age=63).predicted_age
    """

    def __init__(self, bundle: str | Path | ModelBundle | None = None,
                 providers: Optional[Sequence[str]] = None,
                 num_threads: Optional[int] = None,
                 verify: bool = True):
        self.bundle = bundle if isinstance(bundle, ModelBundle) else ModelBundle.load(bundle, verify=verify)
        self.preprocessor = Preprocessor(num_threads=num_threads, **self.bundle.preprocessor_kwargs())
        self.predictor = OnnxAgePredictor(self.bundle.model_path, providers=providers, num_threads=num_threads)
        log.info("Loaded %s v%s (providers: %s)", self.bundle.name, self.bundle.version,
                 ", ".join(self.predictor.providers))

    def predict(self, image_path: str | Path, chronological_age: Optional[float] = None,
                subject_id: Optional[str] = None, skip_bias_correction: bool = False,
                skip_registration: bool = False,
                save_preprocessed_dir: Optional[str | Path] = None,
                output_stem: Optional[str] = None) -> PredictionResult:
        """Predict the brain age of one image. Raises on failure (see ``predict_many`` for batch-safe use).

        ``chronological_age`` is rounded down to whole years (``floor_age``), as the training
        targets; ``brain_age_gap`` is ``predicted_age`` minus that rounded-down age.

        ``output_stem`` names the QC files written to ``save_preprocessed_dir`` (default: subject_id,
        else the input file name).
        """
        chronological_age = floor_age(chronological_age)  # checked before the slow preprocessing
        t_start = time.perf_counter()
        pre = self.preprocessor.run(image_path, skip_bias_correction=skip_bias_correction,
                                    skip_registration=skip_registration, save_dir=save_preprocessed_dir,
                                    output_stem=output_stem or subject_id)
        t0 = time.perf_counter()
        age = float(self.predictor.predict(pre.tensor)[0])
        if not math.isfinite(age):
            raise RuntimeError(f"The model returned a non-finite age ({age}) for {image_path}")
        timings = dict(pre.timings, inference=time.perf_counter() - t0, total=time.perf_counter() - t_start)

        res = PredictionResult(
            input_path=str(image_path), predicted_age=round(age, 3), subject_id=subject_id,
            model_name=self.bundle.name, model_version=self.bundle.version,
            steps=pre.steps + ["sfcn_onnx_inference"],
            timings={k: round(v, 3) for k, v in timings.items()}, saved_files=pre.saved_files)
        if chronological_age is not None:
            res.chronological_age = chronological_age
            res.brain_age_gap = round(age - chronological_age, 3)
        log.info("%s -> %.2f years (%.1fs)", image_path, age, timings["total"])
        return res

    def predict_many(self, image_paths: Iterable[str | Path],
                     chronological_ages: Optional[Iterable[Optional[float]]] = None,
                     subject_ids: Optional[Iterable[Optional[str]]] = None,
                     fail_fast: bool = False, **kwargs) -> List[PredictionResult]:
        """Predict several images sequentially. Errors are recorded per image unless ``fail_fast``."""
        paths = list(image_paths)
        ages = list(chronological_ages) if chronological_ages is not None else [None] * len(paths)
        ids = list(subject_ids) if subject_ids is not None else [None] * len(paths)
        if not (len(ages) == len(ids) == len(paths)):
            raise ValueError("image_paths, chronological_ages and subject_ids must have the same length")
        stems = unique_output_stems(paths, ids)
        results = []
        for path, age, sid, stem in zip(paths, ages, ids, stems):
            try:
                results.append(self.predict(path, chronological_age=age, subject_id=sid,
                                            output_stem=stem, **kwargs))
            except Exception as e:  # noqa: BLE001 - one bad scan must not stop a batch
                if fail_fast:
                    raise
                log.error("%s: %s", path, e)
                results.append(PredictionResult(input_path=str(path), subject_id=sid,
                                                chronological_age=_floor_age_or_none(age),
                                                status="error", error=f"{type(e).__name__}: {e}",
                                                model_name=self.bundle.name, model_version=self.bundle.version))
        return results


def unique_output_stems(paths: List[str | Path], subject_ids: List[Optional[str]]) -> List[str]:
    """One distinct QC file stem per input, so a batch never overwrites its own outputs.

    Default stem: subject_id, else the file name. Inputs sharing a stem (e.g. ``a/T1w.nii.gz`` and
    ``b/T1w.nii.gz``) are prefixed with their parent folder, then numbered if still ambiguous.
    """
    stems = [sid or default_output_stem(Path(p)) for p, sid in zip(paths, subject_ids)]
    counts = Counter(stems)
    stems = [f"{Path(p).parent.name}_{s}" if counts[s] > 1 else s for p, s in zip(paths, stems)]
    counts, seen, out = Counter(stems), Counter(), []
    for s in stems:
        if counts[s] > 1:
            seen[s] += 1
            s = f"{s}_{seen[s]}"
        out.append(s)
    return out
