"""End-to-end pipeline: raw T1w NIfTI -> predicted brain age (years)."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .bundle import ModelBundle
from .inference import OnnxAgePredictor
from .preprocessing import Preprocessor

log = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    input_path: str
    predicted_age: Optional[float] = None
    chronological_age: Optional[float] = None
    brain_age_gap: Optional[float] = None       # predicted - chronological
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
                save_preprocessed_dir: Optional[str | Path] = None) -> PredictionResult:
        """Predict the brain age of one image. Raises on failure (see ``predict_many`` for batch-safe use)."""
        t_start = time.perf_counter()
        pre = self.preprocessor.run(image_path, skip_bias_correction=skip_bias_correction,
                                    skip_registration=skip_registration, save_dir=save_preprocessed_dir)
        t0 = time.perf_counter()
        age = float(self.predictor.predict(pre.tensor)[0])
        timings = dict(pre.timings, inference=time.perf_counter() - t0, total=time.perf_counter() - t_start)

        res = PredictionResult(
            input_path=str(image_path), predicted_age=round(age, 3), subject_id=subject_id,
            model_name=self.bundle.name, model_version=self.bundle.version,
            steps=pre.steps + ["sfcn_onnx_inference"],
            timings={k: round(v, 3) for k, v in timings.items()}, saved_files=pre.saved_files)
        if chronological_age is not None:
            res.chronological_age = float(chronological_age)
            res.brain_age_gap = round(age - float(chronological_age), 3)
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
        results = []
        for path, age, sid in zip(paths, ages, ids):
            try:
                results.append(self.predict(path, chronological_age=age, subject_id=sid, **kwargs))
            except Exception as e:  # noqa: BLE001 - one bad scan must not stop a batch
                if fail_fast:
                    raise
                log.error("%s: %s", path, e)
                results.append(PredictionResult(input_path=str(path), subject_id=sid, chronological_age=age,
                                                status="error", error=f"{type(e).__name__}: {e}",
                                                model_name=self.bundle.name, model_version=self.bundle.version))
        return results
