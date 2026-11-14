"""brainage: production brain-age inference from raw T1w MRI.

Quick start::

    from brainage import BrainAgePipeline
    pipe = BrainAgePipeline("/path/to/bundle")          # loads model + preprocessing assets once
    result = pipe.predict("sub-01_T1w.nii.gz", chronological_age=63)
    print(result.predicted_age, result.brain_age_gap)
"""

__version__ = "0.1.0"

__all__ = ["BrainAgePipeline", "PredictionResult", "ModelBundle", "Preprocessor", "__version__"]


def __getattr__(name):  # lazy imports keep `brainage --help` fast
    if name in ("BrainAgePipeline", "PredictionResult"):
        from . import pipeline
        return getattr(pipeline, name)
    if name == "ModelBundle":
        from .bundle import ModelBundle
        return ModelBundle
    if name == "Preprocessor":
        from .preprocessing import Preprocessor
        return Preprocessor
    raise AttributeError(name)
