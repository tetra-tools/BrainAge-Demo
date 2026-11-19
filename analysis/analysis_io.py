"""Shared input/output helpers for the thesis analyses in ``analysis/``.

The prediction tables are the TSV files written by the training repository
(``RS-JYC-HK-T1wMNIBrainAgeRegression``), one pair per split:

* ``predictions_{split}_{tag}.tsv``             healthy participants ("healthy file")
* ``predictions_prediction-{split}_{tag}.tsv``  clinical participants ("prediction file")

They are read exactly as in the Figure 13 forest-plot notebook: both files of a split are
concatenated (healthy file first), native original scans are kept
(``augmentation == "native"`` and ``is_original``), then identical rows are removed after
dropping the per-file columns (``DROP_COLS``).

Splits (thesis nomenclature):
* ``test``       -> "Test": used for hyperparameter selection.
* ``validation`` -> "External validation": Cam-CAN + OASIS-3, final validation (main result).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

# Defaults are relative to the repository root (paths given on the command line are
# relative to the current directory, as usual).
DEFAULT_PRED_DIR = Path("data/predictions")
DEFAULT_OUT_DIR = Path("results/analysis")
DEFAULT_MODEL_TAG = "lossmae_h32_lr0.04_batch16_dr0.4_model_best_score"

FILE_TEMPLATES = {
    "healthy_file": "predictions_{split}_{tag}.tsv",
    "prediction_file": "predictions_prediction-{split}_{tag}.tsv",
}
SPLIT_TEST = "test"
SPLIT_EXTERNAL = "validation"
SPLITS = (SPLIT_TEST, SPLIT_EXTERNAL)
SPLIT_TRAIN = "train"          # optional training-set table (--train-file)

# Columns of the prediction tables
COL_PRED = "predicted_age_years"
COL_AGE = "age"
COL_SEX = "sex"
COL_HEALTHY = "healthy"
COL_DATABASE = "database"
COL_SUBJECT = "subject_id"
COL_SESSION = "session_id"
COL_AUG = "augmentation"
COL_ORIGINAL = "is_original"
COL_NEURO_DX = "neurological_diagnosis"
COL_PSYCH_DX = "psychiatric_diagnosis"
REQUIRED_COLUMNS = [COL_PRED, COL_AGE, COL_SEX, COL_HEALTHY, COL_DATABASE, COL_SUBJECT,
                    COL_SESSION, COL_AUG, COL_ORIGINAL, COL_NEURO_DX, COL_PSYCH_DX]
# The training table only needs what the MAE uses (sex is optional: missing -> "other / missing")
TRAIN_REQUIRED_COLUMNS = [COL_PRED, COL_AGE, COL_HEALTHY, COL_DATABASE, COL_SUBJECT, COL_SESSION,
                          COL_AUG, COL_ORIGINAL]

# Per-file columns dropped by the forest-plot notebook before drop_duplicates()
DROP_COLS = ["subject_image_count", "subject_sample_count", "image_augmentation", "apoe",
             "split", "t1_path", "infile_index", "path", "ptfile_index", "augmentation",
             "is_original"]

# Labels of the external databases in the ``database`` column -> display names
EXTERNAL_DATABASES = {"camcan": "Cam-CAN", "oasis3": "OASIS-3"}


def resolve_dir(arg: Optional[str], default: Path) -> Path:
    """Command-line path, or the default anchored at the repository root."""
    return Path(arg) if arg else REPO_ROOT / default


def read_split(pred_dir: Path, split: str, model_tag: str, notes: List[str]) -> pd.DataFrame:
    """All rows (every augmentation) of both files of one split, healthy file first."""
    frames = []
    for source, template in FILE_TEMPLATES.items():
        path = pred_dir / template.format(split=split, tag=model_tag)
        if not path.is_file():
            raise FileNotFoundError(f"Prediction table not found: {path}")
        df = pd.read_csv(path, sep="\t")
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name}: missing column(s) {missing}")
        notes.append(f"{split}/{source}: {len(df):,} rows read from {path.name}.")
        frames.append(df.assign(_source=source, _split=split))
    return pd.concat(frames, ignore_index=True)


def read_all(pred_dir: Path, model_tag: str, notes: List[str]) -> pd.DataFrame:
    """All rows of both splits."""
    return pd.concat([read_split(pred_dir, s, model_tag, notes) for s in SPLITS], ignore_index=True)


def read_training(path: Path, notes: List[str]) -> pd.DataFrame:
    """All rows of the training-set prediction table (same format as the other tables)."""
    if not path.is_file():
        raise FileNotFoundError(f"Training prediction table not found: {path}")
    df = pd.read_csv(path, sep="\t")
    missing = [c for c in TRAIN_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing column(s) {missing}")
    if COL_SEX not in df.columns:
        df[COL_SEX] = None
        notes.append(f"{path.name}: no '{COL_SEX}' column; training scans counted as sex 'other / missing'.")
    notes.append(f"{SPLIT_TRAIN}/training_file: {len(df):,} rows read from {path.name}.")
    return df.assign(_source="training_file", _split=SPLIT_TRAIN)


def native_scans(raw: pd.DataFrame, notes: List[str]) -> pd.DataFrame:
    """Forest-plot rows: native original scans, healthy + prediction files, drop_duplicates()."""
    out = []
    for split in raw["_split"].unique():
        d = raw[raw["_split"] == split]
        kept = d[(d[COL_AUG] == "native") & (d[COL_ORIGINAL] == True)]  # noqa: E712 (as notebook)
        cols = [c for c in kept.columns if c not in DROP_COLS and not c.startswith("_")]
        dedup = kept.drop_duplicates(subset=cols, keep="first")
        notes.append(f"{split}: {len(kept):,} native original rows, {len(dedup):,} after "
                     f"drop_duplicates() ({len(kept) - len(dedup):,} identical rows removed).")
        if dedup.empty:
            raise ValueError(f"{split}: no row with augmentation == 'native' and is_original.")
        out.append(dedup)
    return pd.concat(out)


def subject_key(df: pd.DataFrame) -> pd.Series:
    """Participant identifier, unique across databases."""
    return df[COL_DATABASE].astype(str) + "/" + df[COL_SUBJECT].astype(str)


def scan_key(df: pd.DataFrame) -> pd.Series:
    """Scan identifier (participant + session), unique within a split."""
    return subject_key(df) + "/" + df[COL_SESSION].astype(str)


def markdown_table(df: pd.DataFrame, index_label: Optional[str] = None) -> str:
    """Plain Markdown table (no dependency on ``tabulate``)."""
    d = df.reset_index() if index_label is not None else df
    if index_label is not None:
        d = d.rename(columns={d.columns[0]: index_label})
    header = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    def cell(v) -> str:
        return "" if pd.isna(v) else str(v).replace("|", "\\|").replace("\n", " ")

    rows = ["| " + " | ".join(cell(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([header, sep, *rows])
