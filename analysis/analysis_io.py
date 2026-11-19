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

import re
from pathlib import Path
from typing import List, Optional, Tuple

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
TRAIN_FILE_TEMPLATE = "predictions_train_{tag}.tsv"

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

# Transform names found in the ``augmentation`` strings of augmented rows
AUGMENTATION_TRANSFORMS = ("Flip", "Blur", "Noise", "Motion", "Ghosting", "Spike")

# Identifiers are read as text: numeric-looking IDs keep their leading zeros ("007" != "7")
ID_DTYPES = {COL_SUBJECT: str, COL_SESSION: str, COL_DATABASE: str}

# Characters that usually come from a copy-paste or an encoding problem in free text
SUSPECT_TEXT = re.compile("[\u2018\u2019\u201c\u201d\ufffd\u00a0]|\u00c3|\u00e2\u20ac")


class DataError(ValueError):
    """Labelling or encoding problem that would silently drop scans."""


def read_tsv(path: Path) -> pd.DataFrame:
    """Read a prediction table: UTF-8 (byte-order mark tolerated), identifiers as text."""
    try:
        return pd.read_csv(path, sep="\t", dtype=ID_DTYPES, encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DataError(f"{path.name} is not UTF-8 encoded (byte {exc.start}: {exc.object[exc.start:exc.start + 1]!r}); "
                        "re-save it as UTF-8.") from exc


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
        df = read_tsv(path)
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
    df = read_tsv(path)
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
        if set(scan_key(kept)) != set(scan_key(dedup)):        # never expected: drop_duplicates lost a scan
            raise DataError(f"{split}: drop_duplicates() removed every row of some scans.")
        notes.append(f"{split}: {len(kept):,} native original rows, {len(dedup):,} after "
                     f"drop_duplicates() ({len(kept) - len(dedup):,} identical rows removed).")
        if dedup.empty:
            raise ValueError(f"{split}: no row with augmentation == 'native' and is_original.")
        out.append(dedup)
    return pd.concat(out)


def _examples(values, n: int = 8) -> str:
    values = list(values)
    shown = ", ".join(repr(v) for v in values[:n])
    return shown + (f", ... ({len(values):,} in total)" if len(values) > n else "")


def _norm_label(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)


def audit_labels(raw: pd.DataFrame, native: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Find labels or encodings that would silently drop scans or participants.

    Returns (errors, warnings). Errors are problems that remove scans from the analyses
    (or put them in the wrong group); warnings are kept scans that deserve a look.
    """
    errors: List[str] = []
    warnings: List[str] = []
    aug = raw[COL_AUG]
    aug_txt = aug.astype("string")
    is_orig = raw[COL_ORIGINAL] == True  # noqa: E712 (as notebook)

    # numbers stored as text (e.g. decimal commas) would crash or be lost
    for col in (COL_PRED, COL_AGE):
        num = pd.to_numeric(raw[col], errors="coerce")
        bad = raw.loc[num.isna() & raw[col].notna(), col]
        if len(bad):
            errors.append(f"'{col}': {len(bad):,} values are not numbers: {_examples(bad.unique())}.")

    # augmentation labels
    near = aug_txt.str.strip().str.lower().eq("native") & aug_txt.ne("native")
    if near.any():
        errors.append(f"'{COL_AUG}': {int(near.sum()):,} rows labelled {_examples(aug[near].unique())} instead of "
                      "exactly 'native'; they are not treated as native scans.")
    empty = aug.isna() | aug_txt.str.strip().eq("")
    if empty.any():
        by_file = raw.loc[empty].groupby(["_split", "_source"]).size().to_dict()
        msg = (f"'{COL_AUG}': {int(empty.sum()):,} rows have no label ({int((empty & is_orig).sum()):,} of them "
               f"with is_original True); by split/file: {by_file}.")
        (errors if (empty & is_orig).any() else warnings).append(msg)

    # is_original must be a boolean
    valid_orig = raw[COL_ORIGINAL].isin([True, False])
    could_be_native = aug_txt.str.strip().str.lower().eq("native") | empty
    bad_orig = ~valid_orig & could_be_native
    if bad_orig.any():
        errors.append(f"'{COL_ORIGINAL}': {int(bad_orig.sum()):,} native-like rows have a non-boolean value "
                      f"{_examples(raw.loc[bad_orig, COL_ORIGINAL].unique())}; they are not treated as original scans.")
    if (~valid_orig & ~could_be_native).any():
        warnings.append(f"'{COL_ORIGINAL}': {int((~valid_orig & ~could_be_native).sum()):,} augmented rows have a "
                        f"non-boolean value {_examples(raw.loc[~valid_orig & ~could_be_native, COL_ORIGINAL].unique())}.")

    # every scan seen in the files must have a native original row
    for split in raw["_split"].unique():
        r = raw[raw["_split"] == split]
        lost = sorted(set(scan_key(r)) - set(scan_key(native[native["_split"] == split])))
        if lost:
            seen = r[scan_key(r).isin(lost)].groupby([COL_AUG, COL_ORIGINAL], dropna=False).size().to_dict()
            errors.append(f"{split}: {len(lost):,} scans have no native original row and are dropped: "
                          f"{_examples(lost)}; their rows (augmentation, is_original): {seen}.")

    # healthy must be 0 or 1 on native scans
    bad_h = ~native[COL_HEALTHY].isin([0, 1])
    if bad_h.any():
        errors.append(f"'{COL_HEALTHY}': {int(bad_h.sum()):,} native scans are neither 0 nor 1 "
                      f"{_examples(native.loc[bad_h, COL_HEALTHY].unique())}; they are in no population.")

    # external validation labels must be exactly camcan / oasis3
    ext = native[native["_split"] == SPLIT_EXTERNAL]
    other = ext.loc[~ext[COL_DATABASE].isin(list(EXTERNAL_DATABASES)), COL_DATABASE]
    if len(other):
        errors.append(f"'{COL_DATABASE}': {len(other):,} external validation scans are labelled "
                      f"{_examples(other.value_counts(dropna=False).to_dict().items())} instead of "
                      f"{list(EXTERNAL_DATABASES)}; they would be in no cohort.")

    # labels that differ only by case, spaces or punctuation
    labels = raw[COL_DATABASE].dropna().drop_duplicates()
    clash = labels.groupby(_norm_label(labels)).agg(list)
    clash = clash[clash.map(len) > 1]
    if len(clash):
        warnings.append(f"'{COL_DATABASE}': labels that differ only by case/spaces/punctuation (counted as different "
                        f"databases): {_examples(clash.tolist())}.")
    ids = raw[[COL_DATABASE, COL_SUBJECT]].dropna().drop_duplicates()
    id_norm = ids[COL_DATABASE].astype(str) + "/" + ids[COL_SUBJECT].astype(str).str.strip().str.lower()
    dup_ids = ids[id_norm.duplicated(keep=False)]
    if len(dup_ids):
        warnings.append(f"'{COL_SUBJECT}': {len(dup_ids):,} IDs differ only by case/spaces within a database (counted "
                        f"as different participants): {_examples((dup_ids[COL_DATABASE] + '/' + dup_ids[COL_SUBJECT]).tolist())}.")
    for col in (COL_DATABASE, COL_SUBJECT, COL_SESSION):
        v = raw[col].dropna().astype(str)
        padded = v[v != v.str.strip()].unique()
        if len(padded):
            warnings.append(f"'{col}': values with leading/trailing spaces: {_examples(padded)}.")

    # sex values outside female/male (kept, but dropped by the Figure 13 OLS)
    sex = native[COL_SEX].astype(str).str.lower().str.strip()
    odd = native.loc[~sex.isin(["female", "male"]) & (native["_split"] != SPLIT_TRAIN)]
    if len(odd):
        warnings.append(f"'{COL_SEX}': {len(odd):,} native scans not 'female'/'male' after trimming and lower-casing "
                        f"{_examples(odd[COL_SEX].value_counts(dropna=False).to_dict().items())}: kept here (sex "
                        "'other / missing'), dropped by the complete-case filter of the Figure 13 OLS models.")

    # healthy flag vs file of origin
    hf_not = native[(native["_source"] == "healthy_file") & (native[COL_HEALTHY] != 1)]
    pf_h = native[(native["_source"] == "prediction_file") & (native[COL_HEALTHY] == 1)]
    if len(hf_not):
        warnings.append(f"{len(hf_not):,} native scans of the healthy file have healthy != 1 "
                        f"(databases {hf_not[COL_DATABASE].value_counts().to_dict()}).")
    if len(pf_h):
        warnings.append(f"{len(pf_h):,} native scans of the prediction (clinical) file have healthy == 1 and no "
                        f"identical row in the healthy file (databases {pf_h[COL_DATABASE].value_counts().to_dict()}).")

    # diagnosis text: typographic quotes, non-breaking spaces or mojibake do not match the grouping rules
    for col in (COL_NEURO_DX, COL_PSYCH_DX):
        if col in raw.columns:
            v = raw[col].dropna().astype(str)
            sus = v[v.str.contains(SUSPECT_TEXT)].unique()
            if len(sus):
                warnings.append(f"'{col}': {len(sus):,} values contain typographic quotes, non-breaking spaces or "
                                f"mis-decoded characters (they will not match the grouping rules): {_examples(sus)}.")
    return errors, warnings


def scan_accounting(raw: pd.DataFrame, native: pd.DataFrame) -> pd.DataFrame:
    """Scans and participants at each step, per split, so that every scan is accounted for."""
    rows = []
    for split in raw["_split"].unique():
        r, n = raw[raw["_split"] == split], native[native["_split"] == split]
        h = n[n[COL_HEALTHY] == 1]
        hv = h[pd.to_numeric(h[COL_PRED], errors="coerce").notna() & pd.to_numeric(h[COL_AGE], errors="coerce").notna()]
        rows.append({
            "split": split,
            "rows in files": len(r),
            "scans in files": scan_key(r).nunique(),
            "participants in files": subject_key(r).nunique(),
            "native original scans": scan_key(n).nunique(),
            "healthy == 1": scan_key(h).nunique(),
            "healthy == 0": scan_key(n[n[COL_HEALTHY] == 0]).nunique(),
            "healthy, with prediction and age": len(hv),
            "healthy participants, with prediction and age": subject_key(hv).nunique(),
        })
    return pd.DataFrame(rows)


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
