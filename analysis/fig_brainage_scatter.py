#!/usr/bin/env python3
"""Predicted vs chronological brain age in healthy participants (native scans).

Population, for each split: native original scans (``augmentation == "native"`` and
``is_original``, read as in the Figure 13 forest plot), ``healthy == 1``, chronological age
floored to whole years (the precision of the training targets).

* Test (``test``): used for hyperparameter selection.
* External validation (``validation``: Cam-CAN + OASIS-3): final validation, main result.

Outputs (in ``--out-dir``):
  fig_brainage_scatter.pdf / .png   (a) test, (b) external by database, (c) age gap vs age;
                                    16 cm wide, no title, PNG at 300 dpi
  brainage_stats.csv / .md          scan-level statistics with 95% bootstrap CIs
  brainage_mae_by_subgroup.csv/.md  MAE by age band, sex and augmentation (with 95% CIs)
  brainage_text_snippets.md         results sentences and figure legend (English)
  scans_abs_gap_gt15.csv            scans with |gap| > 15 years (listed, not excluded)
  checks.md                         data checks

Confidence intervals: percentile bootstrap resampling participants (not scans), fixed seed.

Example (from the repository root):
  python analysis/fig_brainage_scatter.py --pred-dir data/predictions --out-dir results/analysis
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import analysis_io as aio  # noqa: E402

# Groups: key -> (display label, selector). External validation first (main result).
GROUP_ORDER = ["External", "camcan", "oasis3", "Test"]
TRAIN = "Training"                    # optional (--train-file), shown last
ALL_GROUPS = GROUP_ORDER + [TRAIN]
GROUP_LABELS = {"External": "External validation", "camcan": "Cam-CAN",
                "oasis3": "OASIS-3", "Test": "Test", TRAIN: "Training"}

GAP_FLAG = 15.0
DEFAULT_AGE_BINS = (40, 60, 75)       # bands: <40, 40-<60, 60-<75, >=75 (lower bound included)
N_BOOT = 1000
SEED = 2026

# Okabe-Ito subset, checked for colour-vision-deficiency separation.
COLORS = {"Test": "#0072B2", "camcan": "#D55E00", "oasis3": "#009E73",
          "Test_line": "#00466E", "External_line": "#1A1A1A", "identity": "#555555"}

# Fonts follow the forest-plot notebook (Arial/Helvetica/DejaVu Sans, bold axis labels);
# sizes are set for printing at the true 16 cm width.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
FS = {"label": 9, "tick": 8, "inset": 6.8, "legend": 7, "letter": 11}
FIG_WIDTH_CM, FIG_HEIGHT_CM = 16.0, 15.5


# =============================================================================
# Data
# =============================================================================
def load_population(pred_dir: Path, model_tag: str, train_file, notes: List[str]):
    """Return (all rows, native scans, native healthy scans, all healthy rows incl. augmentations)."""
    raw = aio.read_all(pred_dir, model_tag, notes)
    if train_file is not None:
        raw = pd.concat([raw, aio.read_training(train_file, notes)], ignore_index=True)
    native = aio.native_scans(raw, notes)

    pop = native[native[aio.COL_HEALTHY] == 1].copy()
    for col in (aio.COL_PRED, aio.COL_AGE):      # non-numbers become missing (named by aio.audit_labels)
        pop[col] = pd.to_numeric(pop[col], errors="coerce")
    n0 = len(pop)
    pop = pop.dropna(subset=[aio.COL_PRED, aio.COL_AGE])
    if len(pop) < n0:
        notes.append(f"WARNING: {n0 - len(pop):,} healthy native scans without a numeric prediction or age "
                     "are not analysed.")
    pop["_age"] = np.floor(pop[aio.COL_AGE])
    pop["_subject"] = aio.subject_key(pop)
    pop["_scan"] = aio.scan_key(pop)

    # All rows (native + augmented) of healthy participants, for the MAE by augmentation.
    # Duplicates across the two files are removed with the forest-plot rule (per-file columns
    # ignored), keeping the augmentation columns so different augmentations stay distinct.
    allrows = raw[raw[aio.COL_HEALTHY] == 1]
    cols = [c for c in allrows.columns if not c.startswith("_")
            and (c not in aio.DROP_COLS or c in (aio.COL_AUG, aio.COL_ORIGINAL))]
    allrows = allrows.drop_duplicates(subset=["_split", *cols], keep="first").copy()
    for col in (aio.COL_PRED, aio.COL_AGE):
        allrows[col] = pd.to_numeric(allrows[col], errors="coerce")
    allrows = allrows.dropna(subset=[aio.COL_PRED, aio.COL_AGE])
    allrows["_age"] = np.floor(allrows[aio.COL_AGE])
    allrows["_subject"] = aio.subject_key(allrows)
    return raw, native, pop, allrows


def group_masks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    ext = df["_split"] == aio.SPLIT_EXTERNAL
    db = df[aio.COL_DATABASE].astype(str)
    return {"External": ext, "camcan": ext & (db == "camcan"), "oasis3": ext & (db == "oasis3"),
            "Test": df["_split"] == aio.SPLIT_TEST, TRAIN: df["_split"] == aio.SPLIT_TRAIN}


def data_checks(native: pd.DataFrame, pop: pd.DataFrame, notes: List[str]) -> None:
    ext_db = native.loc[native["_split"] == aio.SPLIT_EXTERNAL, aio.COL_DATABASE].astype(str)
    notes.append(f"External validation databases: {ext_db.value_counts().to_dict()}")
    missing = sorted(set(aio.EXTERNAL_DATABASES) - set(ext_db))
    if missing:
        raise ValueError(f"External validation split has no rows for database(s) {missing}.")
    test_db = native.loc[native["_split"] == aio.SPLIT_TEST, aio.COL_DATABASE].astype(str)
    leak = test_db.isin(list(aio.EXTERNAL_DATABASES)).sum()
    if leak:
        notes.append(f"ANOMALY: {leak:,} test scans belong to camcan/oasis3.")
    notes.append(f"Test databases: {test_db.nunique()}")

    subj = {sp: set(aio.subject_key(native[native["_split"] == sp]))
            for sp in (aio.SPLIT_TRAIN, aio.SPLIT_TEST, aio.SPLIT_EXTERNAL)}
    for a, b in ((aio.SPLIT_TEST, aio.SPLIT_EXTERNAL), (aio.SPLIT_TRAIN, aio.SPLIT_TEST),
                 (aio.SPLIT_TRAIN, aio.SPLIT_EXTERNAL)):
        if subj[a] & subj[b]:
            notes.append(f"ANOMALY: {len(subj[a] & subj[b]):,} participants in both {a} and {b}.")
    train = native[native["_split"] == aio.SPLIT_TRAIN]
    if len(train):
        tdb = train[aio.COL_DATABASE].astype(str)
        notes.append(f"Training databases: {tdb.nunique()}; 'healthy' values: "
                     f"{train[aio.COL_HEALTHY].value_counts(dropna=False).to_dict()}")
        leak_train = tdb.isin(list(aio.EXTERNAL_DATABASES)).sum()
        if leak_train:
            notes.append(f"ANOMALY: {leak_train:,} training scans belong to camcan/oasis3.")

    dup = pop[pop.duplicated(["_split", "_scan"], keep=False)]
    if len(dup):
        notes.append(f"ANOMALY: {dup['_scan'].nunique():,} subject/session pairs have several native "
                     f"healthy scans ({len(dup):,} rows); all are kept.")
    notes.append(f"'{aio.COL_HEALTHY}' values (native scans): "
                 f"{native[aio.COL_HEALTHY].value_counts(dropna=False).to_dict()}")
    frac_int = float(np.mean(np.isclose(pop[aio.COL_AGE] % 1, 0)))
    notes.append(f"'{aio.COL_AGE}' whole numbers: {frac_int:.1%} (age is floored in all analyses).")


def check_accounting(acc: pd.DataFrame, pop: pd.DataFrame, stats: dict) -> List[str]:
    """Every healthy native scan with a prediction is in exactly one group; returns problems."""
    problems = []
    masks = group_masks(pop)
    per_split = acc.set_index("split")["healthy, with prediction and age"]
    for split, gkey in ((aio.SPLIT_TEST, "Test"), (aio.SPLIT_EXTERNAL, "External"), (aio.SPLIT_TRAIN, TRAIN)):
        if split in per_split.index and gkey in stats and stats[gkey]["N scans"] != per_split[split]:
            problems.append(f"{GROUP_LABELS[gkey]}: {stats[gkey]['N scans']:,} scans analysed but "
                            f"{per_split[split]:,} healthy native scans with prediction and age.")
    if int(masks["camcan"].sum() + masks["oasis3"].sum()) != int(masks["External"].sum()):
        problems.append("External validation scans are not all in Cam-CAN or OASIS-3.")
    return problems


# =============================================================================
# Statistics
# =============================================================================
def _participant_sums(x, y, keys):
    """Per-participant sufficient statistics; x and y are shifted by the same constant."""
    shift = float(np.mean(x))
    xs, ys = x - shift, y - shift
    codes, _ = pd.factorize(keys)
    cols = np.column_stack([np.ones_like(xs), xs, ys, xs * xs, ys * ys, xs * ys, np.abs(ys - xs)])
    s = np.zeros((codes.max() + 1, cols.shape[1]))
    np.add.at(s, codes, cols)
    return s


def _metrics_from_sums(t):
    n, sx, sy, sxx, syy, sxy, sabs = np.moveaxis(np.asarray(t, float), -1, 0)
    vxx = sxx - sx * sx / n
    vyy = syy - sy * sy / n
    cxy = sxy - sx * sy / n
    ss_res = syy - 2 * sxy + sxx
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"MAE": sabs / n, "ME": (sy - sx) / n, "r": cxy / np.sqrt(vxx * vyy),
                "R2": 1 - ss_res / vxx, "gap_slope": cxy / vxx - 1}


def _bootstrap(x, y, keys, n_boot, seed):
    s = _participant_sums(x, y, keys)
    p = s.shape[0]
    rng = np.random.default_rng(seed)
    t = np.vstack([rng.multinomial(p, np.full(p, 1.0 / p), size=min(100, n_boot - i)) @ s
                   for i in range(0, n_boot, 100)])
    boot = _metrics_from_sums(t)
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in boot.items()}


def group_stats(g: pd.DataFrame, age_bins, n_boot: int, seed: int) -> dict:
    x, y = g["_age"].to_numpy(float), g[aio.COL_PRED].to_numpy(float)
    e = y - x
    s = {
        "N scans": len(g), "N participants": g["_subject"].nunique(),
        "Age mean": x.mean(), "Age SD": x.std(ddof=1), "Age min": x.min(), "Age max": x.max(),
        "MAE": np.mean(np.abs(e)), "Median AE": np.median(np.abs(e)),
        "RMSE": np.sqrt(np.mean(e ** 2)), "ME": e.mean(), "r": np.corrcoef(x, y)[0, 1],
    }
    s["r2"] = s["r"] ** 2
    s["R2"] = 1 - np.sum(e ** 2) / np.sum((x - x.mean()) ** 2)
    s["slope"], s["intercept"] = np.polyfit(x, y, 1)
    s["gap_r"] = np.corrcoef(x, e)[0, 1]
    s["gap_slope"], _ = np.polyfit(x, e, 1)
    for lab, m in age_band_masks(x, age_bins).items():
        s[f"MAE age {lab}"] = np.mean(np.abs(e[m])) if m.any() else np.nan
        s[f"n age {lab}"] = int(m.sum())
    s["N |gap|>15"] = int(np.sum(np.abs(e) > GAP_FLAG))
    s["CI"] = _bootstrap(x, y, g["_subject"].to_numpy(), n_boot, seed)
    return s


def age_band_labels(age_bins) -> List[str]:
    edges = list(age_bins)
    labels = [f"<{edges[0]}"]
    labels += [f"{a}–{b}" for a, b in zip(edges[:-1], edges[1:])]
    labels += [f"≥{edges[-1]}"]
    return labels


def age_band_masks(x: np.ndarray, age_bins) -> Dict[str, np.ndarray]:
    edges = [-np.inf, *age_bins, np.inf]
    labels = age_band_labels(age_bins)
    return {lab: (x >= lo) & (x < hi) for lab, lo, hi in zip(labels, edges[:-1], edges[1:])}


def mae_with_ci(ae: np.ndarray, keys: np.ndarray, n_boot: int, seed: int):
    """MAE and participant-bootstrap 95% CI."""
    if len(ae) == 0:
        return np.nan, np.nan, np.nan
    codes, uniq = pd.factorize(keys)
    s = np.zeros((len(uniq), 2))
    np.add.at(s, codes, np.column_stack([np.ones_like(ae), ae]))
    p = len(uniq)
    rng = np.random.default_rng(seed)
    t = np.vstack([rng.multinomial(p, np.full(p, 1.0 / p), size=min(100, n_boot - i)) @ s
                   for i in range(0, n_boot, 100)])
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = t[:, 1] / t[:, 0]
    return float(ae.mean()), float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))


def subgroup_table(pop: pd.DataFrame, allrows: pd.DataFrame, age_bins, n_boot: int, seed: int) -> pd.DataFrame:
    """MAE by age band, sex and augmentation, for each group."""
    rows = []

    def add(group, factor, level, frame):
        ae = (frame[aio.COL_PRED] - frame["_age"]).abs().to_numpy(float)
        mae, lo, hi = mae_with_ci(ae, frame["_subject"].to_numpy(), n_boot, seed)
        rows.append({"group": GROUP_LABELS[group], "factor": factor, "level": level,
                     "N rows": len(frame), "N participants": frame["_subject"].nunique(),
                     "MAE": mae, "ci95_low": lo, "ci95_high": hi})

    for gkey, m in group_masks(pop).items():
        if not m.any():
            continue
        g = pop[m]
        for lab, mm in age_band_masks(g["_age"].to_numpy(float), age_bins).items():
            add(gkey, "age (years)", lab, g[mm])
        sex = g[aio.COL_SEX].astype(str).str.lower().str.strip()
        for lev in ("female", "male"):
            add(gkey, "sex", lev, g[sex == lev])
        other = g[~sex.isin(["female", "male"])]
        if len(other):
            add(gkey, "sex", "other / missing", other)
    for gkey, m in group_masks(allrows).items():
        if not m.any():
            continue
        a = allrows[m]
        aug = a[aio.COL_AUG].astype("string").fillna("")
        native_rows = (aug == "native") & (a[aio.COL_ORIGINAL] == True)  # noqa: E712 (figure population)
        augmented = ~native_rows & (aug.str.strip() != "")
        add(gkey, "augmentation", "native", a[native_rows])
        listed = pd.Series(False, index=a.index)
        for t in aio.AUGMENTATION_TRANSFORMS:          # a row is counted once for each transform it contains
            has_t = augmented & aug.str.contains(t, case=False, regex=False)
            listed |= has_t
            add(gkey, "augmentation", f"contains {t}", a[has_t])
        if (augmented & ~listed).any():
            add(gkey, "augmentation", "augmented, none of the listed transforms", a[augmented & ~listed])
        if (aug.str.strip() == "").any():
            add(gkey, "augmentation", "no augmentation label", a[aug.str.strip() == ""])
        add(gkey, "augmentation", "all augmented rows", a[augmented])
        add(gkey, "augmentation", "all rows (native + augmented)", a)
    return pd.DataFrame(rows)


# =============================================================================
# Figure
# =============================================================================
def _minus(text: str) -> str:
    """Typographic minus for negative numbers (keeps 'Cam-CAN', 'OASIS-3')."""
    return re.sub(r"(?<![\w])-(?=\d)", "−", text)


def _inset(ax, text, loc="upper left"):
    x, y, ha, va = {"upper left": (0.03, 0.97, "left", "top"),
                    "lower left": (0.02, 0.04, "left", "bottom")}[loc]
    ax.text(x, y, _minus(text), transform=ax.transAxes, ha=ha, va=va, fontsize=FS["inset"],
            linespacing=1.35, zorder=6,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "linewidth": 0.5, "alpha": 0.85,
                  "boxstyle": "round,pad=0.35"})


def _style(ax):
    ax.tick_params(axis="both", labelsize=FS["tick"], width=0.8, length=3, color="#4a4a4a")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("#4a4a4a")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)


def _letter(ax, letter):
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-26, 6),
                textcoords="offset points", fontsize=FS["letter"], fontweight="bold",
                ha="left", va="bottom")


def _fit_line(ax, x, y, color):
    b, a = np.polyfit(x, y, 1)
    xx = np.array([x.min(), x.max()])
    ax.plot(xx, a + b * xx, color=color, lw=1.4, solid_capstyle="butt", zorder=5)


def _scatter(ax, x, y, color):
    ax.scatter(x, y, s=3, alpha=0.3, linewidths=0, color=color, zorder=2)


def _inset_text(s):
    return (f"N = {s['N scans']:,} scans / {s['N participants']:,} participants\n"
            f"MAE = {s['MAE']:.2f} y\n"
            f"r = {s['r']:.3f}, R² = {s['R2']:.3f}\n"
            f"slope = {s['slope']:.3f}")


def _dot(color, label):
    return Line2D([0], [0], marker="o", lw=0, markersize=4, color=color, label=label)


def _line(color, label, ls="-"):
    return Line2D([0], [0], ls=ls, lw=1.4 if ls == "-" else 1.0, color=color, label=label)


def make_figure(pop: pd.DataFrame, stats: dict, out_dir: Path):
    m = group_masks(pop)
    test, ext = pop[m["Test"]], pop[m["External"]]
    cohorts = {k: pop[m[k]] for k in ("camcan", "oasis3")}
    vals = np.concatenate([test["_age"], test[aio.COL_PRED], ext["_age"], ext[aio.COL_PRED]])
    lo, hi = 5 * np.floor(np.nanmin(vals) / 5), 5 * np.ceil(np.nanmax(vals) / 5)

    cm = 1 / 2.54
    fig = plt.figure(figsize=(FIG_WIDTH_CM * cm, FIG_HEIGHT_CM * cm), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.72])
    ax_a, ax_b, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    # (a) test
    _scatter(ax_a, test["_age"], test[aio.COL_PRED], COLORS["Test"])
    ax_a.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color=COLORS["identity"], zorder=4)
    _fit_line(ax_a, test["_age"].to_numpy(), test[aio.COL_PRED].to_numpy(), COLORS["Test_line"])
    inset_a = _inset_text(stats["Test"])
    if TRAIN in stats:
        inset_a += f"\nTraining MAE = {stats[TRAIN]['MAE']:.2f} y"
    _inset(ax_a, inset_a)
    ax_a.legend(handles=[_line(COLORS["identity"], "y = x", "--"), _line(COLORS["Test_line"], "OLS fit")],
                loc="lower right", frameon=False, fontsize=FS["legend"], handlelength=1.8)

    # (b) external validation, coloured by database
    for k, g in cohorts.items():
        _scatter(ax_b, g["_age"], g[aio.COL_PRED], COLORS[k])
    ax_b.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color=COLORS["identity"], zorder=4)
    _fit_line(ax_b, ext["_age"].to_numpy(), ext[aio.COL_PRED].to_numpy(), COLORS["External_line"])
    _inset(ax_b, _inset_text(stats["External"]))
    ax_b.legend(handles=[_dot(COLORS[k], f"{GROUP_LABELS[k]} (n = {stats[k]['N scans']:,})") for k in cohorts]
                + [_line(COLORS["identity"], "y = x", "--"), _line(COLORS["External_line"], "OLS fit (pooled)")],
                loc="lower right", frameon=False, fontsize=FS["legend"], handlelength=1.8)

    for ax in (ax_a, ax_b):
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Chronological age (years)", fontsize=FS["label"])
        ax.set_ylabel("Predicted brain age (years)", fontsize=FS["label"])
        _style(ax)

    # (c) brain-age gap vs chronological age
    _scatter(ax_c, test["_age"], test[aio.COL_PRED] - test["_age"], COLORS["Test"])
    for k, g in cohorts.items():
        _scatter(ax_c, g["_age"], g[aio.COL_PRED] - g["_age"], COLORS[k])
    ax_c.axhline(0, ls="--", lw=1.0, color=COLORS["identity"], zorder=4)
    _fit_line(ax_c, test["_age"].to_numpy(), (test[aio.COL_PRED] - test["_age"]).to_numpy(), COLORS["Test_line"])
    _fit_line(ax_c, ext["_age"].to_numpy(), (ext[aio.COL_PRED] - ext["_age"]).to_numpy(), COLORS["External_line"])
    gaps = np.concatenate([test[aio.COL_PRED] - test["_age"], ext[aio.COL_PRED] - ext["_age"]])
    ylim = 5 * np.ceil(np.nanmax(np.abs(gaps)) * 1.05 / 5)
    ax_c.set_xlim(lo, hi)
    ax_c.set_ylim(-ylim, ylim)
    ax_c.set_xlabel("Chronological age (years)", fontsize=FS["label"])
    ax_c.set_ylabel("Brain-age gap (years)", fontsize=FS["label"])
    _style(ax_c)
    st, se = stats["Test"], stats["External"]
    _inset(ax_c,
           f"Test: N = {st['N scans']:,} / {st['N participants']:,}, MAE = {st['MAE']:.2f} y, "
           f"r = {st['gap_r']:.3f}, slope = {st['gap_slope']:.3f}\n"
           f"External: N = {se['N scans']:,} / {se['N participants']:,}, MAE = {se['MAE']:.2f} y, "
           f"r = {se['gap_r']:.3f}, slope = {se['gap_slope']:.3f}", loc="lower left")
    ax_c.legend(handles=[_dot(COLORS["Test"], "Test"), _dot(COLORS["camcan"], "Cam-CAN"),
                         _dot(COLORS["oasis3"], "OASIS-3"), _line(COLORS["identity"], "gap = 0", "--"),
                         _line(COLORS["Test_line"], "OLS fit, test"),
                         _line(COLORS["External_line"], "OLS fit, external")],
                loc="upper right", ncol=2, frameon=False, fontsize=FS["legend"], handlelength=1.8,
                columnspacing=1.0)

    for ax, letter in zip((ax_a, ax_b, ax_c), "abc"):
        _letter(ax, letter)

    # No title (the legend is in the thesis); no bbox_inches="tight" so the width stays 16 cm.
    pdf, png = out_dir / "fig_brainage_scatter.pdf", out_dir / "fig_brainage_scatter.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png


# =============================================================================
# Tables and text
# =============================================================================
def _fmt(v, d):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NA"
    return _minus(f"{int(round(v)):,}" if d == 0 else f"{v:.{d}f}")


def _range(lo, hi, d):
    sep = " to " if (lo < 0 or hi < 0) else "–"
    return f"{_fmt(lo, d)}{sep}{_fmt(hi, d)}"


def stat_rows(age_bins):
    rows = [  # (key, label, decimals, has_ci)
        ("N scans", "N scans", 0, False),
        ("N participants", "N participants", 0, False),
        ("Age mean", "Chronological age, mean (years)", 2, False),
        ("Age SD", "Chronological age, SD (years)", 2, False),
        ("Age min", "Chronological age, min (years)", 2, False),
        ("Age max", "Chronological age, max (years)", 2, False),
        ("MAE", "MAE (years)", 2, True),
        ("Median AE", "Median absolute error (years)", 2, False),
        ("RMSE", "RMSE (years)", 2, False),
        ("ME", "Mean signed error, predicted − chronological (years)", 2, True),
        ("r", "Pearson r", 3, True),
        ("r2", "r²", 3, False),
        ("R2", "R² = 1 − SSres/SStot", 3, True),
        ("slope", "Slope, predicted ~ chronological", 3, False),
        ("intercept", "Intercept, predicted ~ chronological (years)", 2, False),
        ("gap_r", "Pearson r, gap ~ chronological", 3, False),
        ("gap_slope", "Slope, gap ~ chronological", 3, True),
    ]
    rows += [(f"MAE age {lab}", f"MAE, age {lab} (years)", 2, False) for lab in age_band_labels(age_bins)]
    rows += [("N |gap|>15", "N scans with |gap| > 15 years", 0, False)]
    return rows


def stats_tables(stats, age_bins):
    long_rows, wide = [], {}
    for key, label, d, has_ci in stat_rows(age_bins):
        wide[label] = {}
        for gkey in [g for g in ALL_GROUPS if g in stats]:
            s = stats[gkey]
            lo, hi = s["CI"][key] if has_ci else (np.nan, np.nan)
            cell = _fmt(s[key], d) + (f" ({_range(lo, hi, d)})" if has_ci else "")
            n_band = s.get("n age " + key[len("MAE age "):]) if key.startswith("MAE age ") else None
            if n_band is not None:
                cell += f" [n = {n_band:,}]"
            wide[label][GROUP_LABELS[gkey]] = cell
            long_rows.append({"group": GROUP_LABELS[gkey], "statistic": label, "estimate": s[key],
                              "ci95_low": lo, "ci95_high": hi, "n_in_band": n_band, "formatted": cell})
    wide_df = pd.DataFrame(wide).T[[GROUP_LABELS[g] for g in ALL_GROUPS if g in stats]]
    return pd.DataFrame(long_rows), wide_df


def subgroup_markdown(sub: pd.DataFrame) -> str:
    cells = [
        f"{_fmt(m, 2)} ({_range(lo, hi, 2)}) [{n:,} / {k:,}]" if n else "—"
        for m, lo, hi, n, k in zip(sub["MAE"], sub["ci95_low"], sub["ci95_high"],
                                   sub["N rows"], sub["N participants"])
    ]
    sub = sub.assign(cell=cells)
    groups = [GROUP_LABELS[g] for g in ALL_GROUPS if GROUP_LABELS[g] in set(sub["group"])]
    blocks = []
    for factor in sub["factor"].drop_duplicates():
        f = sub[sub["factor"] == factor]
        levels = list(f["level"].drop_duplicates())
        wide = pd.DataFrame(index=levels, columns=groups, dtype=object)
        for lev, grp, cell in zip(f["level"], f["group"], f["cell"]):
            wide.loc[lev, grp] = cell
        blocks.append(f"### MAE by {factor}\n\nCells: MAE (95% CI) [N rows / N participants]\n\n"
                      + aio.markdown_table(wide.fillna("—"), index_label=factor))
    return "\n\n".join(blocks)


def _ci(s, key, d=2, unit=" years"):
    lo, hi = s["CI"][key]
    return f"{_fmt(s[key], d)}{unit} (95% CI {_range(lo, hi, d)})"


def _paren_slope(s):
    lo, hi = s["CI"]["gap_slope"]
    return f"slope {_fmt(s['gap_slope'], 3)}, 95% CI {_range(lo, hi, 3)}"


def _direction(s):
    lo, hi = s["CI"]["gap_slope"]
    return "negatively" if hi < 0 else ("positively" if lo > 0 else "not significantly")


def thesis_text(stats) -> str:
    E, C, O, T = (stats[g] for g in GROUP_ORDER)
    s1 = (f"On native scans of healthy participants, with chronological age expressed in whole years "
          f"(floored, as for the training targets), the model reached an MAE of {_ci(E, 'MAE')} in the "
          f"external validation set ({E['N scans']:,} scans from {E['N participants']:,} participants), "
          f"with r = {_ci(E, 'r', 3, '')} and R² = {_ci(E, 'R2', 3, '')}.")
    s2 = (f"The MAE was {_ci(C, 'MAE')} in Cam-CAN and {_ci(O, 'MAE')} in OASIS-3; in the test set used "
          f"for hyperparameter selection ({T['N scans']:,} scans from {T['N participants']:,} participants), "
          f"it was {_ci(T, 'MAE')}, with r = {_ci(T, 'r', 3, '')} and R² = {_ci(T, 'R2', 3, '')}.")
    s3 = (f"The mean signed error (predicted − chronological age) was {_ci(E, 'ME')} in the external "
          f"validation set and {_ci(T, 'ME')} in the test set.")
    dE, dT = _direction(E), _direction(T)
    if dE == dT:
        s4 = (f"The brain-age gap was {dE} associated with chronological age in the external validation "
              f"set ({_paren_slope(E)}) and in the test set ({_paren_slope(T)})")
    else:
        s4 = (f"The brain-age gap was {dE} associated with chronological age in the external validation "
              f"set ({_paren_slope(E)}) and {dT} associated with it in the test set ({_paren_slope(T)})")
    s4 += (", consistent with regression toward the mean; chronological age was therefore included as a "
           "covariate in the OLS models of Figure 13." if dE == dT == "negatively" else ".")
    s5 = ""
    if TRAIN in stats:
        Tr = stats[TRAIN]
        lo, hi = Tr["CI"]["MAE"]
        s5 = (f" For comparison, the MAE on the native training scans was {_fmt(Tr['MAE'], 2)} years "
              f"(95% CI {_range(lo, hi, 2)}; {Tr['N scans']:,} scans from {Tr['N participants']:,} participants), against "
              f"{_fmt(T['MAE'], 2)} years in the test set and {_fmt(E['MAE'], 2)} years in the external "
              "validation set.")
    legend = (
        "Figure X - Predicted brain age versus chronological age in healthy participants (native scans only; "
        "chronological age in whole years, floored). "
        f"(a) Test set, used for hyperparameter selection ({T['N scans']:,} scans, {T['N participants']:,} "
        "participants). "
        f"(b) External validation set ({E['N scans']:,} scans, {E['N participants']:,} participants), "
        f"colored by database (Cam-CAN, {C['N scans']:,} scans; OASIS-3, {O['N scans']:,} scans). "
        "In (a) and (b), each point is one scan; the dashed line is the identity line (y = x, perfect "
        "prediction) and the solid line is the ordinary least-squares (OLS) regression of predicted on "
        "chronological age (pooled across databases in b); both panels share the same axis limits. "
        "(c) Brain-age gap (predicted − chronological age) versus chronological age for the test set (blue) "
        "and the external validation set (database colors as in b); the dashed line marks a gap of zero and "
        "the solid lines are OLS fits for the test set and the pooled external validation set. "
        "Insets report the number of scans and participants, the mean absolute error (MAE), Pearson's r, "
        "R² = 1 − SSres/SStot computed against the identity line, and the OLS slope; in (c), r and slope "
        "refer to the regression of the gap on chronological age. "
        + ("The inset of (a) also gives the MAE on the native training scans of healthy participants, for "
           "comparison. " if TRAIN in stats else "")
        + "Patients are not shown (see Figure 13)."
    )
    return f"## Results text (IV.3.2)\n\n{s1} {s2} {s3} {s4}{s5}\n\n## Figure legend\n\n{legend}\n"


# =============================================================================
# Main
# =============================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pred-dir", help=f"folder with the prediction TSVs (default: <repo>/{aio.DEFAULT_PRED_DIR})")
    p.add_argument("--out-dir", help=f"output folder (default: <repo>/{aio.DEFAULT_OUT_DIR})")
    p.add_argument("--model-tag", default=aio.DEFAULT_MODEL_TAG,
                   help="model part of the TSV file names (default: %(default)s)")
    p.add_argument("--age-bins", type=int, nargs="+", default=list(DEFAULT_AGE_BINS),
                   help="age band edges in years, lower bound included (default: 40 60 75)")
    p.add_argument("--n-boot", type=int, default=N_BOOT,
                   help="bootstrap resamples of participants (default: %(default)s)")
    p.add_argument("--seed", type=int, default=SEED, help="bootstrap seed (default: %(default)s)")
    p.add_argument("--train-file",
                   help="prediction table of the training set (same columns), predicted by the final model; "
                        "default: <pred-dir>/predictions_train_<model-tag>.tsv if it exists")
    p.add_argument("--no-train", action="store_true", help="do not add the training set")
    p.add_argument("--no-strict", action="store_true",
                   help="report labelling/encoding errors in checks.md instead of stopping")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if sorted(set(args.age_bins)) != list(args.age_bins):
        print("error: --age-bins must be strictly increasing", file=sys.stderr)
        return 2
    pred_dir = aio.resolve_dir(args.pred_dir, aio.DEFAULT_PRED_DIR)
    out = aio.resolve_dir(args.out_dir, aio.DEFAULT_OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    notes: List[str] = []

    train_file = None
    if args.train_file:
        train_file = Path(args.train_file)
    elif not args.no_train:
        candidate = pred_dir / aio.TRAIN_FILE_TEMPLATE.format(tag=args.model_tag)
        if candidate.is_file():
            train_file = candidate
            notes.append(f"Training set added automatically from {candidate.name} (use --no-train to leave it out).")
    try:
        raw, native, pop, allrows = load_population(pred_dir, args.model_tag, train_file, notes)
        errors, warnings = aio.audit_labels(raw, native)
        if errors and not args.no_strict:
            raise aio.DataError("labelling/encoding problems would drop scans (rerun with --no-strict to only "
                                "report them):\n  - " + "\n  - ".join(errors))
        notes += [f"ERROR (reported only, --no-strict): {e}" for e in errors] + [f"WARNING: {w}" for w in warnings]
        data_checks(native, pop, notes)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    accounting = aio.scan_accounting(raw, native)

    masks = group_masks(pop)
    groups = GROUP_ORDER + ([TRAIN] if train_file is not None else [])
    stats = {}
    for gkey in groups:
        g = pop[masks[gkey]]
        if len(g) < 3:
            print(f"error: group {GROUP_LABELS[gkey]} has {len(g)} scans", file=sys.stderr)
            return 2
        stats[gkey] = group_stats(g, args.age_bins, args.n_boot, args.seed)

    problems = check_accounting(accounting, pop, stats)
    if problems and not args.no_strict:
        print("error: scans not accounted for:\n  - " + "\n  - ".join(problems), file=sys.stderr)
        return 2
    notes += [f"ERROR (reported only, --no-strict): {p}" for p in problems]
    accounting.to_csv(out / "scan_accounting.csv", index=False)

    pdf, png = make_figure(pop, stats, out)

    long_df, wide_df = stats_tables(stats, args.age_bins)
    long_df.to_csv(out / "brainage_stats.csv", index=False)
    header = ("# Brain-age statistics (healthy participants, native scans)\n\n"
              "Population per split: augmentation == native & is_original, healthy == 1; chronological age "
              "floored to whole years. Test = hyperparameter selection; External validation = Cam-CAN + OASIS-3 "
              "(main result)" + ("; Training = training set (--train-file), for comparison" if TRAIN in stats else "")
              + f". Scan-level statistics; 95% CIs: percentile bootstrap, {args.n_boot:,} resamples "
              f"of participants (seed {args.seed}). Age bands include their lower bound. "
              "Gap = predicted − chronological age.\n\n")
    (out / "brainage_stats.md").write_text(header + aio.markdown_table(wide_df, index_label="Statistic") + "\n")

    sub = subgroup_table(pop, allrows, args.age_bins, args.n_boot, args.seed)
    sub.to_csv(out / "brainage_mae_by_subgroup.csv", index=False)
    (out / "brainage_mae_by_subgroup.md").write_text(
        "# MAE by subgroup (healthy participants)\n\n"
        "Age and sex: native scans (same population as the figure). Augmentation: every row of healthy "
        "participants, native and augmented (each participant contributes several augmented rows). "
        f"95% CIs: participant bootstrap ({args.n_boot:,} resamples, seed {args.seed}).\n\n"
        + subgroup_markdown(sub) + "\n")

    (out / "brainage_text_snippets.md").write_text(thesis_text(stats))

    flagged = pop.assign(gap=pop[aio.COL_PRED] - pop["_age"])
    flagged = flagged.loc[flagged["gap"].abs() > GAP_FLAG,
                          ["_split", aio.COL_DATABASE, aio.COL_SUBJECT, aio.COL_SESSION, "_age", aio.COL_PRED, "gap"]]
    flagged.rename(columns={"_split": "split", "_age": "age_floored"}).to_csv(
        out / "scans_abs_gap_gt15.csv", index=False)
    for gkey in groups:
        notes.append(f"{GROUP_LABELS[gkey]}: {stats[gkey]['N |gap|>15']:,} scan(s) with |gap| > {GAP_FLAG:g} "
                     "years (kept; listed in scans_abs_gap_gt15.csv).")
    (out / "checks.md").write_text(
        "# Data checks\n\n## Scan accounting\n\n" + aio.markdown_table(accounting) + "\n\n"
        "Every scan in the files has a native original row, every native scan is healthy == 0 or 1, and every "
        "healthy native scan with a prediction is in exactly one group.\n\n## Notes\n\n"
        + "\n".join(f"- {n}" for n in notes) + "\n")

    print(f"Figure : {pdf}\n         {png}")
    for name in ("brainage_stats.md", "brainage_mae_by_subgroup.md", "brainage_text_snippets.md", "checks.md",
                 "scan_accounting.csv"):
        print(f"Output : {out / name}")
    print("\n" + wide_df.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
