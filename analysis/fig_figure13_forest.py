#!/usr/bin/env python3
"""Figure 13: associations between diagnostic groups and predicted brain age (forest plot).

Script version of the Figure 13 notebook (multilabel_diagnostic_classification_v2), reduced to
what the figure needs and moved to the thesis diagnosis classification (diagnosis_terms.csv).

Population, for each split: native original scans of the healthy and prediction files, read as
the notebook did (``augmentation == "native"`` and ``is_original``, files concatenated, identical
rows removed: analysis_io.native_scans).

* Test (``test``): used for hyperparameter selection.
* External validation (``validation``: Cam-CAN + OASIS-3): final validation, main result.

Model, fitted separately in each split (ordinary least squares):

  predicted age ~ age_c + sex_male + G_1 + ... + G_k            one indicator per modelled group
                  + G_1:age_c + ... + G_k:age_c                  group x age
                  + G_i:G_j                                      groups co-occurring in >= 10 scans

* age_c: chronological age floored to whole years, centred on the mean of the split's model
  sample, so that each group coefficient is the difference in predicted age from the reference at
  that mean age (the notebook used uncentred age: differences at age 0).
* sex_male: 1 male, 0 female. A scan without sex takes the sex of the same participant's other
  scans when they agree (as demographics_tables.R); scans still without sex are not modelled.
* Groups: thesis classification (diagnosis_classification.py). Groups with fewer than 30 scans in
  the split are merged into Other_neuro_disease / Other_psych_disease (R10, counted on all native
  scans of the split, as the demographics tables); a group is modelled if it has at least 30
  scans in the model sample. A scan can be in several groups.
* Reference: healthy scans (healthy == 1, R11). Patients without a modelled group
  (Unclassified_patient, or only groups that are not modelled) are counted and not modelled.
* Standard errors: heteroskedasticity-consistent HC3 (``--se cluster``: clustered by participant).
  95% CIs from the normal distribution; p < 0.05 uncorrected in the figure, Benjamini-Hochberg
  q-values over the group coefficients of a split in the tables.

The coefficient of a group is the difference for scans of that group without the other group of
any interaction term it belongs to (the interaction adds to both for scans with the two).

Outputs (in ``--out-dir``, default results/analysis/figure13):
  fig13_forest.pdf / .png           (a, b) neurological, (c, d) psychiatric groups; test left,
                                    external validation right; 16 cm wide, PNG at 300 dpi
  figure13_main_effects.csv / .md   plotted coefficients (HC3 SE, 95% CI, p, q, n)
  figure13_all_terms.csv            every model term
  figure13_models.csv / .md         per split: N, centring age, R2, adjusted R2, R2 of age + sex,
                                    formula
  figure13_groups.csv               per split and group: scans, R10 merge, scans modelled
  figure13_legend.md                figure legend and methods sentence (English)
  checks.md                         scan accounting and data checks

Example (from the repository root):
  python analysis/fig_figure13_forest.py --pred-dir data/predictions
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

import analysis_io as aio  # noqa: E402
import diagnosis_classification as dc  # noqa: E402

DEFAULT_OUT_SUBDIR = "figure13"
MIN_COOCCURRENCE = 10
ALPHA = 0.05
SPLIT_ORDER = (aio.SPLIT_TEST, aio.SPLIT_EXTERNAL)          # left, right column (as the notebook)
SPLIT_LABELS = {aio.SPLIT_TEST: "Test", aio.SPLIT_EXTERNAL: "External validation"}
SPLIT_TITLES = {aio.SPLIT_TEST: "Test set", aio.SPLIT_EXTERNAL: "External validation"}
DOMAINS = (dc.NEURO, dc.PSYCH)
DOMAIN_TITLES = {dc.NEURO: "Neurological disorders", dc.PSYCH: "Psychiatric disorders"}
DOMAIN_COLORS = {dc.NEURO: "#0072B2", dc.PSYCH: "#7A5195"}    # notebook colours

# Names shown in the figure (the tables keep the group names of diagnosis_terms.csv)
GROUP_LABELS = {
    "AD": "Alzheimer's disease",
    "MCI": "Mild cognitive impairment",
    "SCC": "Subjective cognitive concern",
    "Impaired_not_MCI": "Impaired, not MCI",
    "Parkinson": "Parkinson's disease",
    "Prodromal_PD": "Prodromal Parkinson's disease",
    "Mutation_PD": "PD gene mutation carriers",
    "Vascular": "Vascular brain injury",
    "TBI": "Traumatic brain injury",
    "FTD": "Frontotemporal dementia",
    "Atypical_parkinsonism": "Atypical parkinsonism",
    "NPH": "Normal pressure hydrocephalus",
    "Movement_disorder": "Movement disorder",
    "Other_neuro_disease": "Other neurological disease",
    "Dementia_unspecified": "Dementia, cause unspecified",
    "Other_neuro_symptom": "Other neurological symptom",
    "Depression": "Depression",
    "Bipolar": "Bipolar disorder",
    "Anxiety": "Anxiety disorders",
    "PTSD": "Post-traumatic stress disorder",
    "OCD": "Obsessive-compulsive disorder",
    "Schizophrenia": "Schizophrenia",
    "Addiction": "Substance use disorders",
    "Developmental": "Developmental disorders",
    "Pain": "Pain-related conditions",
    "Sleep": "Sleep disorders",
    "Other_psych_disease": "Other psychiatric disease",
    "Other_psych_symptom": "Other psychiatric symptom",
}

# Fonts follow the notebook (Arial/Helvetica/DejaVu Sans, bold axis labels); sizes are set for
# printing at the true 16 cm width (same width as fig_brainage_scatter.py).
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
FS = {"tick": 6.5, "ylabel": 6.8, "label": 7.5, "title": 8.5, "value": 5.5, "legend": 7, "letter": 10}
FIG_WIDTH_CM = 16.0
ROW_CM = 0.62                     # height per diagnosis row
COL_HEADER_CM = 0.9               # split titles with N and R2
EXTRA_CM = 2.4                    # legend, x axes and x label

TERM_AGE, TERM_SEX = "age_c", "sex_male"


class ModelError(ValueError):
    """The design matrix cannot be estimated (collinear terms, infinite standard errors)."""


# =============================================================================
# Data
# =============================================================================
def label(group: str) -> str:
    return GROUP_LABELS.get(group, group.replace("_", " "))


def fill_sex(native: pd.DataFrame, notes: List[str]) -> pd.Series:
    """female / male (any case, spaces), else the agreeing sex of the participant's other scans."""
    sex = native[aio.COL_SEX].astype("string").str.strip().str.lower()
    sex = sex.where(sex.isin(["female", "male"]))
    subject = aio.subject_key(native)
    known = sex.groupby(subject).agg(lambda s: "/".join(sorted(set(s.dropna()))))
    known_scan = subject.map(known)
    filled = sex.isna() & known_scan.isin(["female", "male"])
    if filled.any():
        notes.append(f"'sex': {int(filled.sum()):,} native scans had no sex and were given the sex of the same "
                     "participant's other scans.")
    mixed = known[known == "female/male"]
    if len(mixed):
        notes.append(f"WARNING: 'sex': {len(mixed):,} participants are female on some scans and male on others "
                     f"(kept as written): {aio._examples(mixed.index)}.")
    return sex.where(~filled, known_scan)


def prepare(native: pd.DataFrame, notes: List[str]) -> pd.DataFrame:
    """Native scans with numeric age/prediction, floored age, sex code and thesis groups."""
    df = native.copy()
    for col in (aio.COL_PRED, aio.COL_AGE):          # non-numbers become missing (named by aio.audit_labels)
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["_age"] = np.floor(df[aio.COL_AGE])
    df["_sex"] = fill_sex(df, notes)
    df["_subject"] = aio.subject_key(df)
    df["_scan"] = aio.scan_key(df)
    groups, unmapped = [], {}
    for neuro, psych in zip(df[aio.COL_NEURO_DX], df[aio.COL_PSYCH_DX]):
        groups.append(dc.scan_groups(neuro, psych))
        for dx in (neuro, psych):
            for term in dc.classify_cell(dx)[2]:
                unmapped[term] = unmapped.get(term, 0) + 1
    df["_groups"] = groups
    if unmapped:
        notes.append(f"WARNING: {len(unmapped):,} diagnosis terms are not in diagnosis_terms.csv and set no group "
                     f"(scans: {aio._examples(sorted(unmapped.items(), key=lambda kv: -kv[1]))}).")
    return df


def split_sample(d: pd.DataFrame, min_scans: int) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], List[str]]:
    """Model sample of one split.

    Returns (model sample with merged groups, per-group table, scan accounting, modelled groups).
    """
    split = d["_split"].iloc[0]
    merge = dc.rare_groups(dc.group_counts(d["_groups"]), min_scans)          # R10, all native scans
    d = d.assign(_merged=[dc.merge_rare(g, merge) for g in d["_groups"]])
    acc = {"split": split, "native scans": len(d)}

    bad_h = ~d[aio.COL_HEALTHY].isin([0, 1])
    no_num = ~bad_h & d[[aio.COL_PRED, aio.COL_AGE]].isna().any(axis=1)
    no_sex = ~bad_h & ~no_num & d["_sex"].isna()
    acc["excluded: healthy not 0/1"] = int(bad_h.sum())
    acc["excluded: no numeric prediction or age"] = int(no_num.sum())
    acc["excluded: no sex"] = int(no_sex.sum())
    cc = d[~(bad_h | no_num | no_sex)].copy()
    healthy = cc[aio.COL_HEALTHY] == 1
    # healthy scans are the reference: a group written for one of them is not used (reported in checks)
    cc.loc[healthy, "_merged"] = pd.Series([frozenset()] * int(healthy.sum()), index=cc.index[healthy], dtype=object)

    counts = dc.group_counts(cc.loc[~healthy, "_merged"])
    modelled = [g for g in dc.all_groups() if counts.get(g, 0) >= min_scans]
    has_model = cc["_merged"].map(lambda s: bool(s & set(modelled)))
    unclassified = ~healthy & cc["_merged"].map(len).eq(0)
    no_modelled = ~healthy & ~unclassified & ~has_model
    acc[f"excluded: patient without group ({dc.UNCLASSIFIED})"] = int(unclassified.sum())
    acc["excluded: patient whose groups are not modelled"] = int(no_modelled.sum())
    sample = cc[healthy | has_model].copy()
    acc["modelled: reference (healthy == 1)"] = int((sample[aio.COL_HEALTHY] == 1).sum())
    acc["modelled: patients"] = int((sample[aio.COL_HEALTHY] == 0).sum())
    acc["modelled: scans"] = len(sample)
    acc["modelled: participants"] = sample["_subject"].nunique()
    acc["modelled: participants with >= 2 scans"] = int((sample["_subject"].value_counts() >= 2).sum())
    if sum(v for k, v in acc.items() if k.startswith("excluded")) + acc["modelled: scans"] != acc["native scans"]:
        raise RuntimeError(f"{split}: scan accounting does not add up: {acc}")

    raw_counts = dc.group_counts(d["_groups"])
    sample_counts = dc.group_counts(sample["_merged"])
    rows = []
    for g in dc.all_groups():
        if raw_counts.get(g, 0) == 0 and counts.get(g, 0) == 0:
            continue
        rows.append({"split": split, "group": g, "domain": dc.group_domain(g),
                     "native scans": raw_counts.get(g, 0),
                     f"merged into (< {min_scans} scans)": merge.get(g, ""),
                     "scans after merge, complete cases": counts.get(g, 0),
                     "scans in model": sample_counts.get(g, 0) if g in modelled else 0,
                     "modelled": g in modelled})
    return sample, pd.DataFrame(rows), acc, modelled


# =============================================================================
# Model
# =============================================================================
def cooccurring_pairs(sample: pd.DataFrame, modelled: List[str], min_cooccurrence: int) -> List[Tuple[str, str]]:
    """Pairs of modelled groups present together in >= min_cooccurrence scans (notebook rule)."""
    member = {g: sample["_merged"].map(lambda s, g=g: g in s).to_numpy() for g in modelled}
    return [(a, b) for a, b in itertools.combinations(modelled, 2)
            if int((member[a] & member[b]).sum()) >= min_cooccurrence]


def independent_terms(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Columns kept in order, leaving out each column that is a linear combination of the ones
    kept before it (e.g. two groups with the same scans, or a pair interaction equal to a group)."""
    kept: List[str] = []
    dropped: List[str] = []
    rank = 0
    for col in X.columns:
        r = np.linalg.matrix_rank(X[kept + [col]].to_numpy())
        if r > rank:
            kept.append(col)
            rank = r
        else:
            dropped.append(col)
    return kept, dropped


def design(sample: pd.DataFrame, modelled: List[str], pairs: List[Tuple[str, str]]) -> Tuple[pd.DataFrame, float]:
    """Design matrix (intercept, centred age, sex, groups, group x age, group x group) and centre."""
    centre = float(sample["_age"].mean())
    X = pd.DataFrame({"Intercept": 1.0, TERM_AGE: sample["_age"] - centre,
                      TERM_SEX: (sample["_sex"] == "male").astype(float)}, index=sample.index)
    for g in modelled:
        X[g] = sample["_merged"].map(lambda s, g=g: g in s).astype(float)
    for g in modelled:
        X[f"{g}:{TERM_AGE}"] = X[g] * X[TERM_AGE]
    for a, b in pairs:
        X[f"{a}:{b}"] = X[a] * X[b]
    return X, centre


def formula(columns) -> str:
    return "predicted_age ~ " + " + ".join(c for c in columns if c != "Intercept")


def fit(sample: pd.DataFrame, X: pd.DataFrame, se: str):
    y = sample[aio.COL_PRED].astype(float)
    if se == "cluster":
        codes = pd.factorize(sample["_subject"])[0]
        res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": codes})
    else:
        res = sm.OLS(y, X).fit(cov_type="HC3")
    if not np.all(np.isfinite(res.bse)):
        raise ModelError(f"non-finite standard errors in {sample['_split'].iloc[0]} "
                         f"({list(res.bse.index[~np.isfinite(res.bse)])}): a term is carried by a single scan.")
    base = sm.OLS(y, X[["Intercept", TERM_AGE, TERM_SEX]]).fit()
    return res, base


def fit_split(sample: pd.DataFrame, modelled: List[str], min_cooccurrence: int, se: str,
              notes: List[str]) -> Tuple[pd.DataFrame, dict, List[str]]:
    split = sample["_split"].iloc[0]
    pairs = cooccurring_pairs(sample, modelled, min_cooccurrence)
    X, centre = design(sample, modelled, pairs)
    kept, dropped = independent_terms(X)
    if dropped:
        notes.append(f"WARNING: {split}: {len(dropped)} term(s) left out because they are linear combinations of "
                     f"the terms before them (collinear): {', '.join(dropped)}.")
    if any(t in ("Intercept", TERM_AGE, TERM_SEX) for t in dropped):
        raise ModelError(f"{split}: age or sex is collinear with the other terms ({dropped})")
    X = X[kept]
    res, base = fit(sample, X, se)
    ci = res.conf_int(alpha=ALPHA)
    terms = pd.DataFrame({"split": split, "term": res.params.index, "coef": res.params.values,
                          "se": res.bse.values, "z": res.tvalues.values, "p": res.pvalues.values,
                          "ci_low": ci[0].values, "ci_high": ci[1].values})
    terms["kind"] = np.select(
        [terms["term"].isin(modelled), terms["term"].str.endswith(f":{TERM_AGE}"), terms["term"].str.contains(":")],
        ["group", "group x age", "group x group"], default="covariate")
    model = {
        "split": split,
        "N scans": int(res.nobs),
        "N participants": sample["_subject"].nunique(),
        "N reference (healthy)": int((sample[aio.COL_HEALTHY] == 1).sum()),
        "N patients": int((sample[aio.COL_HEALTHY] == 0).sum()),
        "age centre (years)": round(centre, 2),
        "R2": res.rsquared,
        "adjusted R2": res.rsquared_adj,
        "R2 age + sex": base.rsquared,
        "delta R2 (diagnosis terms)": res.rsquared - base.rsquared,
        "terms (excl. intercept)": X.shape[1] - 1,
        "groups modelled": int(sum(g in kept for g in modelled)),
        "group x group interactions": int(sum(f"{a}:{b}" in kept for a, b in pairs)),
        "terms left out (collinear)": "; ".join(dropped),
        "standard errors": "HC3" if se == "hc3" else "clustered by participant",
        "formula": formula(X.columns),
    }
    return terms, model, dropped


def main_effects(terms: pd.DataFrame, sample: pd.DataFrame, modelled: List[str]) -> pd.DataFrame:
    m = terms[terms["kind"] == "group"].copy()
    m["q_fdr_bh"] = multipletests(m["p"], method="fdr_bh")[1] if len(m) else []
    m = m.rename(columns={"term": "group"})
    m["domain"] = m["group"].map(dc.group_domain)
    m["label"] = m["group"].map(label)
    m["n scans"] = m["group"].map(lambda g: int(sample["_merged"].map(lambda s: g in s).sum()))
    m["n participants"] = m["group"].map(
        lambda g: sample.loc[sample["_merged"].map(lambda s: g in s), "_subject"].nunique())
    m["significant"] = m["p"] < ALPHA
    cols = ["split", "domain", "group", "label", "n scans", "n participants", "coef", "se", "ci_low",
            "ci_high", "z", "p", "q_fdr_bh", "significant"]
    return m[cols]


# =============================================================================
# Figure
# =============================================================================
def domain_order(me: pd.DataFrame, domain: str) -> List[str]:
    """Test-set order (coefficient, then p), then groups modelled only in external validation."""
    d = me[me["domain"] == domain]
    test = d[d["split"] == aio.SPLIT_TEST].sort_values(["coef", "p"], ascending=[False, True])["group"].tolist()
    ext = d[(d["split"] == aio.SPLIT_EXTERNAL) & ~d["group"].isin(test)]
    return test + ext.sort_values(["coef", "p"], ascending=[False, True])["group"].tolist()


def _style(ax):
    ax.tick_params(axis="both", labelsize=FS["tick"], width=0.7, length=2.5, color="#4a4a4a")
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.7)
        ax.spines[side].set_color("#777777")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.5)
    ax.set_axisbelow(True)


def _panel(ax, me: pd.DataFrame, domain: str, split: str, order: List[str], with_names: bool):
    color = DOMAIN_COLORS[domain]
    rows = me[(me["domain"] == domain) & (me["split"] == split)].set_index("group")
    ax.axvline(0, color="#333333", ls="--", lw=0.8, zorder=1)
    ticks = []
    for y, g in enumerate(order):
        n = f"n = {int(rows.loc[g, 'n scans']):,}" if g in rows.index else "–"
        ticks.append(f"{label(g)} ({n})" if with_names else n)
        if g not in rows.index:
            continue
        r = rows.loc[g]
        ax.errorbar(r["coef"], y, xerr=[[r["coef"] - r["ci_low"]], [r["ci_high"] - r["coef"]]], fmt="o",
                    markersize=4.2, markerfacecolor=color if r["significant"] else "white",
                    markeredgecolor=color, markeredgewidth=1.0, ecolor=color, elinewidth=1.0, capsize=1.8,
                    zorder=3)
        txt = f"{r['coef'] + 0.0:.2f}".replace("-", "−") if round(r["coef"], 2) != 0 else "0.00"
        ax.annotate(txt, xy=(r["coef"], y), xytext=(0, 3.2), textcoords="offset points", ha="center",
                    va="bottom", fontsize=FS["value"], color=color, zorder=4,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.3})
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(ticks, fontsize=FS["ylabel"])
    ax.set_ylim(len(order) - 0.45, -0.75)
    _style(ax)


def make_figure(me: pd.DataFrame, models: Dict[str, dict], out_dir: Path) -> Tuple[Path, Path]:
    orders = {dom: domain_order(me, dom) for dom in DOMAINS}
    domains = [dom for dom in DOMAINS if orders[dom]]
    if not domains:
        raise ModelError("no group is modelled in any split")
    # rows of the grid: column headers, then the panels of each domain
    ratios = [COL_HEADER_CM / ROW_CM] + [len(orders[dom]) + 0.3 for dom in domains]
    cm = 1 / 2.54
    height = (ROW_CM * sum(ratios) + EXTRA_CM) * cm
    fig = plt.figure(figsize=(FIG_WIDTH_CM * cm, height), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.02, hspace=0.0)
    gs = fig.add_gridspec(len(ratios), 2, height_ratios=ratios)

    for j, split in enumerate(SPLIT_ORDER):                   # column headers, centred on the panels
        ax = fig.add_subplot(gs[0, j])
        ax.axis("off")
        m = models.get(split)
        sub = f"N = {m['N scans']:,} scans, R² = {m['R2']:.3f}" if m else "not fitted"
        ax.text(0.5, 0.05, SPLIT_TITLES[split], transform=ax.transAxes, ha="center", va="bottom",
                fontsize=FS["title"], fontweight="bold")
        ax.text(0.5, -0.02, sub, transform=ax.transAxes, ha="center", va="top", fontsize=FS["tick"])

    letters = iter("abcd")
    for i, dom in enumerate(domains):
        row = me[me["domain"] == dom]
        lo, hi = row["ci_low"].min(), row["ci_high"].max()
        pad = max(0.5, 0.08 * (hi - lo))
        lo, hi = min(lo, 0) - pad, max(hi, 0) + pad
        for j, split in enumerate(SPLIT_ORDER):
            ax = fig.add_subplot(gs[1 + i, j])
            _panel(ax, me, dom, split, orders[dom], with_names=(j == 0))
            ax.set_xlim(lo, hi)
            ax.annotate(next(letters), xy=(0, 1), xycoords="axes fraction", xytext=(-4, 2),
                        textcoords="offset points", fontsize=FS["letter"], fontweight="bold",
                        ha="right", va="bottom")
            if j == 0:                       # domain name on the line of the panel letter
                ax.annotate(DOMAIN_TITLES[dom], xy=(0.005, 1), xycoords=("figure fraction", "axes fraction"),
                            xytext=(0, 3), textcoords="offset points", ha="left", va="bottom",
                            fontsize=FS["label"], fontweight="bold", color=DOMAIN_COLORS[dom], in_layout=False)
    fig.supxlabel("Difference in predicted age from healthy participants, years (95% CI)",
                  fontsize=FS["label"], fontweight="bold")
    handles = [Line2D([0], [0], marker="o", lw=0, color="#333333", markerfacecolor="#333333", markersize=4.2,
                      label="p < 0.05"),
               Line2D([0], [0], marker="o", lw=0, color="#333333", markerfacecolor="white", markeredgewidth=1.0,
                      markersize=4.2, label="p ≥ 0.05"),
               Line2D([0], [0], lw=0, label="– not modelled in this split")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False, fontsize=FS["legend"],
               handlelength=1.0, handletextpad=0.3, columnspacing=1.8)
    # No title (the legend is in the thesis); no bbox_inches="tight" so the width stays 16 cm.
    pdf, png = out_dir / "fig13_forest.pdf", out_dir / "fig13_forest.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png


# =============================================================================
# Text
# =============================================================================
def legend_text(models: Dict[str, dict], min_scans: int, min_cooccurrence: int, se: str) -> str:
    t, e = models.get(aio.SPLIT_TEST), models.get(aio.SPLIT_EXTERNAL)

    def desc(m, name):
        if not m:
            return f"{name}: not fitted"
        return (f"{name}: N = {m['N scans']:,} scans ({m['N participants']:,} participants; "
                f"{m['N reference (healthy)']:,} healthy, {m['N patients']:,} patients), age centred at "
                f"{m['age centre (years)']:.1f} years, R² = {m['R2']:.3f} (adjusted {m['adjusted R2']:.3f}; "
                f"age and sex only {m['R2 age + sex']:.3f})")

    se_txt = ("heteroskedasticity-consistent (HC3) standard errors" if se == "hc3"
              else "standard errors clustered by participant")
    return f"""# Figure 13: legend and methods (generated)

## Legend

**Figure 13. Associations between diagnostic groups and predicted brain age.** Ordinary least squares
regression of predicted brain age, fitted separately in the test set (left; hyperparameter selection) and
the external validation set (right; Cam-CAN and OASIS-3), on chronological age (floored to whole years and
centred on the mean age of the model sample), sex, one indicator per diagnostic group, group × age
interactions and interactions between groups that co-occur in at least {min_cooccurrence} scans. The
reference is the healthy participants. Each point is the coefficient of a group: the difference in
predicted brain age from healthy participants of the same sex at the mean age, for scans without a
co-occurring group of an interaction term. Bars, 95% confidence intervals from {se_txt}; filled points,
p < 0.05 (uncorrected). n, scans of the group in the model; –, group not modelled in that split. (a, b)
Neurological and (c, d) psychiatric groups. Groups follow the diagnostic classification of the Methods; groups with fewer than
{min_scans} scans in a split are merged into other neurological or psychiatric disease, and patients
without a modelled group are not modelled.

## Models

- {desc(t, "Test")}.
- {desc(e, "External validation")}.

## Methods sentence

Associations between diagnostic groups and predicted brain age were estimated by ordinary least squares in
each split, with predicted brain age as the outcome and chronological age (floored, centred), sex, one
indicator per diagnostic group (diagnostic classification of the Methods; groups with fewer than {min_scans} scans merged into
other neurological or psychiatric disease; healthy participants as reference), group × age interactions and
interactions between groups co-occurring in at least {min_cooccurrence} scans as predictors; inference used
{se_txt}.
"""


# =============================================================================
# Main
# =============================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pred-dir", help=f"folder with the prediction TSVs (default: <repo>/{aio.DEFAULT_PRED_DIR})")
    p.add_argument("--out-dir", help=f"output folder (default: <repo>/{aio.DEFAULT_OUT_DIR}/{DEFAULT_OUT_SUBDIR})")
    p.add_argument("--model-tag", default=aio.DEFAULT_MODEL_TAG,
                   help="model part of the TSV file names (default: %(default)s)")
    p.add_argument("--min-scans", type=int, default=dc.MIN_SCANS,
                   help="R10 merge threshold and minimum scans of a modelled group (default: %(default)s)")
    p.add_argument("--min-cooccurrence", type=int, default=MIN_COOCCURRENCE,
                   help="minimum scans with two groups for their interaction term (default: %(default)s)")
    p.add_argument("--se", choices=("hc3", "cluster"), default="hc3",
                   help="hc3: heteroskedasticity-consistent; cluster: clustered by participant (default: hc3)")
    p.add_argument("--no-strict", action="store_true",
                   help="report labelling/encoding errors in checks.md instead of stopping")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    pred_dir = aio.resolve_dir(args.pred_dir, aio.DEFAULT_PRED_DIR)
    out = Path(args.out_dir) if args.out_dir else aio.REPO_ROOT / aio.DEFAULT_OUT_DIR / DEFAULT_OUT_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    notes: List[str] = []
    try:
        raw = aio.read_all(pred_dir, args.model_tag, notes)
        native = aio.native_scans(raw, notes)
        errors, warnings = aio.audit_labels(raw, native)
        df = prepare(native, notes)
        grouped_healthy = df[(df[aio.COL_HEALTHY] == 1) & df["_groups"].map(len).gt(0)]
        if len(grouped_healthy):
            errors.append(f"{len(grouped_healthy):,} healthy (healthy == 1) scans have a diagnosis group "
                          f"({aio._examples(grouped_healthy['_scan'])}); they are kept in the reference without it.")
        if errors and not args.no_strict:
            raise aio.DataError("labelling/encoding problems would drop or misgroup scans (rerun with --no-strict "
                                "to only report them):\n  - " + "\n  - ".join(errors))
        notes += [f"ERROR (reported only, --no-strict): {e}" for e in errors] + [f"WARNING: {w}" for w in warnings]

        accounting, group_tables, all_terms, mains, models = [], [], [], [], {}
        for split in SPLIT_ORDER:
            d = df[df["_split"] == split]
            if d.empty:
                raise ValueError(f"{split}: no native scan")
            sample, groups, acc, modelled = split_sample(d, args.min_scans)
            accounting.append(acc)
            group_tables.append(groups)
            if not modelled:
                notes.append(f"WARNING: {split}: no group has {args.min_scans} scans; no model fitted.")
                continue
            terms, model, dropped = fit_split(sample, modelled, args.min_cooccurrence, args.se, notes)
            lost = groups["group"].isin(dropped)
            groups.loc[lost, "modelled"] = False
            groups.loc[lost, "note"] = "left out: same scans as groups before it (collinear)"
            all_terms.append(terms)
            mains.append(main_effects(terms, sample, modelled))
            models[split] = model
        if not models:
            raise ModelError("no model could be fitted")
        me = pd.concat(mains, ignore_index=True)
        pdf, png = make_figure(me, models, out)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    me.to_csv(out / "figure13_main_effects.csv", index=False)
    pd.concat(all_terms, ignore_index=True).to_csv(out / "figure13_all_terms.csv", index=False)
    pd.concat(group_tables, ignore_index=True).to_csv(out / "figure13_groups.csv", index=False)
    mod = pd.DataFrame(list(models.values()))
    mod.to_csv(out / "figure13_models.csv", index=False)

    show = me.assign(split=me["split"].map(SPLIT_LABELS), coef=me["coef"].round(2), se=me["se"].round(2),
                     **{"95% CI": me["ci_low"].round(2).astype(str) + " to " + me["ci_high"].round(2).astype(str)},
                     p=me["p"].map(lambda v: f"{v:.3g}"), q_fdr_bh=me["q_fdr_bh"].map(lambda v: f"{v:.3g}"))
    se_name = "HC3" if args.se == "hc3" else "clustered by participant"
    (out / "figure13_main_effects.md").write_text(
        "# Figure 13: group coefficients\n\nDifference in predicted brain age from healthy participants (years) at "
        f"the split's mean age, adjusted for sex; SE {se_name}; 95% CI normal; q: Benjamini-Hochberg over the "
        "groups of a split.\n\n"
        + aio.markdown_table(show[["split", "domain", "group", "n scans", "n participants", "coef", "se", "95% CI",
                                   "p", "q_fdr_bh"]]) + "\n")
    mshow = mod.assign(split=mod["split"].map(SPLIT_LABELS))
    for c in ("R2", "adjusted R2", "R2 age + sex", "delta R2 (diagnosis terms)"):
        mshow[c] = mshow[c].map(lambda v: f"{v:.4f}")
    (out / "figure13_models.md").write_text(
        "# Figure 13: models\n\n" + aio.markdown_table(mshow.drop(columns="formula")) + "\n\n## Formulas\n\n"
        + "\n".join(f"- {SPLIT_LABELS[m['split']]}: `{m['formula']}`" for m in models.values()) + "\n")
    (out / "figure13_legend.md").write_text(legend_text(models, args.min_scans, args.min_cooccurrence, args.se))
    acc = pd.DataFrame(accounting).set_index("split").T.reset_index().rename(columns={"index": "scans"})
    (out / "checks.md").write_text(
        "# Figure 13: data checks\n\n## Scan accounting\n\nEvery native scan of a split is in exactly one row "
        "(excluded or modelled).\n\n" + aio.markdown_table(acc) + "\n\n## Notes\n\n"
        + "\n".join(f"- {n}" for n in notes) + "\n")

    print(f"Figure : {pdf}\n         {png}")
    for name in ("figure13_models.md", "figure13_main_effects.md", "figure13_groups.csv", "figure13_all_terms.csv",
                 "figure13_legend.md", "checks.md"):
        print(f"Output : {out / name}")
    print("\n" + mshow.drop(columns="formula").to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
