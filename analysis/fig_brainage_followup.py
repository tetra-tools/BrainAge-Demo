#!/usr/bin/env python3
"""Longitudinal follow-up: predicted vs chronological age of participants scanned several times.

Population: external validation set (Cam-CAN, OASIS-3), native original scans of every
participant (healthy and patients) with at least two sessions. Each participant is put in one
group from the ``healthy`` value of their sessions, in age order:

* Stable healthy : healthy == 1 at every session
* Converter      : healthy == 1 at the first session and healthy == 0 at the last
* Patient        : healthy == 0 at the first session and at the last
* Other          : any other sequence (e.g. patient then healthy); counted, not plotted

Chronological age is floored to whole years (as everywhere in the thesis), so follow-up
intervals are in whole years and a participant whose sessions fall in the same year has no
rate of change (counted in the table).

Mixed models (all scans of the three groups, random intercept and slope of time per participant;
random intercept only if that does not converge):

* Rates: predicted age ~ t + group + group x t + baseline age x t + sex. The stable-healthy slope is
  the model's own pace of ageing (below 1 because predictions shrink towards the mean age), so the
  group x t terms are the excess rates of converters and patients. Fitting every scan with a
  random intercept does not anchor a course on its first (noisy) scan: no regression to the mean
  from the baseline.
* Healthy reference: the stable-healthy fixed effects give each scan the change (and level)
  expected for a stable healthy participant of the same baseline age and sex; the excess over it
  is plotted.
* Conversion: the date is only known to lie between the last healthy and the first patient
  session. Converters are aligned on their first patient session, and a broken-stick model
  (stable healthy + converters) adds a change of slope after a knot placed at that session or at
  the midpoint of the interval; the knot offset is profiled by maximum likelihood (-6 to +2 years).
  Every model is also fitted with converters restricted to a cognitive diagnosis at the first
  patient session (MCI, AD, dementia of unspecified cause, FTD; diagnostic classification of the Methods).

Outputs (in ``--out-dir``):
  fig_brainage_followup.pdf / .png  (a-c) one line per participant, by group: change in predicted age
                                    since the first scan vs years since the first scan, with the group
                                    mean curve (95% CI); (d) the three mean curves together
  brainage_followup_mean_curves.csv mean change in predicted age per year since the first scan
  brainage_followup_stats.csv/.md   per database and group: participants, sessions, follow-up,
                                    baseline age and gap, rate (median, IQR, mean with 95% CI)
  brainage_followup_participants.csv one row per participant (group, sessions, rate, conversion)
  fig_brainage_followup_adjusted.pdf / .png
                                    (a) change in predicted age minus the change expected for a stable
                                    healthy participant (mixed model), by group; (b) converters aligned on
                                    their first patient session, with the broken-stick fit; (c) profile of
                                    the knot offset
  brainage_followup_models.csv/.md  mixed-model coefficients (rates, broken stick) and knot profile
  brainage_followup_excess_curves.csv, brainage_followup_aligned_curves.csv
  brainage_followup_text.md         figure legends
  checks.md                         data checks

Example (from the repository root):
  python analysis/fig_brainage_followup.py --pred-dir data/predictions --out-dir results/analysis/followup
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import patheffects
from scipy import stats as st
import statsmodels.formula.api as smf

import analysis_io as aio
import diagnosis_classification as dc
import fig_brainage_scatter as base          # shared style (fonts, axes) and number formatting
import fig_figure13_forest as f13            # sex coding shared with the forest plot

plt = base.plt

GROUPS = ["Stable healthy", "Converter", "Patient"]
OTHER = "Other"
GROUP_COLORS = {"Stable healthy": "#0072B2", "Converter": "#E69F00", "Patient": "#CC79A7"}
N_BOOT = 1000
SEED = 2026
MIN_N = 10           # mean curve drawn at a time point only if at least MIN_N participants contribute
COGNITIVE_GROUPS = ("MCI", "AD", "Dementia_unspecified", "FTD")   # cognitive conversion
KNOT_OFFSETS = tuple(range(-6, 3))   # knot offsets profiled, years from the first patient session


# =============================================================================
# Data
# =============================================================================
def load_followup(pred_dir: Path, model_tag: str, split: str, strict: bool, notes: List[str]):
    raw = aio.read_all(pred_dir, model_tag, notes)
    native = aio.native_scans(raw, notes)
    errors, warnings = aio.audit_labels(raw, native)
    if errors and strict:
        raise aio.DataError("labelling/encoding problems would drop scans (rerun with --no-strict to only "
                            "report them):\n  - " + "\n  - ".join(errors))
    notes += [f"ERROR (reported only, --no-strict): {e}" for e in errors] + [f"WARNING: {w}" for w in warnings]

    d = native[native["_split"] == split].copy()
    for col in (aio.COL_PRED, aio.COL_AGE):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    n0 = len(d)
    d = d.dropna(subset=[aio.COL_PRED, aio.COL_AGE])
    d = d[d[aio.COL_HEALTHY].isin([0, 1])]
    if len(d) < n0:
        notes.append(f"WARNING: {n0 - len(d):,} scans without numeric prediction/age or healthy 0/1 not used.")
    d["_age"] = np.floor(d[aio.COL_AGE])
    d["_gap"] = d[aio.COL_PRED] - d["_age"]
    d["_sex"] = f13.fill_sex(d, notes)
    d["_groups"] = [dc.scan_groups(n, p) for n, p in zip(d[aio.COL_NEURO_DX], d[aio.COL_PSYCH_DX])]
    d["_subject"] = aio.subject_key(d)
    d["_healthy"] = d[aio.COL_HEALTHY].astype(int)
    # sessions in the same (floored) year are ordered by the number in session_id ("ses-2" before "ses-10")
    d["_ses_num"] = pd.to_numeric(d[aio.COL_SESSION].astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")
    d = d.sort_values(["_subject", "_age", "_ses_num", aio.COL_SESSION], kind="stable")
    first = d.groupby("_subject")[["_age", aio.COL_PRED]].transform("first")
    d["_t"] = d["_age"] - first["_age"]                      # years since the first scan (whole years)
    d["_dpred"] = d[aio.COL_PRED] - first[aio.COL_PRED]       # change in predicted age since the first scan
    return d


def classify(first: int, last: int) -> str:
    if first == 1 and last == 1:
        return "Stable healthy"
    if first == 1 and last == 0:
        return "Converter"
    if first == 0 and last == 0:
        return "Patient"
    return OTHER


def participant_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject, g in d.groupby("_subject", sort=False):
        status = g["_healthy"].to_numpy()
        group = classify(status[0], status[-1])
        if group == "Stable healthy" and (status == 0).any():
            group = OTHER                                   # healthy -> patient -> healthy
        if group == "Patient" and (status == 1).any():
            group = OTHER                                   # patient -> healthy -> patient
        ages, preds = g["_age"].to_numpy(float), g[aio.COL_PRED].to_numpy(float)
        rate = np.polyfit(ages, preds, 1)[0] if np.unique(ages).size > 1 else np.nan
        conv_age = t_dx = t_lh = np.nan
        conv_groups = ""
        if group == "Converter":
            i_dx = int(np.argmax(status == 0))
            conv_age, t_dx = ages[i_dx], ages[i_dx] - ages[0]
            t_lh = ages[i_dx - 1] - ages[0]                 # last healthy session before it
            conv_groups = ", ".join(sorted(g["_groups"].iloc[i_dx]))
        rows.append({
            "participant": subject, "database": g[aio.COL_DATABASE].iloc[0], "group": group,
            "sessions": g[aio.COL_SESSION].nunique(), "scans": len(g),
            "baseline age": ages[0], "last age": ages[-1], "follow-up (years)": ages[-1] - ages[0],
            "baseline gap": g["_gap"].iloc[0], "last gap": g["_gap"].iloc[-1],
            "rate (predicted years per year)": rate, "age at first patient session": conv_age,
            "first patient session (years since first scan)": t_dx,
            "last healthy session (years since first scan)": t_lh,
            "groups at first patient session": conv_groups,
            "cognitive conversion": bool(set(conv_groups.split(", ")) & set(COGNITIVE_GROUPS)),
            "sex": g["_sex"].dropna().iloc[0] if g["_sex"].notna().any() else np.nan,
            "status sequence": "".join("H" if s == 1 else "P" for s in status),
        })
    return pd.DataFrame(rows)


def _boot_mean(x: np.ndarray, n_boot: int, seed: int):
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summary_table(part: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    dbs = [("All external", part)] + [(aio.EXTERNAL_DATABASES.get(db, db), part[part["database"] == db])
                                      for db in sorted(part["database"].unique())]
    for db_label, p in dbs:
        for group in GROUPS + [OTHER]:
            g = p[p["group"] == group]
            if g.empty:
                continue
            rate = g["rate (predicted years per year)"].to_numpy(float)
            ok = rate[~np.isnan(rate)]
            lo, hi = _boot_mean(rate, n_boot, seed)
            rows.append({
                "database": db_label, "group": group, "participants": len(g), "scans": int(g["scans"].sum()),
                "sessions mean": g["sessions"].mean(), "sessions min": g["sessions"].min(),
                "sessions max": g["sessions"].max(),
                "follow-up mean (years)": g["follow-up (years)"].mean(),
                "follow-up min": g["follow-up (years)"].min(), "follow-up max": g["follow-up (years)"].max(),
                "baseline age mean": g["baseline age"].mean(), "baseline age SD": g["baseline age"].std(ddof=1),
                "baseline gap mean": g["baseline gap"].mean(), "baseline gap SD": g["baseline gap"].std(ddof=1),
                "participants with a rate": len(ok),
                "participants without rate (same year)": int(np.isnan(rate).sum()),
                "rate median": np.median(ok) if len(ok) else np.nan,
                "rate Q1": np.percentile(ok, 25) if len(ok) else np.nan,
                "rate Q3": np.percentile(ok, 75) if len(ok) else np.nan,
                "rate mean": ok.mean() if len(ok) else np.nan, "rate mean 95% CI low": lo,
                "rate mean 95% CI high": hi,
            })
    return pd.DataFrame(rows)


def mean_curves(d: pd.DataFrame, part: pd.DataFrame, min_n: int) -> pd.DataFrame:
    """Mean change in predicted age per whole year since the first scan, by group (95% t-interval).

    A participant with several sessions in the same year contributes their mean for that year.
    """
    grp = part.set_index("participant")["group"]
    per = (d.groupby(["_subject", "_t"], as_index=False)[["_dpred"]].mean()
           .assign(group=lambda x: x["_subject"].map(grp)))
    rows = []
    for (group, t), g in per.groupby(["group", "_t"]):
        v = g["_dpred"].to_numpy(float)
        n = len(v)
        half = st.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        rows.append({"group": group, "years since first scan": int(t), "participants": n,
                     "mean change in predicted age": v.mean(), "ci95_low": v.mean() - half,
                     "ci95_high": v.mean() + half, "mean change in brain-age gap": v.mean() - t,
                     "drawn (n >= min-n)": n >= min_n})
    return pd.DataFrame(rows)


def curves_by(df: pd.DataFrame, x: str, value: str, min_n: int) -> pd.DataFrame:
    """Mean of `value` per group and whole-year `x` (one value per participant and year), 95% t-interval."""
    per = df.groupby(["group", "_subject", x], as_index=False)[value].mean()
    rows = []
    for (group, xv), g in per.groupby(["group", x]):
        v = g[value].to_numpy(float)
        n = len(v)
        half = st.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        rows.append({"group": group, "x": int(xv), "participants": n, "mean": v.mean(),
                     "ci95_low": v.mean() - half, "ci95_high": v.mean() + half, "drawn": n >= min_n})
    return pd.DataFrame(rows, columns=["group", "x", "participants", "mean", "ci95_low", "ci95_high", "drawn"])


# =============================================================================
# Mixed models
# =============================================================================
TERM_LABELS = {
    "Intercept": "Stable healthy, first scan, mean baseline age, female (years)",
    "t": "Stable healthy: rate (predicted years per year, mean baseline age)",
    "conv": "Converter - stable healthy at the first scan (years)",
    "pat": "Patient - stable healthy at the first scan (years)",
    "conv:t": "Converter - stable healthy: rate difference",
    "pat:t": "Patient - stable healthy: rate difference",
    "age0c": "Per year of baseline age, at the first scan (years)",
    "age0c:t": "Rate change per year of baseline age",
    "sex_male": "Male - female (years)",
    "post": "Converter: change of rate after the knot",
    "conv:t + post": "Converter - stable healthy: rate difference after the knot",
}
KEY_TERMS = {"rates": ["t", "conv:t", "pat:t", "age0c:t"], "broken stick": ["t", "conv:t", "post", "conv:t + post"]}


def model_data(d: pd.DataFrame, part: pd.DataFrame, notes: List[str]) -> pd.DataFrame:
    """Scans of the plotted groups with the covariates of the mixed models (baseline age centred)."""
    info = part.set_index("participant")
    m = d[d["_subject"].map(info["group"]).isin(GROUPS)].copy()
    no_sex = m["_sex"].isna()
    if no_sex.any():
        notes.append(f"Mixed models: {int(no_sex.sum()):,} scans without sex not modelled "
                     f"({m.loc[no_sex, '_subject'].nunique():,} participants).")
        m = m[~no_sex].copy()
    for col, src in (("group", "group"), ("t_dx", "first patient session (years since first scan)"),
                     ("t_lh", "last healthy session (years since first scan)"), ("cognitive", "cognitive conversion")):
        m[col] = m["_subject"].map(info[src])
    base_age = m.groupby("_subject")["_age"].first()
    m.attrs["age centre"] = float(base_age.mean())
    m["age0c"] = m["_subject"].map(base_age) - m.attrs["age centre"]
    m["pred"], m["t"] = m[aio.COL_PRED].astype(float), m["_t"].astype(float)
    m["sex_male"] = (m["_sex"] == "male").astype(float)
    m["conv"] = (m["group"] == "Converter").astype(float)
    m["pat"] = (m["group"] == "Patient").astype(float)
    return m


STRUCTURES = (("~t", "intercept + slope"), ("1", "intercept"))


def fit_mixed(data: pd.DataFrame, formula: str, notes: List[str], label: str, reml: bool = True,
              structures=STRUCTURES):
    """Random intercept and slope of t per participant, else random intercept only. (result, structure)."""
    for re_formula, name in structures:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = smf.mixedlm(formula, data, groups=data["_subject"], re_formula=re_formula).fit(
                    reml=reml, method=["lbfgs", "powell"])
        except (ValueError, np.linalg.LinAlgError) as exc:
            notes.append(f"{label}: random {name} model failed ({exc}).")
            continue
        if res.converged and np.isfinite(res.bse_fe).all():
            return res, name
        notes.append(f"{label}: random {name} model did not converge.")
    return None, None


def subset(m: pd.DataFrame, definition: str) -> pd.DataFrame:
    """All converters, or converters with a cognitive diagnosis at the first patient session only."""
    return m if definition == "all converters" else m[(m["conv"] == 0) | (m["cognitive"] == True)]  # noqa: E712


def rate_formula(data: pd.DataFrame) -> str:
    terms = ["t", "age0c", "age0c:t", "sex_male"]
    for g in ("conv", "pat"):
        if data[g].any():
            terms += [g, f"{g}:t"]
    return "pred ~ " + " + ".join(terms)


def with_knot(data: pd.DataFrame, knot: pd.Series) -> pd.DataFrame:
    """Stable healthy + converters, `post` = years after the converter's knot (0 before it, 0 for healthy)."""
    data = data[data["group"].isin(["Stable healthy", "Converter"])].copy()
    data["post"] = np.where(data["conv"] == 1, np.clip(data["t"] - knot.loc[data.index], 0, None), 0.0)
    return data


def identifiable(data: pd.DataFrame) -> bool:
    """Converters must have time before and after the knot, else conv:t and post are confounded."""
    c = data[data["conv"] == 1]
    return bool(((c["post"] == 0) & (c["t"] > 0)).any() and (c["post"] > 0).any())


def both_sides(data: pd.DataFrame) -> int:
    """Converters with a scan after their first scan but before the knot and a scan after the knot."""
    c = data[data["conv"] == 1]
    before = c[(c["post"] == 0) & (c["t"] > 0)].groupby("_subject").size()
    after = c[c["post"] > 0].groupby("_subject").size()
    return len(before.index.intersection(after.index))


def knot_profile(data: pd.DataFrame, definition: str, notes: List[str], min_n: int) -> List[dict]:
    """Maximum-likelihood fits of the broken stick with the knot moved, all with ONE random-effects structure
    (likelihoods of different structures are not comparable). Offsets where fewer than `min_n` converters
    have scans on both sides of the knot are not fitted."""
    off_col = "knot offset (years from first patient session)"
    cases = []
    for offset in KNOT_OFFSETS:
        kd = with_knot(data, data["t_dx"] + offset)
        cases.append((offset, kd, both_sides(kd)))
    fitted = [(o, kd) for o, kd, n in cases if n >= min_n]
    lls, used = {}, None
    for structure in STRUCTURES:
        lls = {}
        for offset, kd in fitted:
            res, _ = fit_mixed(kd, BROKEN_STICK, [], "", reml=False, structures=(structure,))
            lls[offset] = res.llf if res is not None else np.nan
        if all(np.isfinite(v) for v in lls.values()):
            used = structure[1]
            break
        failed = [o for o, v in lls.items() if not np.isfinite(v)]
        notes.append(f"Knot profile ({definition}): random {structure[1]} model did not converge at offsets "
                     f"{', '.join(f'{o:+d}' for o in failed)}; whole profile refitted with the next structure.")
    if used is None:
        notes.append(f"Knot profile ({definition}): no random-effects structure converged at every offset.")
    return [{"converters": definition, off_col: o, "converters with scans on both sides": n,
             "log-likelihood (ML)": lls.get(o, np.nan) if used else np.nan, "random effects": used or ""}
            for o, kd, n in cases]


BROKEN_STICK = "pred ~ t + conv + conv:t + post + age0c + age0c:t + sex_male"


def coef_rows(res, model: str, definition: str, knot: str, data: pd.DataFrame, structure: str) -> List[dict]:
    fe, ci, cov = res.fe_params, res.conf_int(), res.cov_params()
    est = {k: (fe[k], ci.loc[k, 0], ci.loc[k, 1], res.pvalues[k]) for k in fe.index}
    if "post" in fe.index:                                    # rate difference after the knot
        e = fe["conv:t"] + fe["post"]
        se = np.sqrt(cov.loc["conv:t", "conv:t"] + cov.loc["post", "post"] + 2 * cov.loc["conv:t", "post"])
        est["conv:t + post"] = (e, e - 1.96 * se, e + 1.96 * se, 2 * st.norm.sf(abs(e / se)))
    n_part = data.groupby("group")["_subject"].nunique()
    return [{"model": model, "converters": definition, "knot": knot, "term": k, "label": TERM_LABELS.get(k, k),
             "estimate": v[0], "ci95_low": v[1], "ci95_high": v[2], "p": v[3],
             "participants": int(data["_subject"].nunique()), "scans": len(data),
             "converters (participants)": int(n_part.get("Converter", 0)),
             "random effects": structure, "baseline age centre": data.attrs.get("age centre", np.nan)}
            for k, v in est.items()]


def healthy_reference(m: pd.DataFrame, res) -> pd.DataFrame:
    """Change and level expected for a stable healthy participant of the same baseline age and sex."""
    fe = res.fe_params
    rate = fe["t"] + fe["age0c:t"] * m["age0c"]
    m = m.copy()
    m["_excess_change"] = m["_dpred"] - rate * m["t"]
    m["_resid_healthy"] = m["pred"] - (fe["Intercept"] + fe["age0c"] * m["age0c"]
                                       + fe["sex_male"] * m["sex_male"] + rate * m["t"])
    return m


def align_converters(m: pd.DataFrame) -> pd.DataFrame:
    """Converters on the time of their first patient session (tau = 0), residuals relative to that scan."""
    c = m[m["group"] == "Converter"].copy()
    c["tau"] = c["t"] - c["t_dx"]
    at_dx = c[c["_healthy"] == 0].groupby("_subject")["_resid_healthy"].first()   # first patient scan
    c["_dresid"] = c["_resid_healthy"] - c["_subject"].map(at_dx)
    return c


def run_models(m: pd.DataFrame, notes: List[str], min_n: int = MIN_N) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[object]]:
    """Rate models and broken-stick models for both converter definitions; knot profile."""
    rows, prof, ref = [], [], None
    for definition in ("all converters", "cognitive converters"):
        data = subset(m, definition)
        if definition != "all converters" and not data["conv"].any():
            notes.append("Mixed models: no converter with a cognitive diagnosis at the first patient session.")
            continue
        res, structure = fit_mixed(data, rate_formula(data), notes, f"Rates ({definition})")
        if res is not None:
            rows += coef_rows(res, "rates", definition, "", data, structure)
            ref = res if definition == "all converters" else ref
        conv = data[data["conv"] == 1]
        if conv.empty:
            continue
        knots = {"first patient session": data["t_dx"], "midpoint (last healthy, first patient)":
                 (data["t_lh"] + data["t_dx"]) / 2}
        for knot_name, knot in knots.items():
            kd = with_knot(data, knot)
            if not identifiable(kd):
                notes.append(f"Broken stick ({definition}, {knot_name}): no converter time on both sides of the knot.")
                continue
            res_k, structure_k = fit_mixed(kd, BROKEN_STICK, notes, f"Broken stick ({definition}, {knot_name})")
            if res_k is not None:
                rows += coef_rows(res_k, "broken stick", definition, knot_name, kd, structure_k)
        prof += knot_profile(data, definition, notes, min_n)
    prof = pd.DataFrame(prof)
    if len(prof):
        best = prof.groupby("converters")["log-likelihood (ML)"].transform("max")
        prof["likelihood ratio vs best"] = 2 * (best - prof["log-likelihood (ML)"])
        prof["within 95% (LR <= 3.84)"] = prof["likelihood ratio vs best"] <= st.chi2.ppf(0.95, 1)
    return pd.DataFrame(rows), prof, ref


# =============================================================================
# Figure
# =============================================================================
def _mean_line(ax, curve: pd.DataFrame, color: str, lw: float = 2.0, label=None):
    c = curve[curve["drawn (n >= min-n)"]].sort_values("years since first scan")
    if c.empty:
        return
    t = c["years since first scan"].to_numpy(float)
    ax.fill_between(t, c["ci95_low"], c["ci95_high"], color=color, alpha=0.22, linewidth=0, zorder=3)
    ax.plot(t, c["mean change in predicted age"], color=color, lw=lw, zorder=4, label=label,
            path_effects=[patheffects.withStroke(linewidth=lw + 1.6, foreground="white")])


def make_figure(d: pd.DataFrame, part: pd.DataFrame, curves: pd.DataFrame, out_dir: Path):
    plotted = part[part["group"].isin(GROUPS)]
    traj = (d[d["_subject"].isin(plotted["participant"])]
            .merge(plotted[["participant", "group"]], left_on="_subject", right_on="participant")
            .sort_values(["_subject", "_t", "_ses_num"], kind="stable"))          # each line in time order
    t_max = max(1.0, float(traj["_t"].max())) if len(traj) else 1.0
    drawn = curves[curves["drawn (n >= min-n)"] & curves["group"].isin(GROUPS)]
    y = np.concatenate([traj["_dpred"].to_numpy(float), [0.0, t_max],      # mean-curve CIs stay inside panels a-c
                        drawn[["ci95_low", "ci95_high"]].to_numpy(float).ravel()])
    y = y[np.isfinite(y)]
    y_lo, y_hi = np.floor(y.min() / 5) * 5, np.ceil(y.max() / 5) * 5

    cm = 1 / 2.54
    fig = plt.figure(figsize=(16 * cm, 13.5 * cm), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.9])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_d = fig.add_subplot(gs[1, :])
    identity = dict(ls="--", lw=0.9, color=base.COLORS["identity"], zorder=1)

    for ax, group, letter in zip(axes, GROUPS, "abc"):
        color = GROUP_COLORS[group]
        for _, t in traj[traj["group"] == group].groupby("_subject", sort=False):
            ax.plot(t["_t"], t["_dpred"], color=color, lw=0.6, alpha=0.3, zorder=2, solid_capstyle="round")
        ax.plot([0, t_max], [0, t_max], **identity)
        ax.axhline(0, ls=":", lw=0.7, color="#888888", zorder=1)
        _mean_line(ax, curves[curves["group"] == group], color)
        ax.set_xlim(0, t_max)
        ax.set_ylim(y_lo, y_hi)
        n = int((plotted["group"] == group).sum())
        ax.set_title(group, fontsize=base.FS["label"] - 1, fontweight="bold", pad=4)
        base._inset(ax, f"n = {n:,} participants")
        ax.set_xlabel("Years since first scan", fontsize=base.FS["label"] - 1)
        if letter == "a":
            ax.set_ylabel("Change in predicted\nbrain age (years)", fontsize=base.FS["label"] - 1)
        else:
            ax.tick_params(labelleft=False)
        base._style(ax)
        base._letter(ax, letter, dy=16)

    # (d) mean curves together
    ax_d.plot([0, t_max], [0, t_max], **identity, label="y = x")
    ax_d.axhline(0, ls=":", lw=0.7, color="#888888", zorder=1)
    for group in GROUPS:
        c = curves[curves["group"] == group]
        n = int((plotted["group"] == group).sum())
        _mean_line(ax_d, c, GROUP_COLORS[group], lw=2.2, label=f"{group} (n = {n:,})")
    ax_d.set_xlim(0, t_max)
    ax_d.set_xlabel("Years since first scan", fontsize=base.FS["label"] - 1)
    ax_d.set_ylabel("Mean change in predicted\nbrain age (years, 95% CI)", fontsize=base.FS["label"] - 1)
    base._style(ax_d)
    base._letter(ax_d, "d")
    ax_d.legend(loc="upper left", frameon=False, fontsize=base.FS["legend"])

    pdf, png = out_dir / "fig_brainage_followup.pdf", out_dir / "fig_brainage_followup.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png


def _curve(ax, c: pd.DataFrame, color: str, lw: float = 2.0, label=None):
    c = c[c["drawn"].astype(bool)].sort_values("x")
    if c.empty:
        return
    ax.fill_between(c["x"], c["ci95_low"], c["ci95_high"], color=color, alpha=0.22, linewidth=0, zorder=3)
    ax.plot(c["x"], c["mean"], color=color, lw=lw, zorder=4, label=label,
            path_effects=[patheffects.withStroke(linewidth=lw + 1.6, foreground="white")])


def make_adjusted_figure(excess: pd.DataFrame, aligned: pd.DataFrame, aligned_curves: pd.DataFrame,
                         coefs: pd.DataFrame, prof: pd.DataFrame, part: pd.DataFrame, out_dir: Path):
    cm = 1 / 2.54
    fig = plt.figure(figsize=(16 * cm, 13.5 * cm), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[0.9, 1.0])
    ax_a, ax_b, ax_c = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :2]), fig.add_subplot(gs[1, 2])
    zero = dict(ls="--", lw=0.9, color=base.COLORS["identity"], zorder=1)

    ax_a.axhline(0, **zero, label="Stable healthy expectation")
    for group in GROUPS:
        n = int((part["group"] == group).sum())
        _curve(ax_a, excess[excess["group"] == group], GROUP_COLORS[group], lw=2.2, label=f"{group} (n = {n:,})")
    ax_a.set_xlabel("Years since first scan", fontsize=base.FS["label"] - 1)
    ax_a.set_ylabel("Excess change in predicted\nbrain age (years, 95% CI)", fontsize=base.FS["label"] - 1)
    ax_a.legend(loc="upper left", frameon=False, fontsize=base.FS["legend"])
    base._style(ax_a)
    base._letter(ax_a, "a")

    color = GROUP_COLORS["Converter"]
    for _, t in aligned.sort_values(["_subject", "tau", "_ses_num"], kind="stable").groupby("_subject", sort=False):
        ax_b.plot(t["tau"], t["_dresid"], color=color, lw=0.6, alpha=0.3, zorder=2)
    ax_b.axvline(0, ls=":", lw=0.8, color="#555555", zorder=1)
    ax_b.axhline(0, ls=":", lw=0.7, color="#888888", zorder=1)
    _curve(ax_b, aligned_curves, color, lw=2.2, label="Mean (95% CI)")
    k = coefs[(coefs["model"] == "broken stick") & (coefs["converters"] == "all converters")
              & (coefs["knot"] == "first patient session")].set_index("term")["estimate"]
    if {"conv:t", "post"} <= set(k.index) and len(aligned):
        lo, hi = float(aligned["tau"].min()), float(aligned["tau"].max())
        ax_b.plot([lo, 0, hi], [k["conv:t"] * lo, 0, (k["conv:t"] + k["post"]) * hi], color="black", lw=1.2,
                  ls="--", zorder=5, label="Broken-stick fit")
    n = int((part["group"] == "Converter").sum())
    base._inset(ax_b, f"n = {n:,} converters")
    ax_b.set_xlabel("Years from the first patient session", fontsize=base.FS["label"] - 1)
    ax_b.set_ylabel("Excess over stable healthy, from\nthe first patient scan (years)", fontsize=base.FS["label"] - 1)
    if ax_b.get_legend_handles_labels()[0]:
        ax_b.legend(loc="lower right", frameon=False, fontsize=base.FS["legend"])
    base._style(ax_b)
    base._letter(ax_b, "b")

    off = "knot offset (years from first patient session)"
    for definition, style in (("all converters", "-"), ("cognitive converters", "--")):
        p = prof[prof["converters"] == definition] if len(prof) else prof
        if len(p):
            ax_c.plot(p[off], p["likelihood ratio vs best"], ls=style, marker="o", ms=3, color=color, label=definition)
    ax_c.axhline(st.chi2.ppf(0.95, 1), ls=":", lw=0.8, color="#555555")
    ax_c.set_xticks(list(KNOT_OFFSETS)[::2])
    ax_c.set_xlabel("Knot (years from the\nfirst patient session)", fontsize=base.FS["label"] - 1)
    ax_c.set_ylabel("Likelihood ratio vs best knot", fontsize=base.FS["label"] - 1)
    if ax_c.get_legend_handles_labels()[0]:
        ax_c.legend(loc="upper center", frameon=False, fontsize=base.FS["legend"] - 1)
    base._style(ax_c)
    base._letter(ax_c, "c")

    pdf, png = out_dir / "fig_brainage_followup_adjusted.pdf", out_dir / "fig_brainage_followup_adjusted.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png


def adjusted_legend_text(min_n: int) -> str:
    return (
        "Figure X - Longitudinal course of predicted brain age relative to stable healthy participants "
        "(external validation set, native scans). Linear mixed model on every scan of the stable healthy "
        "participants, converters and patients: predicted age ~ years since the first scan, group, group x time, "
        "baseline age x time and sex, with a random intercept and slope per participant. (a) Change in predicted "
        "brain age since the first scan minus the change expected for a stable healthy participant of the same "
        "baseline age (stable-healthy fixed effects), group mean at each whole year with its 95% confidence "
        f"interval, drawn where at least {min_n} participants contribute; the dashed line is the stable-healthy "
        "expectation. (b) Converters aligned on their first patient session (year 0; conversion lies between the "
        "last healthy and the first patient session): predicted brain age minus the stable-healthy expectation, "
        "relative to the first patient scan; thin lines are participants, the thick line the mean (95% CI) and the "
        "dashed line the broken-stick mixed model with a change of slope at the first patient session. (c) "
        "Likelihood ratio of the broken-stick model as the knot moves from 6 years before to 2 years after the "
        "first patient session, for all converters and for converters with a cognitive diagnosis (MCI, AD, "
        "dementia of unspecified cause, FTD); the dotted line is the 95% threshold (3.84). Chronological age in "
        "whole years, floored.\n")


def legend_text(part: pd.DataFrame, dbs_with_followup: List[str], dbs_without: List[str], min_n: int) -> str:
    n = {g: int((part["group"] == g).sum()) for g in GROUPS + [OTHER]}
    db_txt = " and ".join(aio.EXTERNAL_DATABASES.get(db, db) for db in dbs_with_followup)
    none_txt = ("; " + ", ".join(aio.EXTERNAL_DATABASES.get(db, db) for db in dbs_without)
                + " had no participant scanned more than once") if dbs_without else ""
    return (
        "Figure X - Longitudinal course of predicted brain age in the external validation set (native scans). "
        f"Participants with at least two sessions in {db_txt}{none_txt}. "
        f"(a) Participants healthy at every session (n = {n['Stable healthy']:,}). "
        f"(b) Converters, healthy at the first session and patients at the last (n = {n['Converter']:,}). "
        f"(c) Patients from the first session (n = {n['Patient']:,}). "
        "Each thin line is one participant: change in predicted brain age since the first scan versus years "
        "since the first scan (chronological age in whole years, floored). The thick line is the group mean at "
        f"each whole year since the first scan, with its 95% confidence interval, drawn where at least {min_n} "
        "participants contribute. (d) The three mean curves. The dashed line is y = x: predicted brain age "
        "increasing as fast as chronological age; the dotted line marks no change"
        + (f". {n[OTHER]:,} participants with another sequence of status are not shown" if n[OTHER] else "")
        + ".\n")


# =============================================================================
# Main
# =============================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pred-dir", help=f"folder with the prediction TSVs (default: <repo>/{aio.DEFAULT_PRED_DIR})")
    p.add_argument("--out-dir", help=f"output folder (default: <repo>/{aio.DEFAULT_OUT_DIR}/followup)")
    p.add_argument("--model-tag", default=aio.DEFAULT_MODEL_TAG,
                   help="model part of the TSV file names (default: %(default)s)")
    p.add_argument("--split", default=aio.SPLIT_EXTERNAL, choices=list(aio.SPLITS),
                   help="split to analyse (default: %(default)s = external validation)")
    p.add_argument("--n-boot", type=int, default=N_BOOT, help="bootstrap resamples (default: %(default)s)")
    p.add_argument("--min-n", type=int, default=MIN_N,
                   help="participants needed to draw the mean curve at a time point (default: %(default)s)")
    p.add_argument("--seed", type=int, default=SEED, help="bootstrap seed (default: %(default)s)")
    p.add_argument("--no-strict", action="store_true",
                   help="report labelling/encoding errors in checks.md instead of stopping")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    pred_dir = aio.resolve_dir(args.pred_dir, aio.DEFAULT_PRED_DIR)
    out = Path(args.out_dir) if args.out_dir else aio.REPO_ROOT / aio.DEFAULT_OUT_DIR / "followup"
    out.mkdir(parents=True, exist_ok=True)
    notes: List[str] = []
    try:
        d = load_followup(pred_dir, args.model_tag, args.split, not args.no_strict, notes)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    sessions = d.groupby("_subject")[aio.COL_SESSION].nunique()
    longitudinal = sessions[sessions >= 2].index
    long_d = d[d["_subject"].isin(longitudinal)]
    dbs_all = sorted(d[aio.COL_DATABASE].astype(str).unique())
    dbs_with = sorted(long_d[aio.COL_DATABASE].astype(str).unique())
    dbs_without = [db for db in dbs_all if db not in dbs_with]
    for db in dbs_all:
        dd = d[d[aio.COL_DATABASE].astype(str) == db]
        n_long = int(dd["_subject"].isin(longitudinal).groupby(dd["_subject"]).first().sum())
        notes.append(f"{aio.EXTERNAL_DATABASES.get(db, db)}: {dd['_subject'].nunique():,} participants, "
                     f"{n_long:,} with at least two sessions.")
    if long_d.empty:
        print("error: no participant with at least two sessions in this split", file=sys.stderr)
        (out / "checks.md").write_text("# Data checks\n\n" + "\n".join(f"- {n}" for n in notes) + "\n")
        return 1

    part = participant_table(long_d)
    if len(part) != len(longitudinal):                        # every followed participant is in the table
        print("error: participants lost while building the follow-up table", file=sys.stderr)
        return 2
    notes.append("Groups: " + ", ".join(f"{g} {int((part['group'] == g).sum()):,}" for g in GROUPS + [OTHER])
                 + f" (total {len(part):,} = participants with at least two sessions).")
    if (part["group"] == OTHER).any():
        seqs = part.loc[part["group"] == OTHER, "status sequence"].value_counts().head(8).to_dict()
        notes.append(f"'Other' status sequences (H = healthy, P = patient): {seqs}")

    stats = summary_table(part, args.n_boot, args.seed)
    part.to_csv(out / "brainage_followup_participants.csv", index=False)
    stats.to_csv(out / "brainage_followup_stats.csv", index=False)
    md = stats.copy()
    md["rate median [IQR]"] = [f"{base._fmt(m, 2)} [{base._fmt(a, 2)}; {base._fmt(b, 2)}]"
                               for m, a, b in zip(md["rate median"], md["rate Q1"], md["rate Q3"])]
    md["rate mean (95% CI)"] = [f"{base._fmt(m, 2)} ({base._range(a, b, 2)})" if not np.isnan(a) else base._fmt(m, 2)
                                for m, a, b in zip(md["rate mean"], md["rate mean 95% CI low"], md["rate mean 95% CI high"])]
    md["follow-up, years mean (range)"] = [f"{base._fmt(m, 2)} ({base._fmt(a, 0)}–{base._fmt(b, 0)})" for m, a, b in
                                           zip(md["follow-up mean (years)"], md["follow-up min"], md["follow-up max"])]
    md["sessions mean (range)"] = [f"{base._fmt(m, 2)} ({a}–{b})" for m, a, b in
                                   zip(md["sessions mean"], md["sessions min"], md["sessions max"])]
    md["baseline age mean (SD)"] = [f"{base._fmt(m, 2)} ({base._fmt(s, 2)})" for m, s in
                                    zip(md["baseline age mean"], md["baseline age SD"])]
    md["baseline gap mean (SD)"] = [f"{base._fmt(m, 2)} ({base._fmt(s, 2)})" for m, s in
                                    zip(md["baseline gap mean"], md["baseline gap SD"])]
    cols = ["database", "group", "participants", "scans", "sessions mean (range)", "follow-up, years mean (range)",
            "baseline age mean (SD)", "baseline gap mean (SD)", "participants with a rate",
            "rate median [IQR]", "rate mean (95% CI)"]
    (out / "brainage_followup_stats.md").write_text(
        "# Longitudinal follow-up (participants with at least two sessions)\n\n"
        "Rate = OLS slope of predicted age on chronological age over the participant's sessions (predicted years "
        "per chronological year; 1 = same pace). Chronological age floored to whole years; participants whose "
        f"sessions fall in the same year have no rate. 95% CI of the mean: bootstrap over participants "
        f"({args.n_boot:,} resamples, seed {args.seed}).\n\n" + aio.markdown_table(md[cols]) + "\n")
    curves = mean_curves(long_d, part, args.min_n)
    curves.to_csv(out / "brainage_followup_mean_curves.csv", index=False)
    pdf, png = make_figure(long_d, part, curves, out)
    legends = "## Figure legend\n\n" + legend_text(part, dbs_with, dbs_without, args.min_n)

    # mixed models, healthy reference and conversion
    m = model_data(long_d, part, notes)
    coefs, prof, ref = run_models(m, notes, args.min_n)
    coefs.to_csv(out / "brainage_followup_models.csv", index=False)
    if len(prof):
        prof.to_csv(out / "brainage_followup_knot_profile.csv", index=False)
    figures = [pdf, png]
    if ref is not None:
        m = healthy_reference(m, ref)
        excess = curves_by(m, "_t", "_excess_change", args.min_n)
        aligned = align_converters(m)
        aligned_curves = curves_by(aligned, "tau", "_dresid", args.min_n)
        excess.rename(columns={"x": "years since first scan"}).to_csv(out / "brainage_followup_excess_curves.csv",
                                                                       index=False)
        aligned_curves.rename(columns={"x": "years from first patient session"}).to_csv(
            out / "brainage_followup_aligned_curves.csv", index=False)
        figures += make_adjusted_figure(excess, aligned, aligned_curves, coefs, prof, part, out)
        legends += "\n## Adjusted figure legend\n\n" + adjusted_legend_text(args.min_n)
    else:
        notes.append("WARNING: the rate model did not converge; no healthy reference and no adjusted figure.")
    (out / "brainage_followup_models.md").write_text(models_markdown(coefs, prof, m))
    (out / "brainage_followup_text.md").write_text(legends)
    (out / "checks.md").write_text("# Data checks\n\n" + "\n".join(f"- {n}" for n in notes) + "\n")

    print("Figure : " + "\n         ".join(str(f) for f in figures))
    for name in ("brainage_followup_stats.md", "brainage_followup_models.md", "brainage_followup_mean_curves.csv",
                 "brainage_followup_participants.csv", "brainage_followup_text.md", "checks.md"):
        print(f"Output : {out / name}")
    print("\n" + md[cols].to_string(index=False))
    return 0


def models_markdown(coefs: pd.DataFrame, prof: pd.DataFrame, m: pd.DataFrame) -> str:
    def p_txt(p):
        return "< 0.001" if p < 0.001 else f"{p:.3f}"
    out = ["# Longitudinal mixed models (external validation, participants with at least two sessions)\n",
           "Predicted age (years) on every scan; t = years since the first scan (floored ages); baseline age "
           f"centred on {base._fmt(m.attrs.get('age centre', np.nan), 1)} years; reference = stable healthy, "
           "female. Wald 95% CI. Rates in predicted years per chronological year.\n"]
    if coefs.empty:
        return "\n".join(out + ["No model could be fitted.\n"])
    for (model, definition, knot), g in coefs.groupby(["model", "converters", "knot"], sort=False):
        g = g[g["term"].isin(KEY_TERMS[model])]
        head = f"## {model.capitalize()}, {definition}" + (f", knot at {knot}" if knot else "")
        meta = (f"{g['participants'].iloc[0]:,} participants ({g['converters (participants)'].iloc[0]:,} converters), "
                f"{g['scans'].iloc[0]:,} scans; random {g['random effects'].iloc[0]}.")
        table = pd.DataFrame({"term": g["label"],
                              "estimate (95% CI)": [f"{base._fmt(e, 2)} ({base._range(a, b, 2)})" for e, a, b
                                                    in zip(g["estimate"], g["ci95_low"], g["ci95_high"])],
                              "p": [p_txt(p) for p in g["p"]]})
        out += [head + "\n", meta + "\n", aio.markdown_table(table) + "\n"]
    fitted = prof.dropna(subset=["log-likelihood (ML)"]) if len(prof) else prof
    if len(fitted):
        best = fitted.loc[fitted.groupby("converters")["log-likelihood (ML)"].idxmax()]
        lines = []
        for _, b in best.iterrows():
            ok = prof[(prof["converters"] == b["converters"]) & prof["within 95% (LR <= 3.84)"]]
            off = "knot offset (years from first patient session)"
            sel = prof[(prof["converters"] == b["converters"]) & prof["log-likelihood (ML)"].notna()]
            lines.append(f"- {b['converters']}: best knot {int(b[off]):+d} years from the first patient session "
                         f"(offsets within the 95% likelihood interval: {int(ok[off].min()):+d} to "
                         f"{int(ok[off].max()):+d}; offsets fitted {int(sel[off].min()):+d} to {int(sel[off].max()):+d}; "
                         f"random {b['random effects']} at every offset).")
        out += ["## Knot profile (maximum likelihood, broken-stick model)\n",
                "Offsets with fewer converters with scans on both sides of the knot than the curve threshold are "
                "not fitted.\n", "\n".join(lines) + "\n"]
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
