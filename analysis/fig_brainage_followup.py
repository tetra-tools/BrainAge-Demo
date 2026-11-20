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

Outputs (in ``--out-dir``):
  fig_brainage_followup.pdf / .png  (a-c) one line per participant, by group: change in predicted age
                                    since the first scan vs years since the first scan, with the group
                                    mean curve (95% CI); (d) the three mean curves together
  brainage_followup_mean_curves.csv mean change in predicted age per year since the first scan
  brainage_followup_stats.csv/.md   per database and group: participants, sessions, follow-up,
                                    baseline age and gap, rate (median, IQR, mean with 95% CI)
  brainage_followup_participants.csv one row per participant (group, sessions, rate)
  brainage_followup_text.md         figure legend
  checks.md                         data checks

Example (from the repository root):
  python analysis/fig_brainage_followup.py --pred-dir data/predictions --out-dir results/analysis/followup
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from matplotlib import patheffects
from scipy import stats as st

import analysis_io as aio
import fig_brainage_scatter as base          # shared style (fonts, axes) and number formatting

plt = base.plt

GROUPS = ["Stable healthy", "Converter", "Patient"]
OTHER = "Other"
GROUP_COLORS = {"Stable healthy": "#0072B2", "Converter": "#E69F00", "Patient": "#CC79A7"}
N_BOOT = 1000
SEED = 2026
MIN_N = 5            # mean curve drawn at a time point only if at least MIN_N participants contribute


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
        conv_age = ages[np.argmax(status == 0)] if group == "Converter" else np.nan
        rows.append({
            "participant": subject, "database": g[aio.COL_DATABASE].iloc[0], "group": group,
            "sessions": g[aio.COL_SESSION].nunique(), "scans": len(g),
            "baseline age": ages[0], "last age": ages[-1], "follow-up (years)": ages[-1] - ages[0],
            "baseline gap": g["_gap"].iloc[0], "last gap": g["_gap"].iloc[-1],
            "rate (predicted years per year)": rate, "age at first patient session": conv_age,
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
    ax_d.plot([0, t_max], [0, t_max], **identity, label="y = x (same pace as chronological age)")
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
    (out / "brainage_followup_text.md").write_text(
        "## Figure legend\n\n" + legend_text(part, dbs_with, dbs_without, args.min_n))
    (out / "checks.md").write_text("# Data checks\n\n" + "\n".join(f"- {n}" for n in notes) + "\n")

    print(f"Figure : {pdf}\n         {png}")
    for name in ("brainage_followup_stats.md", "brainage_followup_mean_curves.csv",
                 "brainage_followup_participants.csv", "brainage_followup_text.md", "checks.md"):
        print(f"Output : {out / name}")
    print("\n" + md[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
