#!/usr/bin/env python3
"""Inventory of databases and diagnoses before cleaning, and of their grouping.

Reads the prediction TSVs like the Figure 13 forest plot (native original scans of both
files, ``drop_duplicates``), keeps every participant (healthy and clinical), and writes in
``--out-dir/inventory/``:

  databases.csv         database x split: scans, participants, healthy / other scans, age range
  diagnoses_raw.csv     every raw value of neurological_diagnosis and psychiatric_diagnosis, as
                        stored: scans, participants, databases, and the groups assigned by the
                        notebook rules (analysis/diagnosis_groups.py)
  diagnosis_tokens.csv  every token after splitting on , ; | : rule applied and group
  diagnosis_groups.csv  groups after cleaning: scans and participants per split, databases
  inventory.md          the four tables in Markdown

Example (from the repository root):
  python analysis/inventory_databases_diagnoses.py --pred-dir data/predictions
"""

from __future__ import annotations

import argparse
import sys
from typing import List

import pandas as pd

import analysis_io as aio
import diagnosis_groups as dg

DX_COLUMNS = {"neuro": aio.COL_NEURO_DX, "psych": aio.COL_PSYCH_DX}
MISSING = "<missing>"


def _db_list(frame: pd.DataFrame) -> str:
    counts = frame[aio.COL_DATABASE].astype(str).value_counts()
    return "; ".join(f"{db} ({n:,})" for db, n in counts.items())


def _split_counts(frame: pd.DataFrame) -> dict:
    out = {}
    for split in aio.SPLITS:
        f = frame[frame["_split"] == split]
        out[f"scans {split}"] = len(f)
        out[f"participants {split}"] = f["_subject"].nunique()
    return out


def database_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, db), f in df.groupby(["_split", aio.COL_DATABASE], sort=True):
        healthy = f[aio.COL_HEALTHY] == 1
        rows.append({"split": split, "database": db, "scans": len(f),
                     "participants": f["_subject"].nunique(),
                     "healthy scans (healthy == 1)": int(healthy.sum()),
                     "other scans": int((~healthy).sum()),
                     "age min": f[aio.COL_AGE].min(), "age max": f[aio.COL_AGE].max()})
    out = pd.DataFrame(rows)
    order = {s: i for i, s in enumerate(aio.SPLITS)}
    return out.sort_values(["split", "database"], key=lambda c: c.map(order) if c.name == "split" else c)


def raw_diagnosis_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain, col in DX_COLUMNS.items():
        key = df[col].where(df[col].notna(), MISSING).astype(str)
        for value, f in df.groupby(key, sort=False):
            original = None if value == MISSING else f[col].iloc[0]
            groups, is_cn = dg.string_groups(original, domain)
            rows.append({"domain": dg.DOMAINS[domain], "raw value": value,
                         "scans": len(f), "participants": f["_subject"].nunique(),
                         **_split_counts(f),
                         "healthy == 1 scans": int((f[aio.COL_HEALTHY] == 1).sum()),
                         "groups (notebook rules)": " + ".join(groups) if groups else "",
                         "CN flag": is_cn,
                         "all tokens excluded": dg._all_tokens_excluded(original, domain),
                         "databases": _db_list(f)})
    return (pd.DataFrame(rows).sort_values(["domain", "scans"], ascending=[True, False])
            .reset_index(drop=True))


def token_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain, col in DX_COLUMNS.items():
        tok = df[[col, "_subject", "_split", aio.COL_DATABASE]].copy()
        tok["token"] = tok[col].map(lambda v: sorted(set(dg.split_tokens(v))))
        tok = tok.explode("token").dropna(subset=["token"])
        for token, f in tok.groupby("token", sort=False):
            rule, groups = dg.token_rule(token, domain)
            rows.append({"domain": dg.DOMAINS[domain], "token": token, "rule": rule, "group": groups,
                         "scans": len(f), "participants": f["_subject"].nunique(),
                         **_split_counts(f), "databases": _db_list(f)})
    return (pd.DataFrame(rows).sort_values(["domain", "rule", "scans"], ascending=[True, True, False])
            .reset_index(drop=True))


def group_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain, col in DX_COLUMNS.items():
        res = df[col].map(lambda v: dg.string_groups(None if pd.isna(v) else v, domain))
        labels = []
        for groups, is_cn in res:
            if groups:
                labels.append(groups)
            elif is_cn:
                labels.append(["CN (no diagnosis group)"])
            else:
                labels.append(["none (no group, not CN)"])
        g = df[["_subject", "_split", aio.COL_DATABASE, aio.COL_HEALTHY]].assign(group=labels).explode("group")
        for group, f in g.groupby("group", sort=False):
            rows.append({"domain": dg.DOMAINS[domain], "group": group, **_split_counts(f),
                         "healthy == 1 scans": int((f[aio.COL_HEALTHY] == 1).sum()),
                         "databases": _db_list(f)})
    out = pd.DataFrame(rows)
    out["_total"] = out[[f"scans {s}" for s in aio.SPLITS]].sum(axis=1)
    return (out.sort_values(["domain", "_total"], ascending=[True, False])
            .drop(columns="_total").reset_index(drop=True))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pred-dir", help=f"folder with the prediction TSVs (default: <repo>/{aio.DEFAULT_PRED_DIR})")
    p.add_argument("--out-dir", help=f"output folder; tables go to <out-dir>/inventory "
                                     f"(default: <repo>/{aio.DEFAULT_OUT_DIR})")
    p.add_argument("--model-tag", default=aio.DEFAULT_MODEL_TAG,
                   help="model part of the TSV file names (default: %(default)s)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    pred_dir = aio.resolve_dir(args.pred_dir, aio.DEFAULT_PRED_DIR)
    out = aio.resolve_dir(args.out_dir, aio.DEFAULT_OUT_DIR) / "inventory"
    out.mkdir(parents=True, exist_ok=True)
    notes: List[str] = []
    try:
        df = aio.native_scans(aio.read_all(pred_dir, args.model_tag, notes), notes)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    df = df.assign(_subject=aio.subject_key(df))

    tables = {
        "databases": database_table(df),
        "diagnoses_raw": raw_diagnosis_table(df),
        "diagnosis_tokens": token_table(df),
        "diagnosis_groups": group_table(df),
    }
    titles = {
        "databases": "Databases (native original scans, all participants)",
        "diagnoses_raw": "Raw diagnosis values before cleaning, with the groups assigned by the notebook rules",
        "diagnosis_tokens": "Tokens after splitting on , ; | and the rule applied to each",
        "diagnosis_groups": "Diagnosis groups after cleaning (a scan can belong to several groups)",
    }
    md = ["# Inventory of databases and diagnoses", "",
          "Rows: native original scans read as in the Figure 13 forest plot (healthy + prediction files, "
          "drop_duplicates), all participants. Grouping rules: analysis/diagnosis_groups.py (copied from the "
          "notebook).", "", *[f"- {n}" for n in notes]]
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
        md += ["", f"## {titles[name]}", "", aio.markdown_table(table)]
    (out / "inventory.md").write_text("\n".join(md) + "\n")

    tok = tables["diagnosis_tokens"]
    print(f"Databases: {df[aio.COL_DATABASE].nunique()}  |  scans: {len(df):,}  |  "
          f"participants: {df['_subject'].nunique():,}")
    for domain in dg.DOMAINS.values():
        raw_n = (tables["diagnoses_raw"]["domain"] == domain).sum()
        unmapped = tok[(tok["domain"] == domain) & tok["rule"].str.startswith("NOT MAPPED")]
        multi = tok[(tok["domain"] == domain) & (tok["rule"] == "matched by several rules")]
        print(f"{domain}: {raw_n} raw values, {(tok['domain'] == domain).sum()} tokens, "
              f"{len(unmapped)} not mapped, {len(multi)} matched by several rules")
        if len(unmapped):
            print("   not mapped: " + "; ".join(f"{t} ({n:,})" for t, n in zip(unmapped["token"], unmapped["scans"])))
        if len(multi):
            print("   several rules: " + "; ".join(f"{t} -> {g}" for t, g in zip(multi["token"], multi["group"])))
    print(f"Tables written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
