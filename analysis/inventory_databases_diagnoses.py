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
  diagnosis_correspondence.csv
                        proposed classification (analysis/diagnosis_terms.csv): every term found in
                        each column with its type, group, the notebook rule, and scans/participants
                        per split
  diagnosis_groups_proposed.csv
                        scans per proposed group (both columns of a scan together), participants,
                        healthy scans, the Other group a split merges it into (< 30 scans), and the
                        patients left without any group (Unclassified_patient)
  inventory.md          all tables in Markdown

Example (from the repository root):
  python analysis/inventory_databases_diagnoses.py --pred-dir data/predictions
"""

from __future__ import annotations

import argparse
import sys
from typing import List

import numpy as np
import pandas as pd

import analysis_io as aio
import diagnosis_classification as dc
import diagnosis_groups as dg

DX_COLUMNS = {"neuro": aio.COL_NEURO_DX, "psych": aio.COL_PSYCH_DX}
MISSING = "<missing>"


def _db_list(frame: pd.DataFrame) -> str:
    counts = frame[aio.COL_DATABASE].astype(str).value_counts()
    return "; ".join(f"{db} ({n:,})" for db, n in counts.items())


SPLIT_ORDER = (aio.SPLIT_TRAIN, *aio.SPLITS)


def _split_counts(frame: pd.DataFrame, splits) -> dict:
    out = {}
    for split in splits:
        f = frame[frame["_split"] == split]
        out[f"scans {split}"] = len(f)
        out[f"participants {split}"] = f["_subject"].nunique()
    return out


def database_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, db), f in df.groupby(["_split", aio.COL_DATABASE], sort=True, dropna=False):   # missing labels kept
        healthy = f[aio.COL_HEALTHY] == 1
        age = np.floor(pd.to_numeric(f[aio.COL_AGE], errors="coerce"))  # floored, as in every analysis
        rows.append({"split": split, "database": MISSING if pd.isna(db) else db, "scans": len(f),
                     "participants": f["_subject"].nunique(),
                     "healthy scans (healthy == 1)": int(healthy.sum()),
                     "other scans": int((~healthy).sum()),
                     "age min": age.min(), "age max": age.max()})
    out = pd.DataFrame(rows)
    order = {s: i for i, s in enumerate(SPLIT_ORDER)}
    return out.sort_values(["split", "database"], key=lambda c: c.map(order) if c.name == "split" else c)


def raw_diagnosis_table(df: pd.DataFrame, splits) -> pd.DataFrame:
    rows = []
    for domain, col in DX_COLUMNS.items():
        key = df[col].where(df[col].notna(), MISSING).astype(str)
        for value, f in df.groupby(key, sort=False):
            original = None if value == MISSING else f[col].iloc[0]
            groups, is_cn = dg.string_groups(original, domain)
            rows.append({"domain": dg.DOMAINS[domain], "raw value": value,
                         "scans": len(f), "participants": f["_subject"].nunique(),
                         **_split_counts(f, splits),
                         "healthy == 1 scans": int((f[aio.COL_HEALTHY] == 1).sum()),
                         "groups (notebook rules)": " + ".join(groups) if groups else "",
                         "groups (proposed)": " + ".join(sorted(_cell_groups(original))),
                         "CN flag": is_cn,
                         "all tokens excluded": dg._all_tokens_excluded(original, domain),
                         "databases": _db_list(f)})
    return (pd.DataFrame(rows).sort_values(["domain", "scans"], ascending=[True, False])
            .reset_index(drop=True))


def token_table(df: pd.DataFrame, splits) -> pd.DataFrame:
    rows = []
    for domain, col in DX_COLUMNS.items():
        tok = df[[col, "_subject", "_split", aio.COL_DATABASE]].copy()
        tok["token"] = tok[col].map(lambda v: sorted(set(dg.split_tokens(v))))
        tok = tok.explode("token").dropna(subset=["token"])
        for token, f in tok.groupby("token", sort=False):
            rule, groups = dg.token_rule(token, domain)
            rows.append({"domain": dg.DOMAINS[domain], "token": token, "rule": rule, "group": groups,
                         "scans": len(f), "participants": f["_subject"].nunique(),
                         **_split_counts(f, splits), "databases": _db_list(f)})
    return (pd.DataFrame(rows).sort_values(["domain", "rule", "scans"], ascending=[True, True, False])
            .reset_index(drop=True))


def group_table(df: pd.DataFrame, splits) -> pd.DataFrame:
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
            rows.append({"domain": dg.DOMAINS[domain], "group": group, **_split_counts(f, splits),
                         "healthy == 1 scans": int((f[aio.COL_HEALTHY] == 1).sum()),
                         "databases": _db_list(f)})
    out = pd.DataFrame(rows)
    out["_total"] = out[[f"scans {s}" for s in splits]].sum(axis=1)
    return (out.sort_values(["domain", "_total"], ascending=[True, False])
            .drop(columns="_total").reset_index(drop=True))


def _cell_groups(dx) -> set:
    flags, _, _ = dc.classify_cell(dx)
    return flags[dc.NEURO] | flags[dc.PSYCH]


def correspondence_table(df: pd.DataFrame, splits) -> pd.DataFrame:
    """Every term found in each diagnosis column, with the proposed classification and counts."""
    rows = []
    for domain, col in DX_COLUMNS.items():
        tok = df[[col, "_subject", "_split"]].copy()
        tok["term"] = tok[col].map(lambda v: sorted(set(dc.split_terms(v))))
        tok = tok.explode("term").dropna(subset=["term"])
        for term, f in tok.groupby("term", sort=False):
            typ, group, dom = dc.term_info(term)
            rule, nb_group = dg.token_rule(term, domain)
            rows.append({"column": dg.DOMAINS[domain], "term": term, "type": typ, "group (proposed)": group,
                         "condition": dc.TYPE_CONDITION[typ], "domain (proposed)": dom,
                         "notebook rule": rule, "notebook group": nb_group,
                         "scans": len(f), "participants": f["_subject"].nunique(), **_split_counts(f, splits)})
    order = {t: i for i, t in enumerate(["CN", "GROUP", "DEMENTIA", "PPA", "MOVEMENT_SYNDROME", "OTHER_DISEASE",
                                         "OTHER_COGNITIVE", "ATTRIBUTION", "SYMPTOM", "DESCRIPTOR", "MISSING",
                                         "UNMAPPED"])}
    out = pd.DataFrame(rows)
    out["_t"] = out["type"].map(order)
    return (out.sort_values(["column", "_t", "scans"], ascending=[True, True, False])
            .drop(columns="_t").reset_index(drop=True))


def scan_group_labels(df: pd.DataFrame) -> pd.Series:
    """Proposed groups of each scan (both columns), as a frozenset; computed once per distinct pair."""
    pairs = df[[aio.COL_NEURO_DX, aio.COL_PSYCH_DX]].astype(object).where(
        df[[aio.COL_NEURO_DX, aio.COL_PSYCH_DX]].notna(), None)
    cache = {}
    out = []
    for n, p in zip(pairs[aio.COL_NEURO_DX], pairs[aio.COL_PSYCH_DX]):
        if (n, p) not in cache:
            cache[(n, p)] = dc.scan_groups(n, p)
        out.append(cache[(n, p)])
    return pd.Series(out, index=df.index)


def proposed_group_table(df: pd.DataFrame, splits) -> pd.DataFrame:
    """Scans per proposed group, counted on whole scans (both columns together)."""
    groups = scan_group_labels(df)
    healthy = df[aio.COL_HEALTHY] == 1
    labels = []
    for g, h in zip(groups, healthy):
        if g:
            labels.append(sorted(g))
        else:
            labels.append(["No group, healthy (healthy == 1)" if h else dc.UNCLASSIFIED])
    long = df[["_subject", "_split", aio.COL_DATABASE, aio.COL_HEALTHY]].assign(group=labels).explode("group")
    rows = []
    for group, f in long.groupby("group", sort=False):
        rows.append({"group": group, "domain": dc.group_domain(group), **_split_counts(f, splits),
                     "healthy == 1 scans": int((f[aio.COL_HEALTHY] == 1).sum()), "databases": _db_list(f)})
    out = pd.DataFrame(rows)
    for split in splits:
        merged = dc.rare_groups(dict(zip(out["group"], out[f"scans {split}"])))
        out[f"merged into (< {dc.MIN_SCANS} scans) {split}"] = out["group"].map(lambda g: merged.get(g, ""))
    order = {g: i for i, g in enumerate(dc.all_groups() + [dc.UNCLASSIFIED, "No group, healthy (healthy == 1)"])}
    out["_o"] = out["group"].map(order).fillna(len(order))
    return out.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pred-dir", help=f"folder with the prediction TSVs (default: <repo>/{aio.DEFAULT_PRED_DIR})")
    p.add_argument("--out-dir", help=f"output folder; tables go to <out-dir>/inventory "
                                     f"(default: <repo>/{aio.DEFAULT_OUT_DIR})")
    p.add_argument("--model-tag", default=aio.DEFAULT_MODEL_TAG,
                   help="model part of the TSV file names (default: %(default)s)")
    p.add_argument("--train-file", help="training-set table (default: <pred-dir>/predictions_train_<tag>.tsv if present)")
    p.add_argument("--no-train", action="store_true", help="do not add the training set")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    pred_dir = aio.resolve_dir(args.pred_dir, aio.DEFAULT_PRED_DIR)
    out = aio.resolve_dir(args.out_dir, aio.DEFAULT_OUT_DIR) / "inventory"
    out.mkdir(parents=True, exist_ok=True)
    notes: List[str] = []
    train_file = aio.training_file(pred_dir, args.model_tag, args.train_file, args.no_train, notes)
    try:
        raw = aio.read_all(pred_dir, args.model_tag, notes)
        if train_file is not None:
            raw = pd.concat([raw, aio.read_training(train_file, notes)], ignore_index=True)
        df = aio.native_scans(raw, notes)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    df = df.assign(_subject=aio.subject_key(df))
    # The inventory never stops on labelling problems: it lists them so they can be fixed.
    errors, warnings = aio.audit_labels(raw, df)
    accounting = aio.scan_accounting(raw, df)
    accounting.to_csv(out / "scan_accounting.csv", index=False)

    splits = [s for s in SPLIT_ORDER if (df["_split"] == s).any()]
    tables = {
        "databases": database_table(df),
        "diagnoses_raw": raw_diagnosis_table(df, splits),
        "diagnosis_tokens": token_table(df, splits),
        "diagnosis_groups": group_table(df, splits),
        "diagnosis_correspondence": correspondence_table(df, splits),
        "diagnosis_groups_proposed": proposed_group_table(df, splits),
    }
    titles = {
        "databases": "Databases (native original scans, all participants)",
        "diagnoses_raw": "Raw diagnosis values before cleaning, with the groups assigned by the notebook rules",
        "diagnosis_tokens": "Tokens after splitting on , ; | and the rule applied to each",
        "diagnosis_groups": "Diagnosis groups after cleaning (a scan can belong to several groups)",
        "diagnosis_correspondence": "Proposed classification: every term found, with its type, group and counts "
                                    "(analysis/diagnosis_terms.csv)",
        "diagnosis_groups_proposed": "Proposed groups, counted on whole scans (both diagnosis columns; a scan "
                                     "can belong to several groups)",
    }
    md = ["# Inventory of databases and diagnoses", "",
          "Rows: native original scans read as in the Figure 13 forest plot (healthy + prediction files, "
          "drop_duplicates), all participants. Grouping rules: analysis/diagnosis_groups.py (copied from the "
          "notebook).", "", *[f"- {n}" for n in notes]]
    md += ["", "## Labelling and encoding problems", "",
           *([f"- ERROR (scans dropped or misgrouped): {e}" for e in errors]
             + [f"- WARNING: {w}" for w in warnings] or ["None found."]),
           "", "## Scan accounting", "", aio.markdown_table(accounting)]
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
    corr = tables["diagnosis_correspondence"]
    unm = corr[corr["type"] == "UNMAPPED"]
    print(f"Proposed classification: {len(corr)} terms, {len(unm)} not in analysis/diagnosis_terms.csv"
          + (": " + "; ".join(f"{t} ({n:,})" for t, n in zip(unm["term"], unm["scans"])) if len(unm) else ""))
    prop = tables["diagnosis_groups_proposed"]
    unc = prop[prop["group"] == dc.UNCLASSIFIED]
    if len(unc):
        print("Patients without any proposed group (Unclassified_patient): "
              + ", ".join(f"{s} {int(unc[f'scans {s}'].iloc[0]):,} scans" for s in splits))
    for e in errors:
        print(f"ERROR: {e}")
    for w in warnings:
        print(f"WARNING: {w}")
    print(f"Tables written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
