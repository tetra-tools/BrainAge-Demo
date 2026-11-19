"""Synthetic end-to-end tests of the analysis scripts (no real data needed).

Run from the repository root:  pytest analysis/tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_DIR))

import analysis_io as aio  # noqa: E402
import diagnosis_groups as dg  # noqa: E402
import fig_brainage_scatter as fig  # noqa: E402
import inventory_databases_diagnoses as inv  # noqa: E402

TAG = aio.DEFAULT_MODEL_TAG
NEURO_DX = ["Cognitively Normal", "Alzheimer's Disease", "MCI, Other", "Dementia",
            "Parkinson's Disease, LRRK2", "Undetermined", "Parkinson's Disease Dementia"]
PSYCH_DX = ["Healthy", "Major Depressive Disorder; Anxiety Disorder", "surgery", None]


def _rows(rng, split, databases, n_subjects, healthy, start):
    rows = []
    for i in range(start, start + n_subjects):
        db = databases[i % len(databases)]
        age0 = rng.uniform(20, 85)
        for ses in range(1 + (i % 3 == 0)):
            age = float(np.floor(age0 + 2 * ses))                 # whole years, as in the real files
            pred = 0.85 * age + 7 + rng.normal(0, 4)
            for aug in ("native", "flip", "noise"):
                rows.append({
                    "subject_id": f"sub-{i:05d}", "session_id": f"ses-{ses + 1}", "database": db,
                    "augmentation": aug, "is_original": aug == "native", "split": split,
                    "t1_path": f"/x/{db}/{i}/{ses}.nii.gz", "path": f"/pt/{db}_{i}_{ses}_{aug}.pt",
                    "image_augmentation": aug, "subject_image_count": 1, "subject_sample_count": 3,
                    "infile_index": 0, "ptfile_index": 0, "apoe": np.nan,
                    "age": age, "predicted_age_years": pred + (0 if aug == "native" else rng.normal(0, 1)),
                    "sex": ["female", "male", "Female ", "F"][i % 4],
                    "healthy": int(healthy),
                    "neurological_diagnosis": "Cognitively Normal" if healthy else NEURO_DX[i % len(NEURO_DX)],
                    "psychiatric_diagnosis": "Healthy" if healthy else PSYCH_DX[i % len(PSYCH_DX)],
                })
    return rows


@pytest.fixture(scope="module")
def pred_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("predictions")
    rng = np.random.default_rng(0)
    for split, dbs in (("test", ["adni", "ukb", "hcp"]), ("validation", ["camcan", "oasis3"])):
        healthy = pd.DataFrame(_rows(rng, split, dbs, 120, True, 0))
        clinical = pd.DataFrame(_rows(rng, split, dbs, 60, False, 10_000))
        clinical = pd.concat([clinical, healthy.iloc[:9]])          # identical rows in both files
        healthy.to_csv(d / aio.FILE_TEMPLATES["healthy_file"].format(split=split, tag=TAG), sep="\t", index=False)
        clinical.to_csv(d / aio.FILE_TEMPLATES["prediction_file"].format(split=split, tag=TAG), sep="\t", index=False)
    return d


def test_figure_script(pred_dir, tmp_path):
    assert fig.main(["--pred-dir", str(pred_dir), "--out-dir", str(tmp_path), "--n-boot", "200"]) == 0
    for name in ("fig_brainage_scatter.pdf", "fig_brainage_scatter.png", "brainage_stats.csv",
                 "brainage_stats.md", "brainage_mae_by_subgroup.csv", "brainage_mae_by_subgroup.md",
                 "brainage_text_snippets.md", "checks.md", "scans_abs_gap_gt15.csv"):
        assert (tmp_path / name).stat().st_size > 0, name

    # MAE recomputed by hand: native original scans, healthy == 1, floored age, per split
    stats = pd.read_csv(tmp_path / "brainage_stats.csv")
    for split, label in (("test", "Test"), ("validation", "External validation")):
        h = pd.read_csv(pred_dir / aio.FILE_TEMPLATES["healthy_file"].format(split=split, tag=TAG), sep="\t")
        h = h[(h["augmentation"] == "native") & h["is_original"] & (h["healthy"] == 1)]
        expected = (h["predicted_age_years"] - np.floor(h["age"])).abs().mean()
        got = stats.query("group == @label and statistic == 'MAE (years)'")["estimate"].item()
        assert got == pytest.approx(expected)
        n = stats.query("group == @label and statistic == 'N scans'")["estimate"].item()
        assert n == len(h)                                           # duplicated rows counted once

    sub = pd.read_csv(tmp_path / "brainage_mae_by_subgroup.csv")
    assert set(sub["factor"]) == {"age (years)", "sex", "augmentation"}
    aug = sub[(sub["factor"] == "augmentation") & (sub["group"] == "Test")]
    assert set(aug["level"]) == {"native", "flip", "noise", "all rows (native + augmented)"}
    native_mae = aug.loc[aug["level"] == "native", "MAE"].item()
    overall = stats.query("group == 'Test' and statistic == 'MAE (years)'")["estimate"].item()
    assert native_mae == pytest.approx(overall)
    sex = sub[(sub["factor"] == "sex") & (sub["group"] == "Test")]
    assert set(sex["level"]) == {"female", "male", "other / missing"}
    filled = sub[sub["N rows"] > 0]                                   # empty subgroups are reported as NA
    assert (filled["ci95_low"] <= filled["MAE"]).all() and (filled["MAE"] <= filled["ci95_high"]).all()
    assert sub.loc[sub["N rows"] == 0, "MAE"].isna().all()

    text = (tmp_path / "brainage_text_snippets.md").read_text()
    assert "external validation set" in text and "hyperparameter selection" in text
    assert "Training" not in set(stats["group"]) and "training scans" not in text


def test_figure_script_with_training(pred_dir, tmp_path):
    rng = np.random.default_rng(5)
    train = pd.DataFrame(_rows(rng, "train", ["adni", "ukb", "hcp"], 200, True, 20_000)).drop(columns="sex")
    train_file = tmp_path / "predictions_train.tsv"
    train.to_csv(train_file, sep="\t", index=False)
    out = tmp_path / "out"
    assert fig.main(["--pred-dir", str(pred_dir), "--out-dir", str(out), "--n-boot", "200",
                     "--train-file", str(train_file)]) == 0
    stats = pd.read_csv(out / "brainage_stats.csv")
    native = train[(train["augmentation"] == "native") & train["is_original"]]
    expected = (native["predicted_age_years"] - np.floor(native["age"])).abs().mean()
    got = stats.query("group == 'Training' and statistic == 'MAE (years)'")["estimate"].item()
    assert got == pytest.approx(expected)
    assert "Training" in (out / "brainage_stats.md").read_text()
    assert "native training scans" in (out / "brainage_text_snippets.md").read_text()
    sub = pd.read_csv(out / "brainage_mae_by_subgroup.csv")
    tr_sex = sub[(sub["group"] == "Training") & (sub["factor"] == "sex") & (sub["N rows"] > 0)]
    assert set(tr_sex["level"]) == {"other / missing"}                      # no sex column in this file


def test_figure_script_missing_training_file(pred_dir, tmp_path):
    assert fig.main(["--pred-dir", str(pred_dir), "--out-dir", str(tmp_path),
                     "--train-file", str(tmp_path / "nope.tsv")]) == 2


def test_figure_script_bad_bins(pred_dir, tmp_path):
    assert fig.main(["--pred-dir", str(pred_dir), "--out-dir", str(tmp_path), "--age-bins", "60", "40"]) == 2


def test_missing_files(tmp_path):
    assert fig.main(["--pred-dir", str(tmp_path), "--out-dir", str(tmp_path)]) == 2
    assert inv.main(["--pred-dir", str(tmp_path), "--out-dir", str(tmp_path)]) == 2


def test_inventory_script(pred_dir, tmp_path):
    assert inv.main(["--pred-dir", str(pred_dir), "--out-dir", str(tmp_path)]) == 0
    out = tmp_path / "inventory"
    dbs = pd.read_csv(out / "databases.csv")
    assert set(dbs["database"]) == {"adni", "ukb", "hcp", "camcan", "oasis3"}
    raw = pd.read_csv(out / "diagnoses_raw.csv", keep_default_na=False)
    neuro = raw[raw["domain"] == "neurological"].set_index("raw value")
    assert neuro.loc["Alzheimer's Disease", "groups (notebook rules)"] == "AD"
    assert neuro.loc["Parkinson's Disease, LRRK2", "groups (notebook rules)"] == "Parkinson + Mutation_PD"
    assert neuro.loc["Undetermined", "groups (notebook rules)"] == ""
    assert neuro.loc["Parkinson's Disease Dementia", "groups (notebook rules)"] == ""
    psych = raw[raw["domain"] == "psychiatric"].set_index("raw value")
    assert psych.loc["Major Depressive Disorder; Anxiety Disorder", "groups (notebook rules)"] == "MDD + Anxiety"
    assert "<missing>" in psych.index
    tok = pd.read_csv(out / "diagnosis_tokens.csv", keep_default_na=False)
    rule = dict(zip(tok["token"], tok["rule"]))
    assert rule["Other"] == "excluded token (ignored)"
    assert rule["Parkinson's Disease Dementia"] == "NOT MAPPED (ignored)"
    groups = pd.read_csv(out / "diagnosis_groups.csv")
    assert {"AD", "MCI", "MDD", "Anxiety", "CN (no diagnosis group)"} <= set(groups["group"])
    assert (out / "inventory.md").stat().st_size > 0


def test_string_groups_matches_notebook_rules():
    assert dg.string_groups("Alzheimer's Disease, Other", "neuro") == (["AD"], False)
    assert dg.string_groups("Other, Unknown", "neuro") == ([], False)
    assert dg.string_groups("Cognitively Normal", "neuro") == ([], True)
    assert dg.string_groups(None, "psych") == ([], False)
    assert dg.string_groups("Healthy", "psych") == ([], True)
    assert dg.string_groups("surgery", "psych") == ([], False)                 # excluded before "Pain"
