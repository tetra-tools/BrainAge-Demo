"""Synthetic tests of fig_figure13_forest.py (Figure 13 forest plot), with planted group effects.

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis_io as aio  # noqa: E402
import fig_figure13_forest as f13  # noqa: E402
from test_analysis import build_pred_dir  # noqa: E402

TAG = aio.DEFAULT_MODEL_TAG
# (neurological cell, psychiatric cell, participants per split, planted difference in years)
PATIENTS = [
    ("Alzheimer's Disease", "Healthy", 110, 4.0),
    ("MCI", None, 90, 2.0),
    ("Parkinson's Disease", None, 70, 1.5),
    ("Parkinson's Disease, LRRK2", None, 30, 1.5),            # Parkinson + Mutation_PD
    ("Cognitively Normal", "Major Depressive Disorder", 80, 1.0),
    ("Alzheimer's Disease", "Major Depressive Disorder", 15, 5.0),  # AD + Depression (interaction)
    ("Cognitively Normal", "Bipolar Disorder", 18, 2.5),      # < 30: merged into Other_psych_disease
    ("Cognitively Normal", "Schizophrenia", 16, 2.5),          # < 30: merged into Other_psych_disease
    ("Cognitively Normal", "Hallucinations", 12, 0.0),         # Other_psych_symptom, < 30, never merged
    ("Undetermined", None, 14, 0.0),                           # no group: Unclassified_patient
]
N_HEALTHY = 500
NOISE_SD = 2.5


def _row(i, split, db, ses, age, pred, sex, healthy, neuro, psych):
    return {"subject_id": f"sub-{i:05d}", "session_id": f"ses-{ses + 1}", "database": db,
            "augmentation": "native", "is_original": True, "split": split,
            "t1_path": f"/x/{db}/{i}/{ses}.nii.gz", "path": f"/pt/{db}_{i}_{ses}.pt",
            "image_augmentation": "native", "subject_image_count": 1, "subject_sample_count": 1,
            "infile_index": 0, "ptfile_index": 0, "apoe": np.nan, "age": age, "predicted_age_years": pred,
            "sex": sex, "healthy": healthy, "neurological_diagnosis": neuro, "psychiatric_diagnosis": psych}


def build_figure13_dir(d: Path, healthy_group: bool = False) -> Path:
    rng = np.random.default_rng(13)
    for split, dbs in (("test", ["adni", "ukb", "ppmi"]), ("validation", ["camcan", "oasis3"])):
        healthy, clinical = [], []
        for i in range(N_HEALTHY):
            age0 = rng.uniform(20, 88)
            sex = ["female", "male"][i % 2]
            for ses in range(1 + (i % 4 == 0)):
                age = float(np.floor(age0 + 2 * ses))
                healthy.append(_row(i, split, dbs[i % len(dbs)], ses, age, age + rng.normal(0, NOISE_SD),
                                    sex if not (ses == 1 and i % 8 == 0) else None, 1,   # sex filled from ses-1
                                    "Cognitively Normal", "Healthy"))
        i = 10_000
        for neuro, psych, n, effect in PATIENTS:
            for _ in range(n):
                age = float(np.floor(rng.uniform(45, 88)))
                clinical.append(_row(i, split, dbs[i % len(dbs)], 0, age, age + effect + rng.normal(0, NOISE_SD),
                                     ["female", "male"][i % 2], 0, neuro, psych))
                i += 1
        if healthy_group:
            healthy[0]["psychiatric_diagnosis"] = "Major Depressive Disorder"
        for source, rows in (("healthy_file", healthy), ("prediction_file", clinical)):
            pd.DataFrame(rows).to_csv(d / aio.FILE_TEMPLATES[source].format(split=split, tag=TAG), sep="\t",
                                      index=False)
    return d


@pytest.fixture(scope="module")
def fig13_out(tmp_path_factory):
    d = build_figure13_dir(tmp_path_factory.mktemp("fig13_pred"))
    out = tmp_path_factory.mktemp("fig13_out")
    assert f13.main(["--pred-dir", str(d), "--out-dir", str(out)]) == 0
    return d, out


def test_outputs_written(fig13_out):
    _, out = fig13_out
    for name in ("fig13_forest.pdf", "fig13_forest.png", "figure13_main_effects.csv", "figure13_main_effects.md",
                 "figure13_all_terms.csv", "figure13_models.csv", "figure13_models.md", "figure13_groups.csv",
                 "figure13_legend.md", "checks.md"):
        assert (out / name).stat().st_size > 0, name


def test_planted_effects_recovered_at_mean_age(fig13_out):
    _, out = fig13_out
    me = pd.read_csv(out / "figure13_main_effects.csv").set_index(["split", "group"])
    for split in ("test", "validation"):
        assert abs(me.loc[(split, "AD"), "coef"] - 4.0) < 0.8
        assert abs(me.loc[(split, "MCI"), "coef"] - 2.0) < 0.8
        assert me.loc[(split, "AD"), "ci_low"] < 4.0 < me.loc[(split, "AD"), "ci_high"] + 0.5
        assert me.loc[(split, "AD"), "significant"]
        assert me.loc[(split, "AD"), "n scans"] == 110 + 15          # AD alone + AD with depression
        assert me.loc[(split, "Depression"), "n scans"] == 80 + 15
    # the interaction between co-occurring groups is in the model; every Mutation_PD scan is also
    # Parkinson here, so Parkinson:Mutation_PD equals Mutation_PD and is left out (reported)
    terms = pd.read_csv(out / "figure13_all_terms.csv")
    test_terms = set(terms.loc[terms["split"] == "test", "term"])
    assert {"AD:Depression", "AD:age_c", "MCI:age_c", "Mutation_PD"} <= test_terms
    assert "Parkinson:Mutation_PD" not in test_terms
    models = pd.read_csv(out / "figure13_models.csv").set_index("split")
    assert models.loc["test", "terms left out (collinear)"] == "Parkinson:Mutation_PD"


def test_r10_merge_and_unmodelled_groups(fig13_out):
    _, out = fig13_out
    g = pd.read_csv(out / "figure13_groups.csv").set_index(["split", "group"])
    merged = g.columns[g.columns.str.startswith("merged into")][0]
    assert g.loc[("test", "Bipolar"), merged] == "Other_psych_disease"
    assert g.loc[("test", "Schizophrenia"), merged] == "Other_psych_disease"
    assert bool(g.loc[("test", "Other_psych_disease"), "modelled"])
    assert g.loc[("test", "Other_psych_disease"), "scans in model"] == 18 + 16
    assert not bool(g.loc[("test", "Other_psych_symptom"), "modelled"])     # 12 scans, never merged
    me = pd.read_csv(out / "figure13_main_effects.csv")
    assert "Bipolar" not in set(me["group"]) and "Other_psych_symptom" not in set(me["group"])


def test_reference_is_healthy_and_every_scan_accounted(fig13_out):
    _, out = fig13_out
    text = (out / "checks.md").read_text()
    acc = pd.read_csv(out / "figure13_models.csv").set_index("split")
    n_healthy = N_HEALTHY + N_HEALTHY // 4
    n_patients = sum(n for _, _, n, _ in PATIENTS)
    for split in ("test", "validation"):
        assert acc.loc[split, "N reference (healthy)"] == n_healthy
        assert acc.loc[split, "N patients"] == n_patients - 12 - 14         # hallucinations, unclassified out
        assert acc.loc[split, "N scans"] == n_healthy + n_patients - 26
    assert "| excluded: patient without group (Unclassified_patient) | 14 | 14 |" in text
    assert "| excluded: patient whose groups are not modelled | 12 | 12 |" in text
    assert "| excluded: no sex | 0 | 0 |" in text                              # filled from ses-1
    assert "given the sex of the same participant's other scans" in text


def test_age_centred_on_model_sample_and_r2(fig13_out):
    d, out = fig13_out
    models = pd.read_csv(out / "figure13_models.csv").set_index("split")
    for split in ("test", "validation"):
        frames = [aio.read_tsv(d / aio.FILE_TEMPLATES[s].format(split=split, tag=TAG)) for s in aio.FILE_TEMPLATES]
        df = pd.concat(frames)
        keep = (df["healthy"] == 1) | ~df["psychiatric_diagnosis"].isin(["Hallucinations"]) & \
            ~df["neurological_diagnosis"].isin(["Undetermined"])
        assert models.loc[split, "age centre (years)"] == pytest.approx(np.floor(df.loc[keep, "age"]).mean(), abs=0.01)
        r2 = models.loc[split]
        assert 0 < r2["R2 age + sex"] < r2["R2"] < 1
        assert r2["adjusted R2"] < r2["R2"]
        assert r2["delta R2 (diagnosis terms)"] == pytest.approx(r2["R2"] - r2["R2 age + sex"])
        assert r2["standard errors"] == "HC3"


def test_cluster_se_changes_se_not_coef(fig13_out, tmp_path):
    d, out = fig13_out
    assert f13.main(["--pred-dir", str(d), "--out-dir", str(tmp_path), "--se", "cluster"]) == 0
    a = pd.read_csv(out / "figure13_all_terms.csv")
    b = pd.read_csv(tmp_path / "figure13_all_terms.csv")
    assert np.allclose(a["coef"], b["coef"])
    assert not np.allclose(a["se"], b["se"])


def test_legend_reports_models(fig13_out):
    _, out = fig13_out
    legend = (out / "figure13_legend.md").read_text()
    models = pd.read_csv(out / "figure13_models.csv").set_index("split")
    assert f"R² = {models.loc['validation', 'R2']:.3f}" in legend
    assert "HC3" in legend and "healthy participants" in legend


def test_healthy_scan_with_group_stops(tmp_path, capsys):
    d = build_figure13_dir(tmp_path, healthy_group=True)
    assert f13.main(["--pred-dir", str(d), "--out-dir", str(tmp_path / "out")]) == 2
    assert "healthy (healthy == 1) scans have a diagnosis group" in capsys.readouterr().err
    assert f13.main(["--pred-dir", str(d), "--out-dir", str(tmp_path / "out2"), "--no-strict"]) == 0
    assert "ERROR (reported only" in (tmp_path / "out2" / "checks.md").read_text()


def test_collinear_groups_left_out_and_reported(tmp_path):
    # in the generic fixture, Parkinson and Mutation_PD (and Depression and Anxiety) have the same scans
    d = build_pred_dir(tmp_path)
    out = tmp_path / "out"
    assert f13.main(["--pred-dir", str(d), "--out-dir", str(out), "--min-scans", "5", "--min-cooccurrence", "3"]) == 0
    g = pd.read_csv(out / "figure13_groups.csv").set_index(["split", "group"])
    assert not bool(g.loc[("test", "Mutation_PD"), "modelled"])
    assert "collinear" in g.loc[("test", "Mutation_PD"), "note"]
    models = pd.read_csv(out / "figure13_models.csv")
    assert models["terms left out (collinear)"].str.contains("Mutation_PD").all()


def test_missing_files(tmp_path):
    assert f13.main(["--pred-dir", str(tmp_path), "--out-dir", str(tmp_path / "out")]) == 2
