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


def build_pred_dir(d: Path, modify=None, encoding: str = "utf-8") -> Path:
    """Write the synthetic tables; ``modify(split, source, df) -> df`` injects a fault."""
    rng = np.random.default_rng(0)
    for split, dbs in (("test", ["adni", "ukb", "hcp"]), ("validation", ["camcan", "oasis3"])):
        healthy = pd.DataFrame(_rows(rng, split, dbs, 120, True, 0))
        clinical = pd.DataFrame(_rows(rng, split, dbs, 60, False, 10_000))
        clinical = pd.concat([clinical, healthy.iloc[:9]])          # identical rows in both files
        for source, df in (("healthy_file", healthy), ("prediction_file", clinical)):
            if modify is not None:
                df = modify(split, source, df.copy())
            path = d / aio.FILE_TEMPLATES[source].format(split=split, tag=TAG)
            df.to_csv(path, sep="\t", index=False, encoding=encoding)
    return d


@pytest.fixture(scope="module")
def pred_dir(tmp_path_factory):
    return build_pred_dir(tmp_path_factory.mktemp("predictions"))


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
    filled_aug = aug[aug["N rows"] > 0]
    assert set(filled_aug["level"]) == {"native", "contains Flip", "contains Noise", "all augmented rows",
                                        "all rows (native + augmented)"}
    n_aug = aug.set_index("level")["N rows"]
    assert n_aug["contains Flip"] + n_aug["contains Noise"] == n_aug["all augmented rows"]
    assert n_aug["native"] + n_aug["all augmented rows"] == n_aug["all rows (native + augmented)"]
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


# ---------------------------------------------------------------- no scan lost to labelling or encoding
def _stat(out: Path, group: str, statistic: str) -> float:
    stats = pd.read_csv(out / "brainage_stats.csv")
    return stats.query("group == @group and statistic == @statistic")["estimate"].item()


def _run(pred_dir: Path, out: Path, *extra) -> int:
    return fig.main(["--pred-dir", str(pred_dir), "--out-dir", str(out), "--n-boot", "50", *extra])


def _first_native(df, database=None, last=False):
    """Index of a healthy native row (``last``: one with no copy in the prediction file)."""
    m = (df["augmentation"] == "native") & (df["healthy"] == 1)
    if database:
        m &= df["database"] == database
    return df.index[m][-1 if last else 0]


def test_accounting_every_scan(pred_dir, tmp_path):
    assert _run(pred_dir, tmp_path) == 0
    acc = pd.read_csv(tmp_path / "scan_accounting.csv").set_index("split")
    assert (acc["scans in files"] == acc["native original scans"]).all()
    assert (acc["healthy == 1"] + acc["healthy == 0"] == acc["native original scans"]).all()
    assert _stat(tmp_path, "Test", "N scans") == acc.loc["test", "healthy, with prediction and age"]
    ext = acc.loc["validation", "healthy, with prediction and age"]
    assert _stat(tmp_path, "External validation", "N scans") == ext
    assert _stat(tmp_path, "Cam-CAN", "N scans") + _stat(tmp_path, "OASIS-3", "N scans") == ext


def test_numeric_ids_keep_leading_zeros(tmp_path):
    def modify(split, source, df):
        if split == "test":                     # two different adni participants: "007" and "7"
            df.loc[df["subject_id"] == "sub-00000", "subject_id"] = "007"
            df.loc[df["subject_id"] == "sub-00003", "subject_id"] = "7"
        return df
    out = tmp_path / "out"
    assert _run(build_pred_dir(tmp_path, modify), out) == 0
    assert _stat(out, "Test", "N participants") == 120


@pytest.mark.parametrize("bad_value", [" native", "Native", "NATIVE", np.nan, ""])
def test_native_label_variant_stops(tmp_path, capsys, bad_value):
    def modify(split, source, df):
        if split == "test" and source == "healthy_file":
            df.loc[_first_native(df, last=True), "augmentation"] = bad_value
        return df
    d = build_pred_dir(tmp_path, modify)
    assert _run(d, tmp_path / "out") == 2
    err = capsys.readouterr().err
    assert "no native original row" in err                                # the scan would be lost
    assert ("instead of exactly 'native'" in err) or ("have no label" in err)   # and the cause is named
    assert _run(d, tmp_path / "out2", "--no-strict") == 0               # reported, not hidden
    assert "ERROR (reported only" in (tmp_path / "out2" / "checks.md").read_text()


@pytest.mark.parametrize("bad_value", ["yes", np.nan])
def test_is_original_not_boolean_stops(tmp_path, capsys, bad_value):
    def modify(split, source, df):
        if split == "validation" and source == "healthy_file":
            df["is_original"] = df["is_original"].astype(object)
            df.loc[_first_native(df), "is_original"] = bad_value
        return df
    assert _run(build_pred_dir(tmp_path, modify), tmp_path / "out") == 2
    assert "is_original" in capsys.readouterr().err


@pytest.mark.parametrize("encode", [lambda v: v.astype(float), lambda v: v.astype(bool)])
def test_healthy_and_original_numeric_encodings_lose_nobody(pred_dir, tmp_path, encode):
    def modify(split, source, df):
        df["healthy"] = encode(df["healthy"])
        df["is_original"] = df["is_original"].astype(int)
        return df
    out = tmp_path / "out"
    assert _run(build_pred_dir(tmp_path, modify), out) == 0
    assert _run(pred_dir, tmp_path / "ref") == 0
    for g in ("Test", "External validation", "Cam-CAN", "OASIS-3"):
        assert _stat(out, g, "N scans") == _stat(tmp_path / "ref", g, "N scans")


@pytest.mark.parametrize("bad_value", ["yes", np.nan, 2])
def test_healthy_unknown_stops(tmp_path, capsys, bad_value):
    def modify(split, source, df):
        if split == "test" and source == "prediction_file":
            df["healthy"] = df["healthy"].astype(object)
            df.loc[df.index[df["augmentation"] == "native"][-1], "healthy"] = bad_value
        return df
    assert _run(build_pred_dir(tmp_path, modify), tmp_path / "out") == 2
    assert "healthy" in capsys.readouterr().err


@pytest.mark.parametrize("label", ["CamCAN", "camcan ", "Cam-CAN"])
def test_external_label_variant_stops(tmp_path, capsys, label):
    def modify(split, source, df):
        if split == "validation" and source == "healthy_file":
            subj = df.loc[_first_native(df, "camcan"), "subject_id"]
            df.loc[df["subject_id"] == subj, "database"] = label     # all rows of one participant
        return df
    assert _run(build_pred_dir(tmp_path, modify), tmp_path / "out") == 2
    assert repr(label) in capsys.readouterr().err


def test_age_with_decimal_comma_stops(tmp_path, capsys):
    def modify(split, source, df):
        if split == "test" and source == "healthy_file":
            df["age"] = df["age"].astype(object)
            df.loc[_first_native(df), "age"] = "45,3"
        return df
    d = build_pred_dir(tmp_path, modify)
    assert _run(d, tmp_path / "out") == 2
    assert "'45,3'" in capsys.readouterr().err
    assert _run(d, tmp_path / "out2", "--no-strict") == 0
    assert "without a numeric prediction or age" in (tmp_path / "out2" / "checks.md").read_text()


def test_byte_order_mark_is_accepted(pred_dir, tmp_path):
    out = tmp_path / "out"
    assert _run(build_pred_dir(tmp_path, encoding="utf-8-sig"), out) == 0
    assert _run(pred_dir, tmp_path / "ref") == 0
    assert _stat(out, "Test", "N scans") == _stat(tmp_path / "ref", "Test", "N scans")


def test_latin1_file_stops_with_clear_message(tmp_path, capsys):
    def modify(split, source, df):
        if source == "prediction_file":
            df.loc[df.index[0], "neurological_diagnosis"] = "Démence"
        return df
    assert _run(build_pred_dir(tmp_path, modify, encoding="latin-1"), tmp_path / "out") == 2
    assert "not UTF-8" in capsys.readouterr().err


def test_sex_variants_are_kept_and_reported(pred_dir, tmp_path):
    def modify(split, source, df):
        df["sex"] = df["sex"].replace({"female": "F", "male": " Male "})
        return df
    out = tmp_path / "out"
    assert _run(build_pred_dir(tmp_path, modify), out) == 0
    assert _run(pred_dir, tmp_path / "ref") == 0
    assert _stat(out, "Test", "N scans") == _stat(tmp_path / "ref", "Test", "N scans")
    assert "'sex'" in (out / "checks.md").read_text()


def test_inventory_reports_typographic_apostrophe(tmp_path, capsys):
    def modify(split, source, df):
        if source == "prediction_file":
            df["neurological_diagnosis"] = df["neurological_diagnosis"].replace(
                {"Alzheimer's Disease": "Parkinson\u2019s Disease"})
        return df
    d = build_pred_dir(tmp_path, modify)
    assert inv.main(["--pred-dir", str(d), "--out-dir", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert "Parkinson\u2019s Disease" in printed and "not mapped" in printed
    tok = pd.read_csv(tmp_path / "inventory" / "diagnosis_tokens.csv", keep_default_na=False)
    assert tok.set_index("token").loc["Parkinson\u2019s Disease", "rule"] == "NOT MAPPED (ignored)"
    md = (tmp_path / "inventory" / "inventory.md").read_text()
    assert "typographic quotes" in md
    dbs = pd.read_csv(tmp_path / "inventory" / "databases.csv")
    acc = pd.read_csv(tmp_path / "inventory" / "scan_accounting.csv")
    assert dbs["scans"].sum() == acc["native original scans"].sum()            # every scan listed


# ---------------------------------------------------------------- R demographics script (skipped without R)
import shutil  # noqa: E402
import subprocess  # noqa: E402

R_SCRIPT = ANALYSIS_DIR / "demographics_tables.R"
_HAS_R = shutil.which("Rscript") is not None and subprocess.run(
    ["Rscript", "-e", "for (p in c('readr','dplyr','stringr','purrr','tidyr')) stopifnot(requireNamespace(p))"],
    capture_output=True).returncode == 0
needs_r = pytest.mark.skipif(not _HAS_R, reason="Rscript with readr/dplyr/stringr/purrr/tidyr not available")


def _run_r(pred_dir: Path, out: Path, *extra):
    return subprocess.run(["Rscript", str(R_SCRIPT), "--pred-dir", str(pred_dir), "--out-dir", str(out),
                           "--no-tables", *extra], capture_output=True, text=True)


@needs_r
def test_r_demographics_counts_match_python(pred_dir, tmp_path):
    res = _run_r(pred_dir, tmp_path)
    assert res.returncode == 0, res.stderr
    subjects = pd.read_csv(tmp_path / "subject_level.tsv", sep="\t", dtype={"subject_id": str})
    notes = []
    native = aio.native_scans(aio.read_all(pred_dir, TAG, notes), notes)
    for split in aio.SPLITS:
        n = native[native["_split"] == split]
        assert (subjects["split"] == split).sum() == aio.subject_key(n).nunique()
    single = subjects["n_sessions"] == 1
    assert (subjects.loc[single, "followup_all"] == 0).all()            # row 1: single session = 0
    assert subjects.loc[single, "followup_multi"].isna().all()          # row 2: only >= 2 sessions
    assert (subjects.loc[~single, "followup_multi"] == subjects.loc[~single, "followup_all"]).all()
    flags = pd.read_csv(tmp_path / "diagnosis_flag_counts.tsv", sep="\t").set_index("flag")
    for split in aio.SPLITS:                                            # same counts as Figure 13 n_positive
        n = native[native["_split"] == split]
        for domain, col in (("neuro", aio.COL_NEURO_DX), ("psych", aio.COL_PSYCH_DX)):
            groups = n[col].map(lambda v: dg.string_groups(None if pd.isna(v) else v, domain)[0])
            for flag in flags.index[flags.index.str.startswith(domain)]:
                expected = int(groups.map(lambda g: flag.split("_flag_")[1] in g).sum())
                assert flags.loc[flag, f"n_positive_{split}"] == expected, (flag, split)
    scans = pd.read_csv(tmp_path / "scan_level.tsv", sep="\t")
    assert len(scans) == len(native)
    assert (scans["n_samples"] == 3).all()                              # native + 2 augmented, no double count


@needs_r
def test_r_flags_match_notebook_rules(tmp_path):
    strings = [("neuro", s) for s in NEURO_DX + ["MCI; Stroke", "Other,,", "Parkinson’s Disease", ""]] + \
              [("psych", s) for s in ["Healthy", "Major Depressive Disorder; Anxiety Disorder", "surgery", "GAD|OCD", ""]]
    strings = [(d, s) for d, s in strings if s is not None]
    pd.DataFrame(strings, columns=["domain", "dx"]).to_csv(tmp_path / "dx.tsv", sep="\t", index=False)
    code = (f'suppressMessages(source("{R_SCRIPT}")); '
            f'd <- readr::read_tsv("{tmp_path / "dx.tsv"}", col_types = "cc", na = character()); '
            f'writeLines(vapply(seq_len(nrow(d)), function(i) paste(sort(dx_groups(d$dx[i], d$domain[i])), '
            f'collapse = "+"), ""), "{tmp_path / "r.txt"}")')
    res = subprocess.run(["Rscript", "-e", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    r_groups = (tmp_path / "r.txt").read_text().split("\n")[:len(strings)]
    py_groups = ["+".join(sorted(dg.string_groups(s, d)[0])) for d, s in strings]
    assert r_groups == py_groups


@needs_r
def test_r_demographics_stops_on_lost_scan(tmp_path):
    def modify(split, source, df):
        if split == "test" and source == "healthy_file":
            df.loc[_first_native(df, last=True), "augmentation"] = "Native"
        return df
    res = _run_r(build_pred_dir(tmp_path, modify), tmp_path / "out")
    assert res.returncode != 0 and "no native original row" in res.stderr
    (tmp_path / "b").mkdir()
    assert _run_r(build_pred_dir(tmp_path / "b", modify), tmp_path / "out2", "--no-strict").returncode == 0


# ---------------------------------------------------------------- longitudinal follow-up
import fig_brainage_followup as fu  # noqa: E402


def test_followup_groups_every_followed_participant(tmp_path):
    changed = {}

    def modify(split, source, df):
        if split != "validation":
            return df
        followed = df[(df["session_id"] == "ses-2") & (df["augmentation"] == "native")]
        if source == "healthy_file":            # healthy at ses-1, patient at ses-2 -> converter
            subj = followed["subject_id"].iloc[-1]          # not one of the rows copied in the prediction file
            changed["converter"] = subj
            df.loc[(df["subject_id"] == subj) & (df["session_id"] == "ses-2"), "healthy"] = 0
        else:                                   # patient at ses-1, healthy at ses-2 -> other
            subj = followed.loc[followed["healthy"] == 0, "subject_id"].iloc[0]
            changed["other"] = subj
            df.loc[(df["subject_id"] == subj) & (df["session_id"] == "ses-2"), "healthy"] = 1
        return df
    d = build_pred_dir(tmp_path, modify)
    out = tmp_path / "out"
    assert fu.main(["--pred-dir", str(d), "--out-dir", str(out), "--n-boot", "50"]) == 0
    part = pd.read_csv(out / "brainage_followup_participants.csv")
    notes = []
    native = aio.native_scans(aio.read_all(d, TAG, notes), notes)
    ext = native[native["_split"] == "validation"]
    sessions = ext.groupby(aio.subject_key(ext))["session_id"].nunique()
    assert len(part) == int((sessions >= 2).sum())                       # nobody followed is lost
    assert part["participant"].is_unique
    group = part.set_index("participant")["group"]
    db = ext.drop_duplicates("subject_id").set_index("subject_id")["database"]
    assert group[f"{db[changed['converter']]}/{changed['converter']}"] == "Converter"
    assert group[f"{db[changed['other']]}/{changed['other']}"] == "Other"
    assert set(group) == {"Stable healthy", "Converter", "Patient", "Other"}
    assert (part["follow-up (years)"] == 2).all() and part["rate (predicted years per year)"].notna().all()
    stats = pd.read_csv(out / "brainage_followup_stats.csv")
    allx = stats[stats["database"] == "All external"]
    assert allx["participants"].sum() == len(part)
    for name in ("fig_brainage_followup.pdf", "fig_brainage_followup.png", "brainage_followup_text.md"):
        assert (out / name).stat().st_size > 0
    curves = pd.read_csv(out / "brainage_followup_mean_curves.csv")
    start = curves[curves["years since first scan"] == 0].set_index("group")
    assert (start["mean change in predicted age"] == 0).all()             # every course starts at 0
    for g, n in part["group"].value_counts().items():
        assert start.loc[g, "participants"] == n                          # every participant in the curve
    later = curves[curves["years since first scan"] == 2].set_index("group")
    for g in later.index:
        sub = part[part["group"] == g]
        expected = (sub["last gap"] - sub["baseline gap"] + 2).mean()     # change in predicted age at 2 years
        assert later.loc[g, "mean change in predicted age"] == pytest.approx(expected)


def test_followup_without_longitudinal_data(tmp_path, capsys):
    def modify(split, source, df):
        return df[df["session_id"] == "ses-1"]
    assert fu.main(["--pred-dir", str(build_pred_dir(tmp_path, modify)), "--out-dir", str(tmp_path / "o")]) == 1
    assert "no participant with at least two sessions" in capsys.readouterr().err


def test_conflicting_rows_for_one_scan_stop(tmp_path, capsys):
    def modify(split, source, df):
        if split == "test" and source == "prediction_file":   # copy of a healthy scan relabelled as patient
            df.loc[df.index[-9:], "healthy"] = 0
        return df
    assert _run(build_pred_dir(tmp_path, modify), tmp_path / "out") == 2
    err = capsys.readouterr().err
    assert "several native original rows" in err and "healthy" in err


def test_inventory_keeps_missing_database_label_and_training(tmp_path):
    def modify(split, source, df):
        if split == "test" and source == "prediction_file":
            df.loc[df["subject_id"] == df["subject_id"].iloc[0], "database"] = np.nan
        return df
    d = build_pred_dir(tmp_path, modify)
    rng = np.random.default_rng(1)
    pd.DataFrame(_rows(rng, "train", ["adni", "ppmi"], 30, True, 40_000)).to_csv(
        d / aio.TRAIN_FILE_TEMPLATE.format(tag=TAG), sep="\t", index=False)
    out = tmp_path / "o"
    assert inv.main(["--pred-dir", str(d), "--out-dir", str(out)]) == 0
    dbs = pd.read_csv(out / "inventory" / "databases.csv", keep_default_na=False)
    assert "<missing>" in set(dbs["database"])                                   # not dropped by groupby
    assert {"train", "test", "validation"} == set(dbs["split"])
    assert "ppmi" in set(dbs.loc[dbs["split"] == "train", "database"])            # training-only database listed
    acc = pd.read_csv(out / "inventory" / "scan_accounting.csv")
    assert dbs["scans"].sum() == acc["native original scans"].sum()
    assert inv.main(["--pred-dir", str(d), "--out-dir", str(tmp_path / "o2"), "--no-train"]) == 0
    dbs2 = pd.read_csv(tmp_path / "o2" / "inventory" / "databases.csv", keep_default_na=False)
    assert "train" not in set(dbs2["split"])


def test_checks_md_does_not_claim_success_with_no_strict(tmp_path):
    def modify(split, source, df):
        if split == "test" and source == "healthy_file":
            df.loc[_first_native(df, last=True), "augmentation"] = "Native"
        return df
    out = tmp_path / "out"
    assert _run(build_pred_dir(tmp_path, modify), out, "--no-strict") == 0
    text = (out / "checks.md").read_text()
    assert "PROBLEMS FOUND" in text and "Every scan in the files has one native" not in text


@needs_r
def test_r_demographics_sex_apoe_followup_and_duplicates(tmp_path):
    def scan(sub, ses, age, healthy, sex, apoe, dx_p="Healthy"):
        return [dict(subject_id=sub, session_id=ses, database="oasis3", split="validation", augmentation=a,
                     is_original=a == "native", age=age, predicted_age_years=age + 1.5, sex=sex, healthy=healthy,
                     apoe=apoe, neurological_diagnosis="Cognitively Normal", psychiatric_diagnosis=dx_p)
                for a in ("native", "RandomFlip", "RandomNoise")]
    hf = pd.DataFrame(scan("A", "ses-1", 60, 1, "Female", "0") + scan("B", "ses-1", 70, 1, "male", "-1.0")
                      + scan("B", "ses-2", 73, 1, None, "-1.0") + scan("C", "ses-1", 65, 1, "female", "Unknown")
                      + scan("D", "ses-1", 66, 1, "Male", "1") + scan("E", "ses-1", 67, 1, None, None))
    pf = pd.concat([hf[hf["subject_id"] == "B"],                         # B also in the clinical file
                    pd.DataFrame(scan("P", "ses-1", 75, 0, "female", "-1",
                                      "Major Depressive Disorder; Anxiety Disorder"))])
    for name, df in (("predictions_validation", hf), ("predictions_prediction-validation", pf)):
        df.to_csv(tmp_path / f"{name}_{TAG}.tsv", sep="\t", index=False)
    res = _run_r(tmp_path, tmp_path / "out")
    assert res.returncode == 0, res.stderr
    s = pd.read_csv(tmp_path / "out" / "subject_level.tsv", sep="\t").set_index("subject_id")
    assert s.loc["B", "n_samples"] == 6                                  # not 12: counted once
    assert (s.loc["A", "followup_all"], s.loc["B", "followup_all"]) == (0, 3)
    assert pd.isna(s.loc["A", "followup_multi"]) and s.loc["B", "followup_multi"] == 3
    assert s["sex"].to_dict() == {"A": "Female", "B": "Male", "C": "Female", "D": "Male", "E": "Unknown",
                                  "P": "Female"}
    assert s["apoe"].to_dict() == {"A": "Non-carrier", "B": "Unknown", "C": "Unknown", "D": "Carrier",
                                   "E": "Unknown", "P": "Unknown"}
    checks = (tmp_path / "out" / "demographics_checks.txt").read_text()
    assert "given the sex of the same participant" in checks
    counts = pd.read_csv(tmp_path / "out" / "split_counts.tsv", sep="\t").set_index("split").loc["validation"]
    assert (counts["scans"], counts["participants"]) == (7, 6)
    assert (counts["CN participants (healthy == 1)"], counts["CN scans (healthy == 1)"]) == (5, 6)   # people, not rows
    f = pd.read_csv(tmp_path / "out" / "diagnosis_flag_counts.tsv", sep="\t").set_index("flag")
    assert f.loc["psych_flag_MDD", "n_positive_validation"] == 1 == f.loc["psych_flag_Anxiety", "n_positive_validation"]


def _decimal_ages(split, source, df):
    df["age"] = df["age"] + 0.7                                         # e.g. 70.7 must be analysed as 70
    return df


def test_inventory_floors_age(tmp_path):
    (tmp_path / "pred").mkdir()
    d = build_pred_dir(tmp_path / "pred", modify=_decimal_ages)
    assert inv.main(["--pred-dir", str(d), "--out-dir", str(tmp_path / "o")]) == 0
    dbs = pd.read_csv(tmp_path / "o" / "inventory" / "databases.csv")
    for col in ("age min", "age max"):
        assert (dbs[col] == np.floor(dbs[col])).all(), col


@needs_r
def test_r_demographics_floors_age(tmp_path):
    (tmp_path / "pred").mkdir()
    d = build_pred_dir(tmp_path / "pred", modify=_decimal_ages)
    res = _run_r(d, tmp_path / "o")
    assert res.returncode == 0, res.stderr
    for name in ("scan_level.tsv", "subject_level.tsv"):
        age = pd.read_csv(tmp_path / "o" / name, sep="\t")["age"]
        assert (age == np.floor(age)).all(), name
    subjects = pd.read_csv(tmp_path / "o" / "subject_level.tsv", sep="\t")
    multi = subjects["followup_multi"].dropna()
    assert len(multi) and (multi == np.floor(multi)).all()              # follow-up from floored ages


def _repeated_flip(split, source, df):
    """Two identical flipped copies of one scan in the healthy file (same text, same prediction)."""
    if source == "healthy_file":
        row = df[(df["augmentation"] == "flip")].iloc[[0]]
        df = pd.concat([df, row], ignore_index=True)
    return df


def test_identical_copies_within_a_file_are_kept(tmp_path):
    (tmp_path / "pred").mkdir()
    d = build_pred_dir(tmp_path / "pred", modify=_repeated_flip)
    raw = aio.read_all(d, TAG, [])
    kept = aio.drop_cross_file_copies(raw)
    assert len(raw) - len(kept) == 2 * 9                                # 9 healthy rows copied per split
    assert (kept["_source"] == "prediction_file").sum() == (raw["_source"] == "prediction_file").sum() - 18
    for split in aio.SPLITS:
        r = kept[(kept["_split"] == split) & (kept["augmentation"] == "flip")]
        assert r.drop(columns=["_source", "_split"]).duplicated(keep=False).sum() == 2   # both copies kept


@needs_r
def test_r_counts_identical_copies_within_a_file(tmp_path):
    (tmp_path / "pred").mkdir()
    d = build_pred_dir(tmp_path / "pred", modify=_repeated_flip)
    res = _run_r(d, tmp_path / "o")
    assert res.returncode == 0, res.stderr
    scans = pd.read_csv(tmp_path / "o" / "scan_level.tsv", sep="\t", dtype={"subject_id": str})
    assert scans["n_samples"].value_counts().to_dict() == {3: len(scans) - 2, 4: 2}   # one scan per split
    assert (scans.loc[scans["n_samples"] == 4, "aug_Flip"] == 2).all()
    assert (scans.loc[scans["n_samples"] == 3, "aug_Flip"] == 1).all()


def test_scatter_and_followup_floor_decimal_ages(tmp_path):
    """Ages stored with decimals give whole-year ages in every output (floored, never rounded)."""
    (tmp_path / "pred").mkdir()
    d = build_pred_dir(tmp_path / "pred", modify=_decimal_ages)          # e.g. 70.7 -> 70
    (tmp_path / "ref").mkdir()
    ref = build_pred_dir(tmp_path / "ref")                              # same tables, whole-year ages
    out, out_ref = tmp_path / "o", tmp_path / "o_ref"
    assert _run(d, out) == 0
    assert _run(ref, out_ref) == 0
    a = pd.read_csv(out / "brainage_stats.csv")
    b = pd.read_csv(out_ref / "brainage_stats.csv")
    pd.testing.assert_series_equal(a["estimate"], b["estimate"])         # same results as whole-year ages
    for stat in ("Chronological age, min (years)", "Chronological age, max (years)"):
        v = a.loc[a["statistic"] == stat, "estimate"]
        assert (v == np.floor(v)).all()
    assert fu.main(["--pred-dir", str(d), "--out-dir", str(tmp_path / "fu"), "--n-boot", "50"]) == 0
    part = pd.read_csv(tmp_path / "fu" / "brainage_followup_participants.csv")
    for col in ("baseline age", "last age", "follow-up (years)"):
        assert (part[col] == np.floor(part[col])).all(), col


# ---------------------------------------------------------------- proposed diagnosis classification

import diagnosis_classification as dc  # noqa: E402

CLASSIFICATION_CASES = [
    # (neurological cell, psychiatric cell, expected groups)
    ("Cognitively Normal", "Healthy", set()),
    ("Alzheimer's Disease", None, {"AD"}),
    ("Dementia", None, {"Dementia_unspecified"}),                                     # R6, no cause
    ("Dementia, Amnestic, Multiple Domain, Traumatic Brain Injury", None, {"TBI"}),     # R6, cause written
    ("Dementia, Other Cognitive Disorder", None, {"Other_neuro_disease"}),              # R6, other cause
    ("Medical Illness-related, Amnestic, Multiple Domain, Dementia", None, {"Other_neuro_disease"}),
    ("Medical Illness-related, Undetermined", None, {"Other_neuro_disease"}),          # R4, alone
    ("MCI, Amnestic, Alcohol-related", None, {"MCI"}),                                  # R4, descriptor
    ("Other Cognitive Disorder", None, {"Other_neuro_disease"}),
    ("Post-Stroke, Aphasia, Broca", None, {"Vascular"}),                                # R5 + hyphen
    ("Hyposmia", None, {"Other_neuro_symptom"}),
    ("Prodromal Synucleinopathy, Hyposmia", None, {"Prodromal_PD"}),
    ("Parkinsonism", None, {"Movement_disorder"}),                                      # R8
    ("MCI, Amnestic, Multiple Domain, Parkinsonism", None, {"MCI", "Movement_disorder"}),
    ("Dementia, Dementia with Lewy Bodies, Parkinsonism", None, {"Atypical_parkinsonism"}),
    ("SWEDD", None, {"Movement_disorder"}),
    ("Hyposmia, Restless Leg Syndrome", None, {"Movement_disorder"}),
    ("Primary Progressive Aphasia", None, {"MCI"}),                                     # R7
    ("Primary Progressive Aphasia, Semantic, FrontoTemporal Dementia", None, {"FTD"}),
    ("Progressive Supranuclear Palsy", None, {"Atypical_parkinsonism"}),
    ("MCI, Amnestic, Major Depressive Disorder", "Healthy", {"MCI", "Depression"}),    # R2
    ("Major Depressive Disorder OR Medication-related OR Insomnia", None, {"Other_neuro_symptom"}),
    ("Cognitively Normal", "Dysthymia", {"Depression"}),
    ("Cognitively Normal", "Sleep Apnea, Insomnia", {"Sleep"}),
    ("Cognitively Normal", "Schizophrenia, Hallucinations", {"Schizophrenia"}),
    ("Cognitively Normal", "Hallucinations", {"Other_psych_symptom"}),
    ("Cognitively Normal", "Major Depressive Disorder, Treated", {"Depression"}),
    ("Cognitively Normal", "Other Psychiatric Disorder", {"Other_psych_disease"}),
    ("LRRK2; GBA", "Unknown", {"Mutation_PD"}),                                         # R1
    ("PARK7; PINK1", None, {"Mutation_PD"}),
    ("Juvenile Autosomal Recessive Parkinsonism", None, {"Parkinson"}),                 # one term, not R8
    ("REM Sleep Behavior Disorder, Hyposmia", None, {"Prodromal_PD"}),                  # R5: symptom dropped
    ("Primary Autonomic Failure", None, {"Prodromal_PD"}),
    ("Multiple System Atrophy, Parkinsonism", None, {"Atypical_parkinsonism"}),         # R8, R9
    ("Fronto-temporal Dementia", None, {"FTD"}),
    ("Dementia, Toxic / Metabolic Disorder", None, {"Other_neuro_disease"}),            # R6, attribution
    ("Dementia, Undetermined etiology", None, {"Dementia_unspecified"}),                # R6, no cause
    ("Aphasia, TranscorticalMotor, Post-Surgical", None, {"Other_neuro_disease"}),      # R4, R5
    ("Motor Neuron Disease", "anxiety", {"Other_neuro_disease", "Anxiety"}),
    ("Bell's Palsy; Muscular Dystrophy", "deferred", {"Other_neuro_disease"}),
    ("Unknown", "Undetermined", set()),
    (None, None, set()),
]


@pytest.mark.parametrize("neuro, psych, expected", CLASSIFICATION_CASES)
def test_proposed_classification_rules(neuro, psych, expected):
    assert set(dc.scan_groups(neuro, psych)) == expected


def test_terms_table_and_unmapped_terms():
    terms = dc.load_terms()
    assert terms["key"].is_unique
    assert dc.term_info("  parkinsonism ")[0] == "MOVEMENT_SYNDROME"       # case and spaces ignored
    flags, cn, unmapped = dc.classify_cell("Cognitively Normal, Made-up Disease")
    assert cn and unmapped == ["Made-up Disease"] and not flags[dc.NEURO]
    assert dc.rare_groups({"AD": 29, "MCI": 30, "Other_neuro_disease": 3, "Sleep": 0, "OCD": 2}) == \
        {"AD": "Other_neuro_disease", "OCD": "Other_psych_disease"}


def test_merge_and_single_label_helpers():
    scans = [frozenset({"AD"}), frozenset({"AD", "OCD"}), frozenset(), frozenset(), frozenset({"FTD"})]
    assert dc.group_counts(scans) == {"AD": 2, "OCD": 1, "FTD": 1}
    mapping = dc.rare_groups({"AD": 40, "OCD": 1, "FTD": 2})
    assert dc.merge_rare(scans[1], mapping) == {"AD", "Other_psych_disease"}
    assert dc.merge_rare(scans[4], mapping) == {"Other_neuro_disease"}
    labels = [dc.single_label(s, h) for s, h in zip(scans, [False, False, True, False, False])]
    assert labels == ["AD", "Mixed", "CN", "Unclassified_patient", "FTD"]
    flags = dc.group_flags(scans, ["AD", "OCD"])
    assert flags.columns.tolist() == ["group_AD", "group_OCD"]
    assert flags["group_AD"].tolist() == [True, True, False, False, False]


def _tricky_diagnoses(split, source, df):
    """Patients get the classification cases above, in turn."""
    patients = df["healthy"] == 0
    cases = CLASSIFICATION_CASES * (patients.sum() // len(CLASSIFICATION_CASES) + 1)
    df.loc[patients, "neurological_diagnosis"] = [c[0] for c in cases[:patients.sum()]]
    df.loc[patients, "psychiatric_diagnosis"] = [c[1] for c in cases[:patients.sum()]]
    return df


def test_inventory_correspondence_and_proposed_groups(tmp_path):
    (tmp_path / "pred").mkdir()
    d = build_pred_dir(tmp_path / "pred", modify=_tricky_diagnoses)
    assert inv.main(["--pred-dir", str(d), "--out-dir", str(tmp_path / "o")]) == 0
    corr = pd.read_csv(tmp_path / "o" / "inventory" / "diagnosis_correspondence.csv", keep_default_na=False)
    assert {"term", "type", "group (proposed)", "condition", "scans test", "scans validation"} <= set(corr.columns)
    assert (corr["type"] != "UNMAPPED").all()
    assert corr.loc[(corr["term"] == "Parkinsonism"), "group (proposed)"].iloc[0] == "Movement_disorder"
    groups = pd.read_csv(tmp_path / "o" / "inventory" / "diagnosis_groups_proposed.csv", keep_default_na=False)
    raw = aio.native_scans(aio.read_all(d, TAG, []), [])
    for split in aio.SPLITS:
        r = raw[raw["_split"] == split]
        expected = {}
        for n, p in zip(r[aio.COL_NEURO_DX], r[aio.COL_PSYCH_DX]):
            for g in dc.scan_groups(None if pd.isna(n) else n, None if pd.isna(p) else p):
                expected[g] = expected.get(g, 0) + 1
        got = dict(zip(groups["group"], groups[f"scans {split}"]))
        for g, n in expected.items():
            assert got[g] == n, (split, g)
        no_group = np.array([not dc.scan_groups(None if pd.isna(n) else n, None if pd.isna(p) else p)
                             for n, p in zip(r[aio.COL_NEURO_DX], r[aio.COL_PSYCH_DX])])
        unclassified = int(((r["healthy"] == 0).to_numpy() & no_group).sum())
        assert got[dc.UNCLASSIFIED] == unclassified


@needs_r
def test_r_proposed_groups_match_python(tmp_path):
    (tmp_path / "pred").mkdir()
    d = build_pred_dir(tmp_path / "pred", modify=_tricky_diagnoses)
    res = _run_r(d, tmp_path / "o")
    assert res.returncode == 0, res.stderr
    scans = pd.read_csv(tmp_path / "o" / "scan_level.tsv", sep="\t", keep_default_na=False, na_values=["NA"])
    group_cols = [c for c in scans.columns if c.startswith("group_")]
    assert "group_Unclassified_patient" in group_cols
    for _, row in scans.iterrows():
        n = None if pd.isna(row["neurological_diagnosis"]) else row["neurological_diagnosis"]
        p = None if pd.isna(row["psychiatric_diagnosis"]) else row["psychiatric_diagnosis"]
        expected = set(dc.scan_groups(n, p))
        got = {c[len("group_"):] for c in group_cols if row[c] in (True, "TRUE")} - {dc.UNCLASSIFIED}
        assert got == expected, (n, p)
        assert (row["group_Unclassified_patient"] in (True, "TRUE")) == (not expected and row["healthy"] == 0)
    counts = pd.read_csv(tmp_path / "o" / "diagnosis_group_counts.tsv", sep="\t", keep_default_na=False)
    for split in aio.SPLITS:
        s = scans[scans["split"] == split]
        for g in group_cols:
            n = int((s[g].astype(str) == "TRUE").sum())
            name = g[len("group_"):]
            if n:
                assert int(counts.loc[counts["group"] == name, f"scans_{split}"].iloc[0]) == n, (split, name)
                merged = counts.loc[counts["group"] == name, f"merged_into_{split}"].iloc[0]
                assert merged == (dc.rare_groups({name: n}).get(name, "")), (split, name)


def _synthetic_followup(seed=1):
    """Courses with known rates: stable healthy 0.9; patients 0.9 + 0.3; converters 0.9 before their
    first patient session and 0.9 + 1.0 after it."""
    rng = np.random.default_rng(seed)
    rows = []

    def add(kind, n, times, rate_fn, status_fn, groups):
        for i in range(n):
            age0, a = float(rng.integers(60, 80)), rng.normal(0, 3)
            k = int(rng.integers(2, len(times)))                   # converter: first patient session index
            for s, t in enumerate(times):
                rows.append({"_subject": f"oasis3/{kind}{i}", aio.COL_DATABASE: "oasis3", aio.COL_SESSION: f"ses-{s + 1}",
                             "_ses_num": s + 1, "_age": age0 + t, "_t": float(t), "_healthy": status_fn(s, k),
                             aio.COL_PRED: a + 0.9 * (age0 + t) + rate_fn(t, times[k]) + rng.normal(0, 1.5),
                             "_sex": ["female", "male"][i % 2], "_groups": frozenset() if status_fn(s, k) else
                             frozenset({"AD"} if i % 2 else {"Parkinson"})})
    add("h", 200, [0, 2, 4, 6], lambda t, tdx: 0.0, lambda s, k: 1, None)
    add("c", 90, [0, 2, 4, 6, 8], lambda t, tdx: 1.0 * max(0.0, t - tdx), lambda s, k: int(s < k), None)
    add("p", 90, [0, 2, 4, 6], lambda t, tdx: 0.3 * t, lambda s, k: 0, None)
    d = pd.DataFrame(rows)
    d["_gap"] = d[aio.COL_PRED] - d["_age"]
    d["_dpred"] = d[aio.COL_PRED] - d.groupby("_subject")[aio.COL_PRED].transform("first")
    return d


def test_followup_mixed_models_recover_known_rates():
    d = _synthetic_followup()
    part = fu.participant_table(d)
    conv = part[part["group"] == "Converter"]
    assert len(conv) == 90 and (conv["first patient session (years since first scan)"]
                                > conv["last healthy session (years since first scan)"]).all()
    assert conv["cognitive conversion"].sum() == 45                       # AD at the first patient session
    notes = []
    m = fu.model_data(d, part, notes)
    coefs, prof, ref = fu.run_models(m, notes)
    assert ref is not None

    def est(model, definition, knot, term):
        sel = coefs[(coefs["model"] == model) & (coefs["converters"] == definition) & (coefs["knot"] == knot)]
        return sel.set_index("term")["estimate"][term]
    assert est("rates", "all converters", "", "t") == pytest.approx(0.9, abs=0.1)
    assert est("rates", "all converters", "", "pat:t") == pytest.approx(0.3, abs=0.12)
    assert est("rates", "all converters", "", "conv:t") > 0.2
    for definition in ("all converters", "cognitive converters"):
        assert est("broken stick", definition, "first patient session", "conv:t") == pytest.approx(0.0, abs=0.2)
        assert est("broken stick", definition, "first patient session", "post") == pytest.approx(1.0, abs=0.3)
    off = "knot offset (years from first patient session)"
    best = prof.loc[prof.groupby("converters")["log-likelihood (ML)"].idxmax()].set_index("converters")[off]
    assert best["all converters"] == 0

    m = fu.healthy_reference(m, ref)
    assert m.loc[m["group"] == "Stable healthy", "_excess_change"].mean() == pytest.approx(0.0, abs=0.15)
    aligned = fu.align_converters(m)
    at_dx = aligned[(aligned["tau"] == 0) & (aligned["_healthy"] == 0)]
    assert at_dx["_dresid"].abs().max() < 1e-9 and at_dx["_subject"].nunique() == 90
    text = fu.models_markdown(coefs, prof, m)
    assert "Broken stick, all converters, knot at first patient session" in text and "best knot +0 years" in text


def test_followup_knot_profile_uses_one_random_structure():
    d = _synthetic_followup()
    part = fu.participant_table(d)
    notes = []
    m = fu.model_data(d, part, notes)
    _, prof, _ = fu.run_models(m, notes, min_n=10)
    off = "knot offset (years from first patient session)"
    for definition, p in prof.groupby("converters"):
        fitted = p[p["log-likelihood (ML)"].notna()]
        assert fitted["random effects"].nunique() == 1, definition
        # offsets with too few converters on both sides of the knot are not fitted
        assert (p.loc[p["log-likelihood (ML)"].isna(), "converters with scans on both sides"] < 10).all()
        assert (fitted["converters with scans on both sides"] >= 10).all()
    assert prof.loc[prof["converters"] == "all converters"].set_index(off).loc[-6, "log-likelihood (ML)"] != \
        prof.loc[prof["converters"] == "all converters"].set_index(off).loc[0, "log-likelihood (ML)"]


def test_followup_knot_before_first_scan_is_set_to_first_scan():
    d = _synthetic_followup()
    part = fu.participant_table(d)
    m = fu.model_data(d, part, [])
    kd = fu.with_knot(m, m["t_dx"] - 20)                    # every knot long before the first scan
    conv = kd[kd["conv"] == 1]
    assert (conv.loc[conv["t"] == 0, "post"] == 0).all()     # no level at the first scan from the knot
    assert np.allclose(conv["post"], conv["t"])
    assert conv["_knot_clipped"].all()
