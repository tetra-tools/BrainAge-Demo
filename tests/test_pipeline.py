import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from brainage import BrainAgePipeline, ModelBundle
from brainage.bundle import BundleError
from brainage.cli import main as cli
from brainage.inference import OnnxAgePredictor, TorchAgePredictor
from brainage.preprocessing import PreprocessingError, Preprocessor, clean_image_data


# --------------------------------------------------------------------- bundle
def test_bundle_manifest_and_checksums(bundle_dir):
    b = ModelBundle.load(bundle_dir)
    m = b.manifest
    assert m["model"]["channel_number"] == 8
    assert m["model"]["onnx_parity_max_abs_diff_years"] < 1e-2
    assert m["preprocessing"]["registration"] == "antsRegistrationSyNQuick.sh -t a"
    assert set(m["files"]) == {"model.onnx", "t1_landmarks.npy", "template.nii.gz", "brain_mask.nii"}
    assert cli(["info", "--bundle", str(bundle_dir)]) == 0


def test_tampered_bundle_is_rejected(bundle_dir, tmp_path):
    copy = tmp_path / "b"
    shutil.copytree(bundle_dir, copy)
    lm = np.load(copy / "t1_landmarks.npy")
    np.save(copy / "t1_landmarks.npy", lm + 1)
    with pytest.raises(BundleError, match="Checksum"):
        ModelBundle.load(copy)
    ModelBundle.load(copy, verify=False)  # explicit opt-out still works


def test_onnx_matches_checkpoint(assets, bundle_dir):
    torch_model = TorchAgePredictor(assets["checkpoint"])
    onnx_model = OnnxAgePredictor(ModelBundle.load(bundle_dir).model_path)
    for batch in (1, 3):  # the exported batch axis must be dynamic
        x = np.random.default_rng(batch).random((batch, 1, 160, 192, 160), dtype=np.float32) * 100
        got = onnx_model.predict(x)
        assert got.shape == (batch,)
        np.testing.assert_allclose(got, torch_model.predict(x), atol=1e-2)


# --------------------------------------------------------------- preprocessing
def test_preprocessing_is_identical_to_training_cache(assets, bundle_dir):
    """For an image already in template space, our tensor must equal what generate_ptfiles.py stored."""
    import torchio as tio

    # --- reference: copied from generate_ptfiles.py (augmentation removed) ---
    preprocess = tio.Compose([
        tio.Lambda(clean_image_data, types_to_apply=[tio.INTENSITY]),
        tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5)),
        tio.HistogramStandardization({"t1": np.load(assets["landmarks"])}),
        tio.Mask(masking_method="brain"),
        tio.CropOrPad(target_shape=(160, 192, 160), padding_mode=0),
    ], exclude=["brain"])
    brain_mask = tio.ScalarImage(assets["mask"])
    brain_mask = tio.LabelMap(tensor=(brain_mask.data > .1).float(), affine=brain_mask.affine)
    sample = preprocess(tio.Subject(t1=tio.ScalarImage(assets["in_space"]), brain=brain_mask))
    expected = sample.t1.data.to(torch.float16).unsqueeze(0).float().numpy()

    pre = Preprocessor(**ModelBundle.load(bundle_dir).preprocessor_kwargs())
    got = pre.run(assets["in_space"], skip_bias_correction=True, skip_registration=True)
    assert got.tensor.shape == (1, 1, 160, 192, 160)
    np.testing.assert_array_equal(got.tensor, expected)


def test_skip_registration_on_native_image_fails_clearly(assets, bundle_dir):
    pre = Preprocessor(**ModelBundle.load(bundle_dir).preprocessor_kwargs())
    with pytest.raises(PreprocessingError, match="template/mask grid"):
        pre.run(assets["raw"], skip_bias_correction=True, skip_registration=True)


# ------------------------------------------------------------------ end to end
@pytest.mark.slow
def test_full_pipeline_on_raw_scan(assets, bundle_dir, tmp_path):
    pipe = BrainAgePipeline(bundle_dir, providers=["CPUExecutionProvider"], num_threads=4)
    res = pipe.predict(assets["raw"], chronological_age=60, subject_id="sub-02",
                       save_preprocessed_dir=tmp_path / "pp")
    assert res.status == "ok" and np.isfinite(res.predicted_age)
    assert res.brain_age_gap == pytest.approx(res.predicted_age - 60, abs=1e-3)
    assert res.steps[:2] == ["n4_bias_correction", "registration:antsRegistrationSyNQuick.sh -t a"]

    # registration quality: the registered scan must overlap the template
    reg = nib.load(res.saved_files["registered"]).get_fdata()
    tpl = nib.load(assets["template"]).get_fdata()
    assert reg.shape == tpl.shape
    corr = np.corrcoef(reg.ravel(), tpl.ravel())[0, 1]
    assert corr > 0.85, corr

    # the saved model input is exactly what went into the network
    model_input = nib.load(res.saved_files["model_input"]).get_fdata()
    assert model_input.shape == (160, 192, 160)


@pytest.mark.slow
def test_cli_batch_with_list_and_error_isolation(assets, bundle_dir, tmp_path):
    lst = tmp_path / "subjects.tsv"
    lst.write_text("subject_id\tpath\tage\n"
                   f"sub-01\t{assets['in_space']}\t55\n"
                   f"sub-missing\t{tmp_path / 'nope.nii.gz'}\t\n")
    out = tmp_path / "results.json"
    code = cli(["predict", "--list", str(lst), "--bundle", str(bundle_dir), "--cpu",
                "--skip-registration", "--output", str(out)])
    assert code == 1  # one image failed
    res = json.loads(out.read_text())
    assert [r["status"] for r in res] == ["ok", "error"]
    assert res[0]["subject_id"] == "sub-01" and res[0]["chronological_age"] == 55

    tsv = tmp_path / "results.tsv"
    assert cli(["predict", str(assets["in_space"]), "--bundle", str(bundle_dir), "--cpu",
                "--skip-registration", "--skip-bias-correction", "--age", "40", "-o", str(tsv)]) == 0
    header, row = tsv.read_text().strip().split("\n")
    assert header.split("\t")[:3] == ["subject_id", "input_path", "predicted_age"]


# ------------------------------------------------------------ review fixes
def test_manifest_must_checksum_every_asset(bundle_dir, tmp_path):
    copy = tmp_path / "b"
    shutil.copytree(bundle_dir, copy)
    manifest = json.loads((copy / "manifest.json").read_text())
    del manifest["files"]["t1_landmarks.npy"]
    (copy / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(BundleError, match="no checksum"):
        ModelBundle.load(copy)
    manifest["files"] = {}
    (copy / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(BundleError, match="no checksum"):
        ModelBundle.load(copy, verify=False)


def test_export_rejects_non_finite_model(assets, tmp_path):
    from brainage.export import export_bundle

    ckpt = torch.load(assets["checkpoint"], weights_only=False)
    ckpt["state_dict"]["classifier.4.bias"] = torch.tensor([float("nan")])
    bad = tmp_path / "nan.ckpt"
    torch.save(ckpt, bad)
    out = tmp_path / "bundle"
    with pytest.raises(RuntimeError, match="NaN"):
        export_bundle(bad, assets["landmarks"], assets["template"], assets["mask"], out, version="0.0.0")
    assert not out.exists()  # no half-written bundle left behind


def test_unique_output_stems():
    from brainage.pipeline import unique_output_stems

    paths = ["/d/a/T1w.nii.gz", "/d/b/T1w.nii.gz", "/d/sub-01_T1w.nii.gz", "/x/a/T1w.nii", "/d/c.nii.gz"]
    stems = unique_output_stems(paths, [None, None, None, None, "sub-09"])
    assert stems == ["a_T1w_1", "b_T1w", "sub-01", "a_T1w_2", "sub-09"]
    assert len(set(stems)) == len(stems)


def test_batch_qc_outputs_do_not_collide(assets, bundle_dir, tmp_path):
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        shutil.copy(assets["in_space"], tmp_path / sub / "T1w.nii.gz")
    pipe = BrainAgePipeline(bundle_dir, providers=["CPUExecutionProvider"])
    res = pipe.predict_many([tmp_path / "a" / "T1w.nii.gz", tmp_path / "b" / "T1w.nii.gz"],
                            skip_bias_correction=True, skip_registration=True,
                            save_preprocessed_dir=tmp_path / "qc", fail_fast=True)
    files = [f for r in res for f in r.saved_files.values()]
    assert len(set(files)) == 4 and all(Path(f).is_file() for f in files)


def test_registration_command_matches_conversion_script():
    """Flags of `antsRegistrationSyNQuick.sh -d 3 -t a`, as used to build the training derivatives."""
    from brainage.preprocessing import syn_quick_affine_args

    args = syn_quick_affine_args("F", "M", "out_", "W", "IW")
    flags = dict(zip(args[::2], args[1::2]))  # last occurrence wins for repeated stage flags
    assert flags["--winsorize-image-intensities"] == "[0.005,0.995]"
    assert flags["--use-histogram-matching"] == "0"
    assert flags["--initial-moving-transform"] == "[F,M,1]"
    assert flags["--interpolation"] == "Linear"
    assert [args[i + 1] for i, a in enumerate(args) if a == "--transform"] == ["Rigid[0.1]", "Affine[0.1]"]
    assert [args[i + 1] for i, a in enumerate(args) if a == "--metric"] == ["MI[F,M,1,32,Regular,0.25]"] * 2
    assert [args[i + 1] for i, a in enumerate(args) if a == "--convergence"] == ["[1000x500x250x0,1e-6,10]"] * 2
    assert flags["--random-seed"] == "123"
    assert not any(a.startswith("SyN") for a in args)
