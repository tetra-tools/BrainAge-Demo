# BrainAge-Demo

This repository is associated with a paper currently under review.

Production package for **brain age prediction from a raw T1-weighted MRI**, built from the
training code of [`RS-JYC-HK-T1wMNIBrainAgeRegression`](https://github.com/tetra-tools/RS-JYC-HK-T1wMNIBrainAgeRegression)
(SFCN, Peng et al. 2021). One command goes from a NIfTI file to an age in years:

```
raw T1w .nii(.gz)
  └─ N4 bias-field correction ............ ANTsPy
  └─ rigid + affine registration ......... ANTsPy, same stages as antsRegistrationSyNQuick.sh -t a
                                           → avg152T1 1 mm (182×218×182)
  └─ intensity rescale (p0.5–p99.5 → 0–1)  TorchIO  ┐
  └─ histogram standardisation ........... TorchIO  │ identical to training
  └─ brain mask ........................... TorchIO  │ (generate_ptfiles.py),
  └─ crop/pad to 160×192×160 ............. TorchIO  ┘ without augmentation
  └─ SFCN ................................ ONNX Runtime
→ predicted age (years) [+ brain-age gap if the chronological age is given; rounded down to whole years]
```

Contents:

* `src/brainage/` – Python library + `brainage` CLI
* `Dockerfile` – CPU image (`runtime`: model mounted at run time; `bundled`: model inside)
* `docs/MODEL_FORMAT.md` – **how the trained model is saved (ONNX bundle) and where it is stored**
* `docs/PREPROCESSING.md` – step-by-step equivalence with training, fixes vs `predict_image_sfcn.py`
* `tests/` – synthetic end-to-end tests (no real data needed)

---

## 1. The model bundle

The library never loads a raw checkpoint in production. It loads a **model bundle**: a folder with
`model.onnx`, the training landmarks, the registration template, the brain mask and a
`manifest.json` holding versions, preprocessing parameters and SHA-256 checksums.

* Get a published bundle (GitHub Release of this repo):
  ```bash
  gh release download model-sfcn-v1.0.0 -R tetra-tools/BrainAge-Demo -D models/
  sha256sum -c models/brainage-sfcn-1.0.0.tar.gz.sha256
  tar -xzf models/brainage-sfcn-1.0.0.tar.gz -C models/
  ```
* Or build one from a Lightning checkpoint with `brainage export` (see
  [docs/MODEL_FORMAT.md](docs/MODEL_FORMAT.md)).

Check a bundle: `brainage info --bundle models/brainage-sfcn-1.0.0`.

## 2. Docker (recommended)

Build:

```bash
docker build -t brainage:0.1.0 .
# optional: image with the model inside (no volume needed at run time)
docker build -t brainage:0.1.0-sfcn1.0.0 --target bundled --build-arg BUNDLE_DIR=models/brainage-sfcn-1.0.0 .
```

Run (inputs in `/data`, results in `/out`, bundle in `/opt/brainage/model`):

```bash
# one image
docker run --rm \
  -v /path/to/models/brainage-sfcn-1.0.0:/opt/brainage/model:ro \
  -v /path/to/scans:/data:ro -v "$PWD/results:/out" \
  brainage:0.1.0 predict /data/sub-01_T1w.nii.gz --age 63 -o /out/sub-01.json

# a whole folder (recursive search for .nii / .nii.gz), 8 threads
docker run --rm -v ...model...:/opt/brainage/model:ro -v /path/to/bids:/data:ro -v "$PWD/results:/out" \
  brainage:0.1.0 predict /data --threads 8 -o /out/brain_age.tsv

# with the bundled image
docker run --rm -v /path/to/scans:/data:ro brainage:0.1.0-sfcn1.0.0 predict /data/sub-01_T1w.nii.gz
```

Notes:
* The container runs as uid 1000; if `/out` is not writable, add `-u $(id -u):$(id -g)`.
* Singularity/Apptainer (HPC): `apptainer build brainage.sif docker-daemon://brainage:0.1.0`, then
  `apptainer run -B models/brainage-sfcn-1.0.0:/opt/brainage/model brainage.sif predict ...`.
* The image is CPU-only: preprocessing (N4 + registration, ~1 min) dominates and is CPU-bound,
  SFCN inference takes a few seconds on CPU. For GPU inference, install `brainage[gpu]` (instead of
  `onnxruntime`) on a CUDA base image; ONNX Runtime picks `CUDAExecutionProvider` automatically.
* Building behind a proxy without access to `download.pytorch.org`: add
  `--build-arg TORCH_INDEX_URL=https://pypi.org/simple` (larger CUDA wheel, same behaviour).

## 3. Python library

Install (Python ≥ 3.10):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU torch, optional but lighter
pip install "brainage[cpu] @ git+https://github.com/tetra-tools/BrainAge-Demo"     # ONNX Runtime CPU
# or, on a CUDA machine (never both: the two packages overwrite each other):
# pip install "brainage[gpu] @ git+https://github.com/tetra-tools/BrainAge-Demo"
# + checkpoint export tools: pip install "brainage[cpu,export] @ git+https://github.com/tetra-tools/BrainAge-Demo"
```

Use:

```python
from brainage import BrainAgePipeline

pipe = BrainAgePipeline("models/brainage-sfcn-1.0.0", num_threads=8)   # load once

res = pipe.predict("sub-01_T1w.nii.gz", chronological_age=63.9, subject_id="sub-01")
print(res.chronological_age, res.predicted_age, res.brain_age_gap)   # e.g. 63.0  66.41  3.41
print(res.timings)                               # seconds per step
res.to_dict()                                    # JSON-serialisable

# batch: one failing scan does not stop the others (status="error", error="...")
results = pipe.predict_many(["a.nii.gz", "b.nii.gz"], chronological_ages=[70, 45])

# already-processed inputs (e.g. the training derivatives space-avg152T1_desc-biascorr)
pipe.predict("sub-01_space-avg152T1_desc-biascorr_T1w.nii.gz",
             skip_bias_correction=True, skip_registration=True)

# lower-level pieces
tensor = pipe.preprocessor.run("sub-01_T1w.nii.gz").tensor   # (1, 1, 160, 192, 160) float32
age = pipe.predictor.predict(tensor)                           # np.array([66.41])
```

**Chronological age is always rounded down to whole years** (63.9 → 63), in the API, `--age`
and `--list`: the model was trained on ages rounded down, so the brain-age gap is the predicted
age minus the rounded-down age. The predicted age is the model output and is not rounded.
`chronological_age` in the results is the rounded-down value. A missing age (empty, `None`,
NaN) gives no gap; an age that is not a number, infinite or negative is rejected.

## 4. CLI reference

```
brainage predict [INPUTS ...] [--list FILE] [--age AGE] [--bundle DIR] [-o FILE] [--json]
                 [--save-preprocessed DIR] [--skip-bias-correction] [--skip-registration]
                 [--threads N] [--cpu | --providers P ...] [--no-verify] [--fail-fast]
brainage info    [--bundle DIR]
brainage export  --checkpoint CKPT --landmarks NPY --template NII --mask NII --version V --out DIR [--archive]
brainage archive BUNDLE_DIR [-o FILE]
```

| Option | Meaning |
|---|---|
| `INPUTS` | `.nii`/`.nii.gz` files and/or folders (searched recursively) |
| `--list` | TSV/CSV with a `path` column, optional `age` (rounded down to whole years) and `subject_id` (relative paths are relative to the list) |
| `--age` | chronological age for a single input, rounded down to whole years → `brain_age_gap = predicted − rounded-down age` |
| `--bundle` | bundle folder; default `$BRAINAGE_BUNDLE`, then `/opt/brainage/model` |
| `-o/--output` | `.tsv`, `.csv` (one row per image) or `.json` (full details incl. timings) |
| `--save-preprocessed` | writes `<stem>_space-template_desc-biascorr_T1w.nii.gz` (after N4 + registration) and `<stem>_desc-modelinput_T1w.nii.gz` (exact network input) for QC. `<stem>` = `subject_id`, else the file name without `_T1w`; inputs with the same name in one batch are prefixed with their parent folder, so nothing is overwritten |
| `--skip-bias-correction` | input already N4-corrected |
| `--skip-registration` | input already in the template grid (182×218×182, avg152T1 1 mm); otherwise an explicit error is raised |
| `--threads` | threads for ANTs, torch and ONNX Runtime. `--threads 1` makes the registration bit-reproducible (fixed seed, ~2.5× slower); with more threads the registered image varies slightly between runs, as with the original `antsRegistrationSyNQuick.sh -n 16` |
| `--no-verify` | skip bundle checksum verification (faster start) |
| `--fail-fast` | stop at the first error instead of reporting it and continuing |

Output columns (`.tsv/.csv`): `subject_id, input_path, predicted_age, chronological_age,
brain_age_gap, status, error, model_name, model_version, total_seconds`. `chronological_age` is
the rounded-down age used for `brain_age_gap`.

Exit codes: `0` all images OK, `1` at least one image failed, `2` configuration error
(bundle missing/corrupted, bad arguments).

## 5. Input requirements and limits

* One 3D T1-weighted volume per file (a 4D file with a single volume is accepted), any orientation
  and resolution, **whole head, not skull-stripped**: training used whole-head images registered to
  avg152T1 and masked with the template mask (`ravg152T1_1mm_defaced > 0.1`), so do not skull-strip.
* The model was trained on healthy `run-01` scans; predictions for ages outside the training range,
  for very different acquisitions or for strong pathology should be interpreted with care.
* Registration failures are not silently ignored (unlike the original script): the image is reported
  as `status=error`. Use `--save-preprocessed` to visually check registrations.
* Runtime on CPU (4 threads, measured on synthetic 1.1 mm scans): ~25 s N4 + ~25 s registration
  + < 5 s for the rest. With `--threads 1` (reproducible), registration takes ~65 s.
* In the Python API, set `num_threads` (or `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`) before ANTs is
  first used in the process: ITK fixes its thread count at first use.

## 6. Development

```bash
pip install -e ".[cpu,dev]"
pytest -m "not slow"      # ~20 s: bundle, checksums, ONNX parity, bit-exact preprocessing vs training
pytest                    # + full raw-scan pipeline with N4 and registration (~2 min)
```

The tests build a synthetic template, mask, scans, landmarks and a random-weight checkpoint, so
they need no data and no trained model.
