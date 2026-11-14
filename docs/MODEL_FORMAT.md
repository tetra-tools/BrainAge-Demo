# Trained model: format and storage

## TL;DR

| What | Format | Where |
|---|---|---|
| **Production model** | **Model bundle** = `model.onnx` + preprocessing assets + `manifest.json` (SHA-256), packed as `brainage-sfcn-<version>.tar.gz` | **GitHub Release** of `tetra-tools/BrainAge-Demo` (tag `model-sfcn-v<version>`), never committed to git |
| Docker image | OCI image, `runtime` (model mounted) and `bundled` (model inside) | **GHCR**: `ghcr.io/tetra-tools/brainage:<code-version>` and `:<code-version>-sfcn<model-version>` |
| Training source of truth | Lightning `model_best_score.ckpt` (+ W&B run) | Kept where it is (`/ix1/hkarim/.../checkpoints/sfcn/...`, W&B artifact). Its SHA-256 is written into the bundle manifest for traceability |

## Why ONNX (and not the `.ckpt` or a `state_dict`)

* **Self-describing graph**: the architecture is inside the file. A `.ckpt` needs the exact Python
  class (`models/sfcn.py`), PyTorch Lightning, and is a pickle (arbitrary code execution on load).
* **Small runtime**: `onnxruntime` (CPU or CUDA), no Lightning; the image only keeps torch because
  TorchIO needs it for the preprocessing.
* **Frozen inference behaviour**: BatchNorm in eval mode and Dropout removed are baked into the graph;
  nobody can forget `model.eval()`.
* **Verified at export**: `brainage export` compares PyTorch and ONNX Runtime outputs and refuses to
  write the bundle if they differ by more than 0.01 year (the diff is stored in the manifest).
* Exported in **float32**, dynamic batch axis, opset 18, single file (no external `.data`).

TorchScript or `torch.export` (`.pt2`) would also work but tie production to a PyTorch version;
safetensors only stores weights (code still needed). ONNX is the most portable choice here.

## Why a *bundle* and not just `model.onnx`

The network is only valid with the exact preprocessing it was trained with. Shipping the model alone
is how silent errors happen (wrong landmarks → shifted intensities → biased ages, no crash). The bundle
contains everything that defines the model's input distribution:

```
brainage-sfcn-1.0.0/
├── manifest.json      # version, architecture, preprocessing parameters, sha256 of every file
├── model.onnx         # SFCN (h32, MAE) – float32
├── t1_landmarks.npy   # cached_data/t1_landmarks.npy (HistogramStandardization, trained on IXI)
├── template.nii       # /ix1/hkarim/team/references/ravg152T1_1mm.nii  (registration target)
└── brain_mask.nii     # /ix1/hkarim/team/references/ravg152T1_1mm_defaced.nii (> 0.1 at runtime)
```

Checksums are verified each time the bundle is loaded (`--no-verify` to skip), so a modified or
partially downloaded bundle is rejected.

## Creating the bundle (on the cluster, where the checkpoint lives)

```bash
pip install "brainage[export] @ git+https://github.com/tetra-tools/BrainAge-Demo"
brainage export \
  --checkpoint /ix1/hkarim/users/jpc211/projects/RS-JYC-HK-T1wMNIBrainAgeRegression/checkpoints/sfcn/lossmae_h32_lr0.04_batch16_dr0.4/model_best_score.ckpt \
  --landmarks  /ix1/hkarim/users/jpc211/projects/RS-JYC-HK-T1wMNIBrainAgeRegression/cached_data/t1_landmarks.npy \
  --template   /ix1/hkarim/team/references/ravg152T1_1mm.nii \
  --mask       /ix1/hkarim/team/references/ravg152T1_1mm_defaced.nii \
  --version 1.0.0 \
  --notes "SFCN h32 MAE, trained on healthy run-01 split=train; test MAE = <fill in>" \
  --out dist/brainage-sfcn-1.0.0 --archive
```

Or with Docker (no local install):

```bash
docker run --rm -v /ix1:/ix1:ro -v "$PWD/dist:/out" brainage:0.1.0 export \
  --checkpoint /ix1/.../model_best_score.ckpt --landmarks /ix1/.../t1_landmarks.npy \
  --template /ix1/hkarim/team/references/ravg152T1_1mm.nii \
  --mask /ix1/hkarim/team/references/ravg152T1_1mm_defaced.nii \
  --version 1.0.0 --out /out/brainage-sfcn-1.0.0 --archive
```

> The checkpoint must be the one whose training cache was built with **this** `t1_landmarks.npy`
> and **this** mask. If several `cached_data` folders exist, take the landmarks from the one used
> for that run.

## Publishing

```bash
gh release create model-sfcn-v1.0.0 -R tetra-tools/BrainAge-Demo \
  --title "SFCN brain-age model 1.0.0" --notes-file dist/brainage-sfcn-1.0.0/manifest.json \
  dist/brainage-sfcn-1.0.0.tar.gz dist/brainage-sfcn-1.0.0.tar.gz.sha256
```

Fetching it (users with access to the repo):

```bash
gh release download model-sfcn-v1.0.0 -R tetra-tools/BrainAge-Demo -D models/
sha256sum -c models/brainage-sfcn-1.0.0.tar.gz.sha256 && tar -xzf models/brainage-sfcn-1.0.0.tar.gz -C models/
brainage info --bundle models/brainage-sfcn-1.0.0
```

Why GitHub Releases: same access control as the code (private while the paper is under review),
immutable versioned assets up to 2 GB, nothing to set up, and the code version ↔ model version link
is visible in one place. A ~12 MB ONNX + ~15 MB of NIfTI does not justify Git LFS (bandwidth quotas)
or a dedicated bucket. If the lab already runs an S3/GCS bucket or a model registry (W&B Registry,
MLflow), the same `.tar.gz` + `.sha256` can live there instead; nothing in the code depends on where
the bundle comes from.

At publication time, archive the same `.tar.gz` on **Zenodo** to get a DOI for the paper. Before
making it public, check the redistribution terms of the training datasets (for the weights) and keep
the MNI/avg152T1 copyright notice with the template.

## Versioning

* Bundle `version` = model version (SemVer): MAJOR when inputs/preprocessing change (new template,
  landmarks, registration), MINOR for a retrained model with the same preprocessing, PATCH for
  metadata-only fixes.
* Image tags: `brainage:<code-version>` (no model) and `brainage:<code-version>-sfcn<model-version>`.
* `manifest.format_version` is the bundle layout version understood by the library (currently 1).
