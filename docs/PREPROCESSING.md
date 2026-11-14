# Production preprocessing vs training

Source of truth for training: `generate_ptfiles.py` in
[`RS-JYC-HK-T1wMNIBrainAgeRegression`](https://github.com/tetra-tools/RS-JYC-HK-T1wMNIBrainAgeRegression).
Training read BIDS derivatives `space-avg152T1_desc-biascorr`, produced by the dataset conversion
scripts (e.g. the A4 → BIDS script) with two ANTs command-line calls:

```bash
N4BiasFieldCorrection -d 3 -v 1 -s 4 -b [100] -c [50x50x50x50,0.0] -i raw.nii.gz -o space-native_desc-biascorr.nii.gz
antsRegistrationSyNQuick.sh -d 3 -f <avg152T1 template> -m space-native_desc-biascorr.nii.gz -o prefix_ -n 16 -t a
# prefix_Warped.nii.gz -> space-avg152T1_desc-biascorr.nii.gz
```

then applied the TorchIO chain below and cached tensors as float16.

| # | Step | Training | `brainage` (production) |
|---|---|---|---|
| 1 | Bias field correction | `N4BiasFieldCorrection -s 4 -b [100] -c [50x50x50x50,0.0]`, no mask | ANTsPy N4 with the same parameters (no mask = whole image). `--skip-bias-correction` if already done |
| 2 | Registration to avg152T1 1 mm | `antsRegistrationSyNQuick.sh -t a` (rigid + affine, MI, winsorize [0.005,0.995], linear interpolation) | the same `antsRegistration` stages called in process through ANTsPy (`syn_quick_affine_args`); only `--float 1` instead of the script's double precision. `--skip-registration` if already in template grid |
| 3 | Read NIfTI | `tio.ScalarImage(path)` | registered image written to a temp NIfTI and read with `tio.ScalarImage` (same code path) |
| 4 | NaN/Inf → 0 | `clean_image_data` | identical |
| 5 | Intensity rescale | `RescaleIntensity((0, 1), percentiles=(0.5, 99.5))` | identical |
| 6 | Histogram standardisation | `HistogramStandardization({'t1': landmarks})`, IXI landmarks | identical, landmarks shipped in the bundle |
| 7 | Brain mask | `ravg152T1_1mm_defaced.nii > 0.1` | identical, mask shipped in the bundle, threshold in the manifest |
| 8 | Crop/pad | `CropOrPad((160, 192, 160), padding_mode=0)` | identical |
| 9 | Precision | cached as float16 | float16 round-trip, then float32 into ONNX |
| – | Augmentation | `RandomFlip/Blur/Noise/Motion/Ghosting/Spike` on non-original copies | **none** |

`tests/test_pipeline.py::test_preprocessing_is_identical_to_training_cache` checks that steps 3–9
produce a tensor **bit-identical** to the training code for an image already in template space.

## Issues fixed compared with `predict_image_sfcn.py`

The existing single-image script of the training repo does not reproduce the training inputs:

1. **Non-linear SyN registration** (`type_of_transform='SyN'`) whereas training images were only
   rigid + affine registered → the anatomy is warped onto the template, which likely attenuates the
   age-related morphology (ventricle size, sulcal widening) the model relies on, and in any case
   gives the network inputs it never saw during training.
2. **Different mask**: `ravg152T1_1mm_mask.nii.gz` used as is instead of `ravg152T1_1mm_defaced.nii > 0.1`.
3. **Landmarks passed as an array** to `tio.HistogramStandardization` instead of `{'t1': array}`.
4. **ANTs → TorchIO in-memory conversion** uses the ANTs (LPS) direction/origin as a RAS affine,
   so the image geometry is mislabelled (x/y flipped in world coordinates) and inconsistent with
   the mask. Voxel-wise steps still line up, but any geometry check, resampling or saved output is
   wrong. We go through a NIfTI file instead, exactly like training.
5. `import antspy as ants` (the package is imported as `ants`), template/mask/landmarks pointing to
   cluster paths, preprocessing silently continuing when N4 or the registration fails, preprocessed
   image saved with an identity affine.

## Remaining differences with the conversion scripts

* **Determinism**: a fixed `--random-seed 123` is always passed. With `--threads 1` the registration
  is bit-reproducible; with several threads (like the script's `-n 16`, which had no seed either) the
  registered image varies slightly between runs (≈1 % mean relative intensity difference on the
  synthetic test scan).
* **Precision**: the script runs `antsRegistration` in double precision (`--float 0`); ANTsPy passes
  images in memory as float, so the optimisation runs in single precision. Expected effect: sub-voxel.
* **ANTs version**: the derivatives were produced with the cluster's ANTs (`module load ants`, version
  recorded by `get_ants_version()` in the conversion log); the image ships the ANTs bundled in
  `antspyx==0.6.3`. Same algorithms, possibly different defaults across major versions.
* **Template file**: the script takes it from `--fixed`. The bundle must contain that exact file
  (most likely `/ix1/hkarim/team/references/ravg152T1_1mm.nii`, the template also used by the SHAP
  warps) – check the `--fixed` argument in the conversion job scripts before exporting.
* The script's `--mask` argument is not used by the registration; masking happens only in the
  TorchIO chain (step 7), which the bundle reproduces.

**Recommended validation before release**: take ~50 test-split subjects, run
`brainage predict` on their *raw* T1w, and compare with the ages predicted from the training cache
(`predict_sfcn.py --population test`). A mean absolute difference well below the model MAE
(target < 1 year) confirms the production preprocessing matches.

```bash
brainage predict --list test_raw.tsv --bundle dist/brainage-sfcn-1.0.0 -o prod_test.tsv
# join prod_test.tsv with results/predictions_test_*.tsv on subject_id and compare predicted ages
```
