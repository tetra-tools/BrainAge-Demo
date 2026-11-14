# Production preprocessing vs training

Source of truth for training: `generate_ptfiles.py` in
[`RS-JYC-HK-T1wMNIBrainAgeRegression`](https://github.com/tetra-tools/RS-JYC-HK-T1wMNIBrainAgeRegression).
Training read BIDS derivatives `space-avg152T1_desc-biascorr` (bias-corrected, rigid + affine
registered to avg152T1 1 mm), then applied the TorchIO chain below and cached tensors as float16.

| # | Step | Training | `brainage` (production) |
|---|---|---|---|
| 1 | Bias field correction | done upstream (`desc-biascorr`) | ANTsPy N4: shrink 4, 4×50 iterations, spline 100 mm. `--skip-bias-correction` if already done |
| 2 | Registration to avg152T1 1 mm | done upstream, **rigid + affine** (`apply_syn_to_shap.py`: *"Resolve the rigid+affine t1_path"*) | ANTsPy `antsRegistrationSyNQuick[a]` (rigid + affine), seed 123. `--skip-registration` if already in template grid |
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

## To confirm with the authors of the upstream derivatives

The `desc-biascorr` / `space-avg152T1` files were produced outside this repository. The defaults
(N4 before registration, `antsRegistrationSyNQuick[a]`, template `ravg152T1_1mm.nii`) are the best
reconstruction from the code; if the upstream pipeline used other settings (e.g. FSL FLIRT 12 dof,
different N4 parameters, another template file), set them in the bundle `manifest.json`
(`preprocessing.registration`, `preprocessing.n4`) — no code change needed for ANTsPy transforms.

**Recommended validation before release**: take ~50 test-split subjects, run
`brainage predict` on their *raw* T1w, and compare with the ages predicted from the training cache
(`predict_sfcn.py --population test`). A mean absolute difference well below the model MAE
(target < 1 year) confirms the production preprocessing matches.

```bash
brainage predict --list test_raw.tsv --bundle dist/brainage-sfcn-1.0.0 -o prod_test.tsv
# join prod_test.tsv with results/predictions_test_*.tsv on subject_id and compare predicted ages
```
