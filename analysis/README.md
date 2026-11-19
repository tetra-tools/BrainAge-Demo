# Thesis analyses (Chapter IV)

Scripts that analyse the SFCN predictions written by the training repository
(`RS-JYC-HK-T1wMNIBrainAgeRegression`). They are not part of the `brainage` package or of the
Docker image.

| Script | Output |
|---|---|
| `fig_brainage_scatter.py` | predicted vs chronological age figure (healthy, native scans), statistics with bootstrap CIs, MAE by age band / sex / augmentation, thesis text |
| `inventory_databases_diagnoses.py` | tables of all databases and of all raw neurological / psychiatric diagnoses before cleaning, with their grouping |

Shared code: `analysis_io.py` (reading the TSVs as the Figure 13 forest plot does) and
`diagnosis_groups.py` (diagnosis cleaning and grouping rules copied from the forest-plot notebook).

## Install

```bash
pip install -r analysis/requirements.txt
```

## Inputs

Put the prediction tables in `data/predictions/` (ignored by git) or pass `--pred-dir`:

```
predictions_test_<tag>.tsv                    predictions_prediction-test_<tag>.tsv
predictions_validation_<tag>.tsv              predictions_prediction-validation_<tag>.tsv
<tag> = lossmae_h32_lr0.04_batch16_dr0.4_model_best_score   (--model-tag)
```

Columns used: `subject_id`, `session_id`, `database`, `augmentation`, `is_original`, `age`,
`predicted_age_years`, `sex`, `healthy`, `neurological_diagnosis`, `psychiatric_diagnosis`.

Splits:
* `test` = **Test**, used for hyperparameter selection;
* `validation` = **External validation** (`database` = `camcan`, `oasis3`), final validation and main result.

Default paths are relative to the repository root; paths given on the command line are
relative to the current directory.

## Run

```bash
python analysis/fig_brainage_scatter.py               # -> results/analysis/
python analysis/inventory_databases_diagnoses.py      # -> results/analysis/inventory/
python analysis/fig_brainage_scatter.py --pred-dir path/to/tsv --out-dir path/to/out --n-boot 1000 --seed 2026
```

Both scripts are executable (`./analysis/fig_brainage_scatter.py --help`). Exit code `0` on
success, `2` if an input file or column is missing.

### `fig_brainage_scatter.py`

Population, per split: native original scans (`augmentation == "native"` and `is_original`,
healthy and prediction files concatenated and de-duplicated as in the forest plot),
`healthy == 1`, chronological age floored to whole years (the precision of the training
targets). Nothing is smoothed or excluded; scans with |gap| > 15 years are listed.

Outputs: `fig_brainage_scatter.pdf/.png` (16 cm wide, no title, PNG 300 dpi),
`brainage_stats.csv/.md`, `brainage_mae_by_subgroup.csv/.md` (age band, sex, augmentation),
`brainage_text_snippets.md` (results sentences and legend), `scans_abs_gap_gt15.csv`, `checks.md`.

Confidence intervals: percentile bootstrap resampling participants (`database/subject_id`),
1,000 resamples, seed 2026.

Training MAE (to compare with test and external validation, e.g. for overfitting): pass
`--train-file path/to/training_predictions.tsv`, a table of the training set predicted by the
final model, with the same columns (`sex` optional). It gets the same population rules (native,
healthy, floored age) and adds a "Training" column to the tables, a line in the inset of panel
(a) and a comparison sentence to the text. Use predictions of the final model on the native
training scans, not the loss logged during training (augmented samples, dropout active).
`checks.md` reports participants shared between training and test or external validation.

### `inventory_databases_diagnoses.py`

Rows: native original scans of all participants. Outputs in `inventory/`: `databases.csv`,
`diagnoses_raw.csv` (each raw value with the groups assigned), `diagnosis_tokens.csv` (rule
applied to each token, including tokens that are not mapped), `diagnosis_groups.csv`, and
`inventory.md` with all tables.

## Tests

```bash
pytest analysis/tests      # synthetic data, a few seconds
```
