# Thesis analyses (Chapter IV)

Scripts that analyse the SFCN predictions written by the training repository
(`RS-JYC-HK-T1wMNIBrainAgeRegression`). They are not part of the `brainage` package or of the
Docker image.

| Script | Output |
|---|---|
| `fig_brainage_scatter.py` | predicted vs chronological age figure (healthy, native scans), statistics with bootstrap CIs, MAE by age band / sex / augmentation, thesis text |
| `inventory_databases_diagnoses.py` | tables of all databases and of all raw neurological / psychiatric diagnoses before cleaning, with their grouping |
| `fig_brainage_followup.py` | longitudinal course of predicted age (external validation, participants with ≥ 2 sessions): stable healthy, converters, patients |
| `demographics_tables.R` | demographics tables by split (subject level and scan level, `.docx`), diagnosis flag counts |

Shared code: `analysis_io.py` (reading the TSVs as the Figure 13 forest plot does) and
`diagnosis_groups.py` (diagnosis cleaning and grouping rules copied from the forest-plot notebook).

## Install

```bash
pip install -r analysis/requirements.txt
# R script: readr, dplyr, stringr, purrr, tidyr; gtsummary + gt for the .docx tables
Rscript -e 'install.packages(c("readr","dplyr","stringr","purrr","tidyr","gtsummary","gt"))'
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

### `fig_brainage_followup.py`

```bash
python analysis/fig_brainage_followup.py           # -> results/analysis/followup/
```

Participants of the external validation set (`--split`) with at least two sessions, native
scans of healthy participants and patients. Group from `healthy` at the first and last session
(age order): stable healthy (1 → 1, never 0), converter (1 → 0), patient (0 → 0, never 1), other
(any other sequence: counted, not plotted). Figure: (a–c) one line per participant, change in
predicted age since the first scan vs years since the first scan, with the group mean curve and
its 95% CI (drawn where at least `--min-n` participants contribute, default 5); (d) the three
mean curves. Ages are whole years, so sessions within the same year share a time point.
Outputs: figure (PDF + PNG), `brainage_followup_mean_curves.csv`, per-participant table with
the rate of change, summary table per database and group, legend, `checks.md`.

### `demographics_tables.R`

```bash
Rscript analysis/demographics_tables.R            # -> results/analysis/demographics/
```

Reads all prediction tables of one model (`--model-tag`), every column as text, keeps one row
per scan (native original scan; rows present in both files counted once) and applies the
Figure 13 diagnosis rules (tokens split on `, ; |`). Subject level: baseline status, age, sex,
APOE, sessions, samples (native + augmented, rows present in both files counted once) and
follow-up duration in two rows (all subjects with single session = 0; subjects with at least
two sessions only). Sex: female/male in any case; a missing value is filled from the same
participant's other scans when they agree. APOE: 0 = non-carrier, > 0 = carrier; -1, -1.0,
`Unknown`, empty and NA = unknown. Scan level: status, age, diagnosis flags, augmented samples
per transform. `diagnosis_flag_counts.tsv` counts flags as Figure 13 (`n_positive`: all native
scans of the split), with the number of healthy scans among them. `split_counts.tsv` gives
scans, participants and CN participants (healthy == 1) per split; `demographics_checks.txt`
repeats them and lists every raw sex and APOE value with its coded category.
It stops on the same labelling problems as the Python scripts (`--no-strict` to only report
them) and writes `subject_level.tsv`, `scan_level.tsv`, `diagnosis_flag_counts.tsv` and
`demographics_checks.txt`; `--no-tables` skips the `.docx` tables.

## No scan lost

Every script reads identifiers as text (leading zeros kept) and stops, listing the rows, when
a labelling or encoding problem would drop or misgroup a scan: `augmentation` not exactly
`native` or empty on an original row, `is_original` not boolean, `healthy` not 0/1, age or
prediction not numeric, external label other than `camcan`/`oasis3`, a scan with no native
original row, a file that is not UTF-8. `scan_accounting.csv` follows every scan from the files
to the analysed groups. Use `--no-strict` to report these problems without stopping.

## Tests

```bash
pytest analysis/tests      # synthetic data, ~20 s; the R tests are skipped if Rscript is missing
```
