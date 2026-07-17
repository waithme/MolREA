# Organized Result Files

This directory groups result files by experiment. Original files in `data/` were copied or consolidated here; no original result files were deleted.

## 01_qm9_baseline_fusion

- `qm9_baseline_fusion_raw.csv`: seed-level QM9 baseline and fusion results combined from `baseline_summary.csv` and `fusion_summary.csv`, with an added `stage` column.
- `qm9_baseline_fusion_summary.csv`: target/model-level QM9 baseline-vs-fusion statistics from `qm9_baseline_fusion_target_model_stats.csv`.
- `qm9_fusion_only_target_model_summary_legacy.csv`: fusion-only target/model statistics retained for traceability.
- `tables/`: QM9 supplementary table files.

## 02_full_model_benchmark

- `full_model_benchmark_raw.csv`: seed-level full-model results copied from `summary.csv`.
- `full_model_benchmark_summary.csv`: dataset-level full-model summary copied from `fusion_summary_all.csv`.
- `full_model_activity_cliff_summary.csv`: activity-cliff-only full-model summary copied from `cliff_fusion_summary.csv`.

## 03_progressive_feature_ablation

- `progressive_feature_ablation_raw.csv`: seed-level progressive feature ablation results copied from `ablation_summary.csv`.
- `progressive_feature_ablation_summary.csv`: aggregated progressive feature ablation statistics copied from `ablation_summary_stats.csv`.
- `progressive_feature_ablation_raw_dedup_worst.csv`: retained cleaned raw copy used in earlier checks; row count matches the original raw file.
- `tables/`: LaTeX tables for the progressive ablation experiment.

## 04_activity_cliff_mechanism_ablation

- `activity_cliff_mechanism_ablation_raw.csv`: consolidated seed-level raw file used for the activity-cliff mechanism ablation table. It combines feature-only controls, mechanism controls, and Full results.
- `activity_cliff_mechanism_ablation_summary.csv`: aggregated table statistics copied from `temp_abla2_cliff_ablation_combined_stats.csv`.
- `source_temp_feature_controls_raw.csv`: original feature-control source file copied from `temp_summary.csv`.
- `source_abla2_mechanism_controls_raw.csv`: original mechanism-control source file copied from `abla2_summary.csv`.
- `tables/`: Markdown and LaTeX previews for the activity-cliff mechanism ablation tables, including both cliff-sample RMSE and overall test RMSE views.

## 05_cliffdata_compare

- `cliffdata_compare_row.csv`: consolidated seed-level activity-cliff benchmark rows for four models: `CSGL`, `MolR`, `RXGL`, and `MolREA`. The first three come from their `official_cliff_split_results.csv` files; `MolREA` is taken from the fusion rows in `02_full_model_benchmark/full_model_benchmark_raw.csv`.
- `cliffdata_compare_summary.csv`: aggregated model-by-dataset activity-cliff comparison summary. It reports overall test RMSE and cliff-only test RMSE means, sample standard deviations, formatted strings, and the corresponding test-set sizes.
- `cliffdata_compare.md`: Markdown comparison tables derived from the summary file, with separate tables for overall test RMSE and cliff-only test RMSE.

## Manifest

- `organized_results_manifest.csv`: file-level index for all organized result artifacts, including role and row count where applicable.
