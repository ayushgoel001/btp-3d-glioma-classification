# Frozen GPU experiment evidence

These files are copied from the supplied completed Colab export. They preserve patient predictions, metrics, histories, split assignments and environment records. No metric is replaced by a new run during repository preparation.

- `fixed_split/`: fixed ROI experiment and saved example inference.
- `cv5/fold_0/` through `cv5/fold_4/`: five ROI fits with separate validation and test folds.
- `cv5/out_of_fold_predictions.csv`: one outer-test prediction per patient.
- `cv5/cross_validation_metrics.json`: pooled results and fold summaries.
- `whole_brain_fixed/`: comparison using the same fixed patient split.
- `overfit_check/`: training-only debugging result, not evaluation.
- Top-level CSV and TXT files: cohort audit, runtime manifests and environment.

Paths inside saved manifests refer to the original Colab runtime. Their referenced MRI and caches are not present in this repository. Regenerate manifests and tensors for a new run. The portable ID/label/split record is `../../data/metadata/patient_splits.csv`.

Model checkpoints and the complete results ZIP remain outside Git. The configured class weights in JSON files are historical settings; at physical batch one, the effective archived loss was unweighted. See `../../docs/KNOWN_LIMITATIONS.md`.
