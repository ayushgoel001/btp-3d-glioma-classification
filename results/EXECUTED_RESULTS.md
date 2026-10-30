# Executed GPU results

These values come from the saved real-data Colab experiment, using all 369 labelled BraTS 2020 patients. No new GPU training was performed while preparing this repository.

Main models use four MRI channels, 96-cubed inputs, base width 16 and seed 42. The effective objective is unweighted cross-entropy because configured weights cancel at physical batch one.

## Aggregate results

| Experiment | n | Accuracy | Balanced accuracy | Macro-F1 | HGG recall | LGG recall | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ROI fixed split | 56 | 83.93% | 71.59% | 73.63% | 93.18% | 50.00% | 0.8097 |
| ROI pooled OOF | 369 | 87.26% | 84.67% | 81.96% | 89.08% | 80.26% | 0.8815 |
| Whole-brain fixed split | 56 | 75.00% | 65.91% | 64.94% | 81.82% | 50.00% | 0.7936 |

Fixed ROI and whole-brain results refer to the same 56 patients. The pooled OOF result combines one held-out prediction per cohort patient from five fitted models.

## Fold results

| Fold | n | Accuracy | Balanced accuracy | Macro-F1 | HGG recall | LGG recall | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 74 | 90.54% | 84.12% | 84.99% | 94.92% | 73.33% | 0.8791 |
| 1 | 74 | 89.19% | 88.25% | 84.73% | 89.83% | 86.67% | 0.9412 |
| 2 | 74 | 83.78% | 74.92% | 74.92% | 89.83% | 60.00% | 0.8802 |
| 3 | 74 | 83.78% | 87.39% | 80.05% | 81.03% | 93.75% | 0.9116 |
| 4 | 73 | 89.04% | 88.16% | 84.66% | 89.66% | 86.67% | 0.8920 |

## Mean fold scores

| Metric | Mean | Sample SD |
| --- | ---: | ---: |
| Balanced accuracy | 84.57% | 5.65 pp |
| Macro-F1 | 81.87% | 4.40 pp |
| HGG recall | 89.05% | 5.01 pp |
| LGG recall | 80.08% | 13.44 pp |
| ROC-AUC | 0.9008 | 0.0261 |

Sample SD describes the five partitions and uses denominator four. It is not a confidence interval. Pooled AUC 0.8815 differs from mean fold AUC 0.9008 because pooled ranking compares scores across different fitted models.

## Pooled confusion counts

| True class | Predicted LGG | Predicted HGG | Total |
| --- | ---: | ---: | ---: |
| LGG | 61 | 15 | 76 |
| HGG | 32 | 261 | 293 |

The pooled predictions contain 15 LGG-to-HGG errors and 32 HGG-to-LGG errors. Correct predictions total 322 of 369.

## Training durations and selection

| Model | Selected epoch | Stopped epoch | Training minutes |
| --- | ---: | ---: | ---: |
| ROI fixed | 10 | 22 | 9.69 |
| CV fold 0 | 22 | 34 | 13.18 |
| CV fold 1 | 11 | 23 | 9.17 |
| CV fold 2 | 29 | 41 | 15.98 |
| CV fold 3 | 9 | 21 | 8.15 |
| CV fold 4 | 19 | 31 | 12.50 |
| Whole-brain fixed | 6 | 18 | 7.73 |

The five ROI fits total 3539.85 seconds (59.00 minutes). The saved Tesla T4 timers exclude download, preprocessing and document preparation.

## Debugging check

The training-only eight-patient memorization check passed after 47 epochs, with training accuracy 100.0%, loss 0.01994 and matching predictions after checkpoint reload. It uses width four and dropout zero. It is not held-out evaluation.

## Evidence

- [Compact GPU export](gpu/)
- [Pooled patient predictions](gpu/cv5/out_of_fold_predictions.csv)
- [Source and document hashes](provenance/source_record.json)
- [Original executed notebook](../notebooks/BTP1_GPU_Executed.ipynb)
- [Implementation limits](../docs/KNOWN_LIMITATIONS.md)

Raw MRI, cache tensors, checkpoints and the full results ZIP are excluded from Git. The original local export retains the checkpoints. The repository evidence supports arithmetic and partition checks without independently repeating GPU training.
