# Implementation findings and study limits

## Class weights cancel at the recorded physical batch size

`src/btp3d/train.py` configures `CrossEntropyLoss(weight=class_weights)` with the default mean reduction. For one class-index target, PyTorch divides the weighted loss by that same target weight:

```text
loss = w[y] * (-log p[y]) / w[y] = -log p[y]
```

The recorded physical batch is one. Accumulating four singleton gradients does not recover the removed weight. The archived experiments are therefore effectively unweighted. The report and README describe this effective objective, even though `class_weights_lgg_hgg` remains in the saved metric JSON as a record of the configured values.

A future correction can explicitly reduce unreduced weighted losses using a declared normalization convention. It must be checked with unequal weights and retrained as a separately named experiment. No corrected-loss GPU result is included here.

## Final accumulation window

Every backward loss is divided by the configured factor of four. When fewer than four patients remain at the end of an epoch, that update is smaller than the mean over the actual window. The fixed training split has 257 patients, so its last window has one patient.

## Checkpoints and cache reuse

The best-model checkpoint stores weights and model/preprocessing configuration. It omits the full optimizer, scheduler and scaler state needed to resume the exact training trajectory after an interruption. Rerunning training starts a new trajectory.

Preprocessing reuses existing cache filenames unless told to overwrite. Keep a separate cache directory for each spatial size, crop mode and margin to prevent accidental reuse of a different configuration.

## Interpretation

- ROI inference needs a segmentation mask. The project does not include an automatic segmenter.
- Evaluation uses one selected cohort and one seed. There is no external or held-out-site study.
- Only 76 cohort patients are LGG; the fixed test split contains 12 LGG patients.
- The target is the legacy BraTS HGG/LGG label, not molecular diagnosis or four tumour types.
- Fold variation describes patient partitions and overlapping fitted training sets. It is not a confidence interval or repeated-seed uncertainty.
- Pooled AUC and average fold AUC are distinct. Predictions from separate models can have different score scales.
- No calibration, threshold optimization, self-supervised pretraining or five-fold whole-brain run was performed.
- The fixed and pooled ROI results come from different models and partitions. Their difference is not a measured improvement in one deployed model.

The outputs support an academic internal baseline. They do not establish clinical diagnostic performance.
