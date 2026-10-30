# Implemented method

This document describes the procedure that produced the saved GPU results. Training modules match the source snapshot in the original GPU export.

## Patient representation

One sample is one patient, with channels ordered FLAIR, T1, T1ce, T2. The label mapping is LGG = 0 and HGG = 1. The supplied segmentation defines ROI localization and is not a fifth input channel.

All four modalities and the mask must agree in shape, affine transform and spacing. Duplicate patient IDs, missing files, unexpected segmentation values and cached-label mismatches cause errors. Accepted mask values are 0, 1, 2 and 4.

## Preprocessing

1. Convert NIfTI volumes to canonical RAS orientation and verify geometry.
2. Calculate nonzero mean and population standard deviation separately for each full patient modality. Standardize nonzero values and retain zero background.
3. For ROI input, crop the bounding box of `seg > 0` plus 16 voxels of context, clipped to image bounds.
4. For whole-brain input, crop the union of nonzero MRI foregrounds with margin zero.
5. Resize MRI with trilinear interpolation to `4 × 96 × 96 × 96`. Resize the mask with nearest-neighbour interpolation for QA.
6. Cache float16 images, the target and metadata. Metadata include crop bounds, geometry, normalization statistics, preprocessing configuration and source hashes.

The loader converts cache images to float32. Use separate cache directories for different spatial sizes or crop configurations. Existing cache files are not automatically reprocessed to match a changed configuration.

## Network

The main model contains 2,081,186 trainable parameters and uses GroupNorm with eight groups.

| Component | Operation | Feature tensor, excluding batch |
| --- | --- | --- |
| Input | Four modalities | `4 × 96 × 96 × 96` |
| Stem | 5-cubed Conv3d, stride 2, padding 2; GN; ReLU | `16 × 48 × 48 × 48` |
| Pool | 3-cubed max pool, stride 2, padding 1 | `16 × 24 × 24 × 24` |
| Stage 1 | Two residual blocks, width 16 | `16 × 24 × 24 × 24` |
| Stage 2 | Two blocks, width 32, first stride 2 | `32 × 12 × 12 × 12` |
| Stage 3 | Two blocks, width 64, first stride 2 | `64 × 6 × 6 × 6` |
| Stage 4 | Two blocks, width 128, first stride 2 | `128 × 3 × 3 × 3` |
| Head | Global average pool, dropout 0.30, linear | Two logits |

Each residual block has two 3-cubed convolutions with GroupNorm, a ReLU after the first normalization and another after the residual sum. Shape changes use a 1-cubed projected shortcut with GroupNorm. All weights start from random initialization.

## Training

| Setting | Recorded value |
| --- | --- |
| Seed | 42 |
| Physical batch | 1 patient |
| Gradient accumulation | 4 steps |
| AdamW learning rate | 0.0002 |
| Weight decay | 0.0001 |
| Scheduler | Cosine, configured over 60 epochs |
| Precision | CUDA AMP |
| Gradient norm clipping | 5 |
| Early-stopping patience | 12 epochs |
| Checkpoint criterion | Highest validation balanced accuracy |

Training augmentation applies a left-right flip with probability 0.5, independent channel intensity scaling in [0.9, 1.1], and foreground Gaussian noise with SD 0.03 and probability 0.5. Validation and test have no augmentation.

**Effective loss:** unweighted cross-entropy. The implementation supplies inverse-frequency weights to PyTorch's mean-reduced loss. At physical batch one, the target weight appears in both numerator and denominator and cancels. Accumulation takes place after this reduction. The final shorter accumulation window is also divided by four. These implementation limitations remain in the preserved source and require new experiments to correct.

## Evaluation

The fixed split contains 257 training, 56 validation and 56 test patients. Both crop settings share it. Each CV rotation tests fold `k`, validates on `(k + 1) mod 5` and trains on the remaining three folds. All partitions within a run are disjoint, and each of the 369 patients has one outer-test prediction.

HGG is positive. HGG sensitivity and LGG specificity are the two class recalls. Balanced accuracy is their mean. Macro-F1 is the mean of the two class F1 scores. ROC-AUC ranks HGG scores. The saved classification rule uses a threshold of 0.5.

Pooled metrics use all OOF predictions together. Mean fold metrics average five fold scores; sample SD uses denominator four. Fold SD is not a confidence interval. Softmax scores have not been calibrated.

## Engineering verification

An eight-patient training-only check uses a smaller width-four network with dropout zero. It reaches training accuracy 1.0 and loss 0.01994 after 47 epochs, and predictions match after checkpoint reload. These are debugging outcomes, not held-out evaluation.

Metric recomputation, split checks and source-hash checks can run from the compact repository evidence. Seven full model checkpoints remain in the original local export and are excluded from Git.
