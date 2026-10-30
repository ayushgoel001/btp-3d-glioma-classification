# Data acquisition and cohort

The experiments use the labelled **BraTS 2020 training cohort**, comprising 369 patients: 76 LGG and 293 HGG. Each patient contributes FLAIR, T1, T1ce, T2 and a tumour segmentation mask. The local uncompressed set contains 1,845 NIfTI files, approximately 30.83 GiB.

The original images generally have shape `240 × 240 × 155` with 1 mm isotropic spacing. The provider supplies registered, skull-stripped data. The project additionally canonicalizes orientation, validates geometry and applies the preprocessing described in `docs/METHOD.md`.

## Sources

- [Official BraTS 2020 data description](https://www.med.upenn.edu/cbica/brats2020/data.html)
- Colab mirror used by the downloader: `awsaf49/brats20-dataset-training-validation` on Kaggle

Follow the provider's normal access and usage conditions. The downloader cannot grant dataset access. It sets `DISABLE_COLAB_CACHE=true` to avoid the failed Colab cache-service route encountered during the original session.

```bash
python -m pip install -e ".[download]"
python scripts/download_brats2020.py \
  --mapping-csv data/metadata/name_mapping.csv \
  --output-dir data/raw --workers 2
```

The helper handles the mirror's special segmentation filename for patient 355. It skips already readable local NIfTI files. A download progress bar alone does not establish completeness; build the strict manifest and require all 369 audit rows to pass before training.

## Files included in Git

- `metadata/name_mapping.csv`: supplied grade mapping and original public subject identifiers.
- `metadata/patient_splits.csv`: portable patient IDs, labels, fixed split and five-fold assignment. It omits runtime-specific image and cache paths.

Original volumes, processed tensors and survival metadata are not needed in the published repository and are not included. The target is the legacy dataset HGG/LGG grouping. It is not a contemporary molecular diagnosis.
