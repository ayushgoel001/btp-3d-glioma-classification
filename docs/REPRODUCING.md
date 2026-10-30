# Reproducing the experiments

## 1. Inspect the existing record

```bash
python scripts/verify_results.py
```

This standard-library-only command checks saved metrics, patient partitions, fold coverage and source hashes. It does not retrain the models or require MRI data.

Read `notebooks/BTP1_GPU_Cleaned_Review.ipynb` for the cleaned experimental record. The original executed notebook preserves setup failures and the eventual successful run. Fresh runs use `BTP1_GPU_Run_Notebook.ipynb`.

## 2. Install the project

Use Python 3.11 or later for a new environment. Choose a suitable PyTorch build using its official installation instructions. Colab normally supplies a CUDA-enabled build.

```bash
python -m pip install -e ".[download,dev]"
python -m pytest -q
```

The synthetic CPU tests exercise code paths only. Their inputs do not contribute to the reported results.

## 3. Run in Colab

1. Upload `notebooks/btp1_gpu_code.zip` to `MyDrive/BTP1_GPU/`.
2. Open the fresh-run notebook in Colab and select a GPU runtime.
3. Follow Steps 1–8 to set up, obtain the data, audit all 369 patients, cache ROI96 inputs, inspect QA and run the training-only debugging check.
4. Step 9 trains the fixed ROI model. Step 10 reads its result and demonstrates checkpoint inference.
5. Enable Step 11 for all five ROI rotations and Step 12 for the whole-brain fixed comparison.
6. Step 13 exports a results ZIP. Retain the executed notebook and complete output directory.

The template writes to a timestamped run directory. It does not overwrite the archived `roi96_seed42` folder. The code bundle contains source and labels, not MRI volumes or pretrained weights. Regenerate it after source changes:

```bash
python scripts/build_colab_bundle.py
```

The bundle preserves the archived effective loss for reproduction. Correcting the loss or accumulation requires a new experiment label and new results.

## 4. Direct command-line workflow

The examples below use Bash line continuation, as in a Linux or Colab terminal. On Windows, use the notebook or put each command on one line. Run from the repository root.

### Download and audit

```bash
python scripts/download_brats2020.py \
  --mapping-csv data/metadata/name_mapping.csv --output-dir data/raw --workers 2

python -m btp3d.manifest \
  --data-root data/raw --mapping-csv data/metadata/name_mapping.csv \
  --output runs/reproduction/manifest.csv --seed 42 --folds 5 --strict \
  --audit-output runs/reproduction/data_audit.csv
```

Require 369 unique patients and 369 valid audit rows. New manifests rebuild image paths for the current machine; archived Colab paths are evidence, not portable file locations.

### Cache and inspect ROI input

```bash
python -m btp3d.preprocess \
  --manifest runs/reproduction/manifest.csv --output-dir data/processed/roi96 \
  --spatial-size 96 96 96 --crop-mode roi --margin 16 --workers 2

python -m btp3d.qa \
  --manifest data/processed/roi96/manifest_cached.csv \
  --output-dir runs/reproduction/qa --per-split 2
```

Use a fresh cache directory when changing configuration. Inspect anatomy, channel alignment and tumour context before training. The notebook also filters the fixed training subset before the eight-patient overfit check.

### Fixed ROI model

```bash
python -m btp3d.train \
  --manifest data/processed/roi96/manifest_cached.csv \
  --output-dir runs/reproduction/fixed_split \
  --seed 42 --epochs 60 --batch-size 1 --gradient-accumulation 4 \
  --learning-rate 0.0002 --weight-decay 0.0001 --patience 12 \
  --num-workers 2 --base-channels 16 --group-norm-groups 8 --dropout 0.30
```

### Five-fold ROI evaluation

```bash
python -m btp3d.cross_validate \
  --manifest data/processed/roi96/manifest_cached.csv \
  --output-dir runs/reproduction/cv5 \
  --seed 42 --epochs 60 --batch-size 1 --gradient-accumulation 4 \
  --learning-rate 0.0002 --weight-decay 0.0001 --patience 12 \
  --num-workers 2 --base-channels 16 --group-norm-groups 8 --dropout 0.30
```

### Whole-brain fixed comparison

```bash
python -m btp3d.preprocess \
  --manifest runs/reproduction/manifest.csv --output-dir data/processed/brain96 \
  --spatial-size 96 96 96 --crop-mode brain --margin 0 --workers 2

python -m btp3d.train \
  --manifest data/processed/brain96/manifest_cached.csv \
  --output-dir runs/reproduction/whole_brain_fixed \
  --seed 42 --epochs 60 --batch-size 1 --gradient-accumulation 4 \
  --learning-rate 0.0002 --weight-decay 0.0001 --patience 12 \
  --num-workers 2 --base-channels 16 --group-norm-groups 8 --dropout 0.30
```

### Inference with an ROI checkpoint

```bash
python -m btp3d.infer \
  --checkpoint runs/reproduction/fixed_split/best_model.pt \
  --flair /path/patient_flair.nii.gz --t1 /path/patient_t1.nii.gz \
  --t1ce /path/patient_t1ce.nii.gz --t2 /path/patient_t2.nii.gz \
  --seg /path/patient_seg.nii.gz --output runs/reproduction/prediction.json
```

Provide real paths to four aligned modalities and the required mask. Inference uses the saved checkpoint preprocessing. Softmax values are uncalibrated scores.

## 5. Environment and runtime

The archived run records Python 3.13, PyTorch 2.11.0+cu130, NumPy 2.1.3, pandas 2.2.3, nibabel 5.4.2 and scikit-learn 1.6.1, with a Tesla T4. The full package list is retained in `results/gpu/environment.txt`.

Recorded training timers are approximately 9.69 minutes for fixed ROI, 59.00 minutes for five ROI folds together and 7.73 minutes for fixed whole brain. Download and preprocessing are excluded. These are measured past runtimes, not guarantees for another session. Exact GPU results can vary with library versions and kernels.

## 6. Documents

Compile `report/latex/main.tex` with XeLaTeX, or run `tectonic main.tex` inside that directory. The bibliography is embedded. Keep the accompanying images and generated table files in place. The report PDF remains a review draft.

The PowerPoint in `presentation/` is the retained 16-slide deck. It contains editable charts and tables.
