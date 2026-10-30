# Colab notebooks

| File | Role |
| --- | --- |
| `BTP1_GPU_Cleaned_Review.ipynb` | Readable edited record of the completed GPU experiments |
| `BTP1_GPU_Executed.ipynb` | Unmodified executed notebook, including the historical setup failures |
| `BTP1_GPU_Run_Notebook.ipynb` | Fresh-run template with empty outputs and timestamped result directories |
| `btp1_gpu_code.zip` | Source and label bundle consumed by the fresh-run notebook |

The cleaned notebook removes duplicate failed download cells, consolidates setup and corrects stale comments. Its retained training, evaluation, inference and QA outputs remain unchanged from the original executed record. It does not represent a new training run. Original execution counts can have gaps.

The original notebook includes successful GPU experiments after earlier download errors. Use the cleaned copy for reading and the fresh-run template for execution. Refer to the method note about effective class weighting before interpreting or repeating the baseline.

The code bundle does not contain original MRI, cached tensors or checkpoints. Upload it to `MyDrive/BTP1_GPU/`, then open the fresh-run notebook in Colab. See `docs/REPRODUCING.md` for details.
