# GitHub Upload Notes

This folder contains the files intended for repository upload and code/documentation marking.

Included:

- `configs/`: benchmark configuration.
- `scripts/`: numbered runnable scripts, in the order used for download, inspection, splitting, modelling, evaluation, secondary analysis and report generation.
- `src/ica2/`: reusable project code for config loading, splitting, metrics, label harmonisation, embedding-cache validation and the shared linear probe.
- `data/splits/`: small reproducible train/validation/test split files.
- `docs/`: design notes, dataset summary, constraints and result notes.
- `results/`: benchmark tables, metrics and report figures.
- `reports/`: final Word reports and Markdown source text.
- `requirements.txt`: minimal Python package list for the non-model-specific parts of the workflow.

Excluded because they are too large or machine-specific:

- `data/raw/*.h5ad`
- `data/processed/*.npz` embedding caches
- `.conda-envs/`
- `third_party/` cloned model repositories
- downloaded model checkpoints and weights
- logs and render temporary folders

The included CSV predictions and metric files are enough to regenerate the final tables and figures. They are not enough to recompute foundation-model embeddings from scratch. To rerun the complete model workflow, install or clone the required model repositories locally, download the model checkpoints, set `CMML_MODEL_DIR`, then follow the command sequence in `README.md`.
