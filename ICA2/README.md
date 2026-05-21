# ICA2 Mini-project 1: Single-cell Foundation Model Benchmark

This repository contains the code, documentation, reports and result summaries for a mini-project benchmarking foundation-model embeddings for human pancreas scRNA-seq cell type annotation.

## Reproducibility Status

The repository is designed for code/documentation review and partial reruns without large binary assets. It includes the final metrics, predictions, figures, reports and all project scripts.

It does **not** include the following large or machine-specific files:

- raw h5ad data: `data/raw/*.h5ad`
- generated embedding caches: `data/processed/*.npz`
- downloaded model checkpoints and weights
- cloned upstream repositories under `third_party/`
- local conda environments

Therefore, a teacher can immediately inspect the code, reports and final result files, and can regenerate summary tables/figures from the included prediction CSVs. A complete from-scratch rerun of all three foundation models requires downloading the public dataset, downloading model checkpoints and installing/cloning the upstream model repositories.

## Repository Layout

```text
configs/               benchmark configuration
data/splits/           fixed seed-42 train/validation/test split
docs/                  project notes and result summaries
results/figures/       report-ready figures
results/metrics/       benchmark metrics, predictions and secondary analyses
scripts/               numbered runnable scripts
src/ica2/              reusable project code
requirements.txt       core Python requirements for non-model-specific scripts
UPLOAD_NOTES.md        notes on what is included/excluded from upload
```

## Main Results

The main benchmark table is in `results/metrics/benchmark_summary.md`.

| Method | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| SVD + logistic regression | 0.9858 | 0.9790 | 0.9419 | 0.9578 |
| scGPT embedding + logistic regression | 0.9711 | 0.9663 | 0.9152 | 0.9333 |
| SCimilarity embedding + logistic regression | 0.9715 | 0.8835 | 0.9144 | 0.8872 |
| scFoundation embedding + logistic regression | 0.9817 | 0.9654 | 0.9245 | 0.9412 |


## Core Setup

Create an environment and install the core dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set the model directory. If this is not set, the config defaults to `D:\cmml_models`.

```powershell
$env:CMML_MODEL_DIR = "D:\cmml_models"
```

The benchmark configuration is stored in `configs/benchmark.yaml`.

## What Can Be Reproduced Immediately

Because prediction files and metric CSVs are included, the final benchmark Markdown tables and secondary analysis figures can be regenerated without raw data or model weights:

```powershell
python scripts/12_render_benchmark_markdown.py
python scripts/13_run_secondary_analysis.py
python scripts/14_build_word_reports.py
```

These commands regenerate:

- `results/metrics/benchmark_summary.md`
- `results/metrics/per_batch_metrics.*`
- `results/metrics/per_class_f1_matrix.csv`
- confusion matrix CSVs
- report figures in `results/figures/`
- Word reports in `reports/`

## Reproducing the Baseline From Public Data

Download the public Open Problems pancreas h5ad file:

```powershell
python scripts/1_download_pancreas_dataset.py
```

Then inspect the dataset, recreate the split and rerun the SVD baseline:

```powershell
python scripts/4_inspect_pancreas_dataset.py
python scripts/5_create_pancreas_splits.py
python scripts/6_run_baseline_logreg.py
python scripts/12_render_benchmark_markdown.py
python scripts/13_run_secondary_analysis.py
```

This reproduces the traditional baseline using the downloaded dataset.

## Full Foundation-model Rerun

To rerun all three foundation models from scratch, additional external assets are required.

Clone the upstream repositories:

```powershell
New-Item -ItemType Directory -Force third_party
git clone https://github.com/bowang-lab/scGPT.git third_party/scGPT
git clone https://github.com/Genentech/scimilarity.git third_party/scimilarity
git clone https://github.com/biomap-research/scFoundation.git third_party/scFoundation
```

Install the model packages according to their upstream instructions. In the local project, separate environments were used for scGPT, SCimilarity and scFoundation because their dependencies differ.

Download model assets:

```powershell
.\scripts\2_download_and_extract_scimilarity.ps1
python scripts/3_download_scfoundation_model.py
```

The scGPT whole-human checkpoint is not distributed in this repository. Place `best_model.pt`, `args.json` and `vocab.json` under:

```text
%CMML_MODEL_DIR%/scgpt/whole-human/
```

The full benchmark workflow is:

```powershell
python scripts/8_run_scimilarity_linear_probe.py --embeddings-only
python scripts/9_run_scgpt_linear_probe.py --embeddings-only --max-length 512
python scripts/10_run_scfoundation_linear_probe.py --embeddings-only --max-genes 128

python scripts/11_run_cached_embedding_probe.py --method scimilarity
python scripts/11_run_cached_embedding_probe.py --method scgpt
python scripts/11_run_cached_embedding_probe.py --method scfoundation

python scripts/12_render_benchmark_markdown.py
python scripts/13_run_secondary_analysis.py
```

`scripts/11_run_cached_embedding_probe.py` checks both cell order and embedding metadata. If model paths or parameters such as `max_length` or `max_genes` do not match the current config, it raises an error rather than silently reusing stale embeddings.

## Important Limitations for Reproduction

- The full foundation-model rerun is not self-contained in this upload because model weights and upstream repositories are external.
- The local scFoundation workflow was run on CPU. A fresh upstream clone may require applying the same CPU-compatible loading behaviour described in `docs/known_constraints.md`.
- The uploaded result files are sufficient to verify the reported metrics and regenerate the report tables/figures, but not to recompute foundation-model embeddings without external assets.

## Script Order

1. `scripts/1_download_pancreas_dataset.py`
2. `scripts/4_inspect_pancreas_dataset.py`
3. `scripts/5_create_pancreas_splits.py`
4. `scripts/6_run_baseline_logreg.py`
5. `scripts/8_run_scimilarity_linear_probe.py`
6. `scripts/9_run_scgpt_linear_probe.py`
7. `scripts/10_run_scfoundation_linear_probe.py`
8. `scripts/11_run_cached_embedding_probe.py`
9. `scripts/12_render_benchmark_markdown.py`
10. `scripts/13_run_secondary_analysis.py`
