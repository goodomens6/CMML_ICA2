# ICA2 Experiment Design

## Core question

How well do three single-cell foundation models perform on the same cell type annotation task, compared with a traditional baseline, on a dataset with known batch structure?

## Dataset

- Dataset: Human pancreas from Open Problems in Single-Cell Analysis
- Local file: `data/raw/human_pancreas_openproblems_v1_log_cp10k.h5ad`
- Organism: human
- Task labels: `obs["cell_type"]`
- Batch variable: `obs["batch"]`
- Main layers: raw counts in `layers["counts"]`, baseline input in `layers["log_normalized"]`

## Primary experiment

1. Use the Human pancreas dataset as the shared benchmark dataset.
2. Create one fixed label-stratified train/validation/test split and reuse it for every method.
3. Train one traditional baseline on the training split:
   - Recommended first baseline: train-only TruncatedSVD followed by the same shared logistic-regression linear probe used for the foundation-model embeddings.
4. Evaluate three foundation-model workflows on the same test split:
   - scGPT embedding + shared logistic-regression linear probe
   - SCimilarity embedding + shared logistic-regression linear probe
   - scFoundation embedding + shared logistic-regression linear probe
5. Report:
   - accuracy
   - macro precision
   - macro recall
   - macro F1
   - per-cell-type F1
   - confusion matrix

SCimilarity emits ontology-style labels that do not exactly match the pancreas dataset's fine-grained labels. Preserve its raw predictions, and report it in a separate ontology-harmonized benchmark table rather than mixing those scores directly into the original 14-class table.

## Secondary analysis

Use the existing batch annotations for within-split robustness analysis:

1. Evaluate each method separately within each batch.
2. Compare macro F1 across batches.
3. Identify cell types whose performance changes most across batches.

This secondary analysis supports discussion of batch-associated performance differences within the same random split. It is not a cross-batch generalization claim.

If the project later needs a true cross-batch generalization experiment, add a separate leave-one-batch-out split set rather than reusing the current random split.

## Preprocessing plan

1. Keep raw counts for methods that require counts.
2. Fit baseline dimensionality reduction on the training split only, then transform validation and test cells.
3. For the three foundation models, keep the downstream classifier fixed and vary only the embedding generator.
4. Preserve a single label vocabulary across all supervised methods.
5. Record all preprocessing choices in `reports/supporting/supplementary_methods.md`.

For CPU-feasible scGPT inference on this machine, the whole-human model is run on the top 511 expressed genes per cell plus the `<cls>` token (`max_length=512`). This setting remains deterministic and tractable without changing the downstream probe.

For CPU-feasible scFoundation inference on this machine, the local adapter follows the official cell-embedding pooling scheme but keeps the top 128 expressed genes per cell before adding the two scFoundation count/resolution tokens. The full untruncated CPU workflow was validated on small inputs but is too memory- and time-intensive for the full benchmark.

Embedding caches are paired with `.meta.json` sidecar files recording the model path and input-size parameters. If `max_length` or `max_genes` changes, the corresponding embedding script must be rerun with `--force` so the benchmark does not silently reuse an incompatible cache.

## Figure plan

Main report:

1. Figure 1: grouped bar chart comparing accuracy and macro F1 for all methods.
2. Figure 2: per-batch macro F1 heatmap or per-cell-type F1 heatmap.

Supporting material:

1. Confusion matrices.
2. Class distribution plot.
3. Batch distribution plot.
4. Optional UMAP colored by cell type and batch.

## Decision rules

- Use macro F1 as the main score because class imbalance is expected.
- Use the same held-out test split for all methods.
- Treat the current split as label-stratified random evaluation, not leave-one-batch-out evaluation.
- If one foundation model cannot run end-to-end on the local machine, document the failure mode clearly and keep the benchmark protocol unchanged.

## Current outputs

1. Main benchmark table: `results/metrics/benchmark_summary.md`
2. Per-batch metrics: `results/metrics/per_batch_metrics.md`
3. Per-cell-type F1 matrix: `results/metrics/per_class_f1_matrix.csv`
4. Main benchmark figure: `results/figures/benchmark_accuracy_macro_f1.png`
5. Batch robustness figure: `results/figures/per_batch_macro_f1_heatmap.png`
6. Supporting per-cell-type figure: `results/figures/per_class_f1_heatmap.png`

## Immediate next steps

1. Turn `reports/main/report_outline.md` into the final 1000-word report.
2. Use the benchmark bar chart and per-batch heatmap as the two main figures.
3. Put the per-cell-type heatmap and supplementary methods in supporting material.
4. Optionally add a leave-one-batch-out experiment only if the report needs a stronger cross-batch generalization claim.
