# Results Summary

## Primary Benchmark

All supervised methods use the same train/validation/test split and the same downstream multinomial logistic-regression probe. The comparison therefore isolates the representation used by the classifier as much as possible.

| Method | Accuracy | Macro F1 | Interpretation |
| --- | ---: | ---: | --- |
| SVD + logreg | 0.9858 | 0.9578 | Strongest overall baseline on this dataset. |
| scFoundation + logreg | 0.9817 | 0.9412 | Best foundation-model embedding in the current local benchmark. |
| scGPT + logreg | 0.9711 | 0.9333 | Competitive, but below the SVD baseline and scFoundation. |
| SCimilarity + logreg | 0.9715 | 0.8872 | Good accuracy but weaker macro F1, driven by rare-cell performance. |

## Secondary Patterns

The per-batch analysis shows that `fluidigmc1` is the hardest batch for every method. This supports a cautious discussion of batch-associated robustness, but it is not a leave-one-batch-out generalization experiment because the current split is label-stratified random.

The per-cell-type analysis shows that most common endocrine and exocrine labels are classified well by all methods. The largest weaknesses are concentrated in smaller or more ambiguous classes:

- SCimilarity: `epsilon` and `t_cell`
- scGPT: `schwann`, with lower scores for some endocrine subtypes
- scFoundation: `mast` and `epsilon`

## Figure Recommendation

Use these two figures in the main report:

1. `results/figures/benchmark_accuracy_macro_f1.png`: primary accuracy and macro F1 comparison.
2. `results/figures/per_batch_macro_f1_heatmap.png`: batch-associated performance differences.

Keep `results/figures/per_class_f1_heatmap.png` for supporting material. It is useful, but too detailed for one of only two main report figures.

## Report Claim

The current result does not show that foundation-model embeddings automatically beat a carefully controlled classical baseline on this pancreas annotation task. Instead, the cleaner claim is that foundation models produce competitive embeddings under one shared probe, with scFoundation closest to the SVD baseline, while SCimilarity's ontology-oriented pretraining appears less aligned with the fine-grained 14-class pancreas label space.

## Limitations To State

- The split is label-stratified random, not leave-one-batch-out.
- scGPT and scFoundation were run with deterministic top-expressed-gene truncation for CPU-feasible inference.
- SCimilarity zero-shot ontology labels are reported separately from the supervised 14-class benchmark.
