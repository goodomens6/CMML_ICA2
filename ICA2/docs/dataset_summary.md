# Human Pancreas Dataset Summary

## Local file

`data/raw/human_pancreas_openproblems_v1_log_cp10k.h5ad`

## Shape

- 16,382 cells
- 18,771 genes

## Main fields

- `obs`: `tech`, `celltype`, `size_factors`, `n_counts`, `cell_type`, `batch`
- `var`: `n_cells`, `feature_name`, `hvg`, `hvg_score`
- `layers`: `counts`, `log_normalized`, `normalized`
- `obsm`: `X_pca`
- `obsp`: `knn_connectivities`, `knn_distances`
- `varm`: `pca_loadings`

## Cell-type counts

| Cell type | Cells |
| --- | ---: |
| acinar | 1,669 |
| activated_stellate | 464 |
| alpha | 5,493 |
| beta | 4,169 |
| delta | 1,055 |
| ductal | 2,142 |
| endothelial | 313 |
| epsilon | 32 |
| gamma | 699 |
| macrophage | 79 |
| mast | 42 |
| quiescent_stellate | 193 |
| schwann | 25 |
| t_cell | 7 |

## Batch counts

| Batch | Cells |
| --- | ---: |
| celseq | 1,004 |
| celseq2 | 2,285 |
| fluidigmc1 | 638 |
| inDrop1 | 1,937 |
| inDrop2 | 1,724 |
| inDrop3 | 3,605 |
| inDrop4 | 1,303 |
| smarter | 1,492 |
| smartseq2 | 2,394 |

## Analysis implications

- The dataset is strongly imbalanced across cell types.
- `epsilon`, `schwann`, and especially `t_cell` are rare classes, so macro F1 and per-class F1 are more informative than accuracy alone.
- The nine batches make the dataset suitable for a secondary robustness analysis on batch effects.
