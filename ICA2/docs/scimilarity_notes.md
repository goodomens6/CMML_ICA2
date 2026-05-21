# SCimilarity Notes

## Local runtime adjustments

- The local `scimilarity` copy now imports the TileDB backend only when the TileDB kNN path is actually selected. This allows the HNSW path to run on Windows without requiring the TileDB vector-search DLL.
- The model archive stores annotation files at the model root, while the current library expects them under an `annotation/` subdirectory. The benchmark therefore passes relative filenames for the bundled root-level files.

## Evaluation note

SCimilarity predicts ontology-style labels such as `pancreatic A cell`, `type B pancreatic cell`, and `pancreatic stellate cell`, while the pancreas benchmark uses dataset labels such as `alpha`, `beta`, `activated_stellate`, and `quiescent_stellate`.

For fair reporting, preserve raw SCimilarity predictions and evaluate a separate ontology-harmonized label space:

- `pancreatic A cell` -> `alpha`
- `type B pancreatic cell` -> `beta`
- `pancreatic D cell` -> `delta`
- `pancreatic acinar cell` -> `acinar`
- `pancreatic ductal cell` -> `ductal`
- `pancreatic stellate cell` -> `stellate`
- dataset `activated_stellate` and `quiescent_stellate` -> `stellate`

This harmonized result should not be mixed directly with the fine-grained 14-class benchmark without stating the different label space.

## Main benchmark note

For the primary benchmark table, SCimilarity is evaluated as an embedding model: all cells are embedded once, then the shared multinomial logistic-regression linear probe is fit on the training split and scored on the held-out test split. The same downstream probe should be reused for scGPT and scFoundation so that the foundation-model comparison varies the embedding generator, not the classifier.
