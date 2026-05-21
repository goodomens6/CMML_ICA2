# Model Input Truncation Notes

## scGPT

Current setting: `max_length=512`.

For each cell, the script reads raw counts from `layers["counts"]`, keeps only genes present in the scGPT vocabulary, then creates one sequence:

1. Position 0 is the `<cls>` token.
2. Positions 1-511 are the 511 genes with the largest expression values in that cell.
3. If a cell has fewer than 511 matched expressed genes, the rest of the sequence is padded.
4. Expression values for the selected genes are converted into per-cell quantile bins before being passed to scGPT.
5. The final cell embedding is the normalized hidden state of the `<cls>` token.

A longer sensitivity run was tested separately and did not improve held-out macro F1 in this benchmark. The main workflow therefore uses `max_length=512`, which is the CPU-feasible setting used in the primary comparison.

Command:

```powershell
python .\scripts\9_run_scgpt_linear_probe.py --embeddings-only --force --max-length 512
python .\scripts\11_run_cached_embedding_probe.py --method scgpt
```

## scFoundation

Current setting: `max_genes=128`.

For each cell, the script aligns raw counts to the scFoundation 19,264-gene index, performs log-normalization, then:

1. Keeps the 128 genes with the largest log-normalized expression values.
2. Sets other gene inputs to zero.
3. Adds the two scFoundation special count/resolution tokens.
4. Runs the scFoundation encoder on CPU.
5. Builds the cell embedding by concatenating the last two special-token embeddings, max-pooled gene embeddings, and mean-pooled gene embeddings.

The full nonzero-gene version is primarily a memory problem, not just a time problem. A batch of 16 cells attempted to allocate roughly 20 GB, so a full run would require much smaller batches and would likely be impractical on this machine. A more realistic sensitivity check is to try a larger cap such as `max_genes=256` or `512`.

Command:

```powershell
python .\scripts\10_run_scfoundation_linear_probe.py --embeddings-only --force --max-genes 256
python .\scripts\11_run_cached_embedding_probe.py --method scfoundation
```

## Effect On Results

These settings can affect results because excluded genes cannot contribute to the foundation-model embedding. The current benchmark is still internally fair because every cell in a method uses the same deterministic rule, and all methods share the same downstream classifier and split. In the report, these settings should be described as CPU-feasible local inference settings rather than as the maximum possible performance of each pretrained model.
