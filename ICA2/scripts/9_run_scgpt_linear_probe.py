from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ica2.config import load_benchmark_config, resolve_from_root
from ica2.models.embedding_cache import load_embedding_cache, save_embedding_cache
from ica2.models.embedding_benchmark import run_embedding_benchmark


CONFIG = load_benchmark_config(ROOT)
DATASET_CONFIG = CONFIG["dataset"]
SPLIT_CONFIG = CONFIG["split"]
SCGPT_CONFIG = CONFIG["methods"]["scgpt"]
METHOD_NAME = SCGPT_CONFIG["linear_probe_name"]
DATASET_PATH = resolve_from_root(ROOT, DATASET_CONFIG["path"])
SPLIT_PATH = ROOT / "data" / "splits" / f"pancreas_seed{SPLIT_CONFIG['seed']}_splits.csv"
EMBEDDINGS_PATH = ROOT / "data" / "processed" / "scgpt_embeddings.npz"
METRICS_PATH = ROOT / "results" / "metrics" / f"{METHOD_NAME}_metrics.csv"
PREDICTIONS_PATH = ROOT / "results" / "metrics" / f"{METHOD_NAME}_predictions.csv"
PER_CLASS_PATH = ROOT / "results" / "metrics" / f"{METHOD_NAME}_per_class_f1.csv"
BENCHMARK_PATH = ROOT / "results" / "metrics" / "benchmark_summary.csv"


def scgpt_embedding_metadata(max_length: int) -> dict[str, object]:
    return {
        "model": "scgpt_whole_human",
        "model_path": str(Path(SCGPT_CONFIG["model_path"])),
        "counts_layer": DATASET_CONFIG["counts_layer"],
        "gene_col": SCGPT_CONFIG["gene_col"],
        "max_length": int(max_length),
        "sequence_policy": "cls_token_plus_top_expressed_genes",
        "value_policy": "per_cell_quantile_bins_51",
    }


def compute_embeddings(
    adata: sc.AnnData,
    *,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    from scgpt.model import TransformerModel
    from scgpt.utils import load_pretrained

    model_input = adata.copy()
    counts = model_input.layers[DATASET_CONFIG["counts_layer"]].tocsr()
    model_dir = Path(SCGPT_CONFIG["model_path"])
    vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
    with (model_dir / "args.json").open("r", encoding="utf-8") as handle:
        model_configs = json.load(handle)

    gene_names = model_input.var[SCGPT_CONFIG["gene_col"]].astype(str).to_numpy()
    keep_genes = np.array([gene in vocab for gene in gene_names])
    counts = counts[:, keep_genes]
    gene_ids = np.array([vocab[gene] for gene in gene_names[keep_genes]], dtype=np.int64)

    pad_token_id = vocab[model_configs["pad_token"]]
    cls_token_id = vocab["<cls>"]
    pad_value = model_configs["pad_value"]
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_configs["embsize"],
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        nlayers_cls=model_configs["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=model_configs["dropout"],
        pad_token=model_configs["pad_token"],
        pad_value=pad_value,
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        use_fast_transformer=False,
        pre_norm=False,
    )
    print(
        f"scGPT matched {int(keep_genes.sum())}/{len(keep_genes)} genes; "
        f"embedding {model_input.n_obs} cells with max_length={max_length}, "
        f"batch_size={batch_size}",
        flush=True,
    )
    load_pretrained(
        model,
        torch.load(model_dir / "best_model.pt", map_location="cpu"),
        verbose=False,
    )
    model.eval()

    embeddings = np.zeros((model_input.n_obs, model_configs["embsize"]), dtype=np.float32)
    for batch_start in range(0, model_input.n_obs, batch_size):
        batch_end = min(batch_start + batch_size, model_input.n_obs)
        if batch_start == 0 or (batch_start // batch_size) % 10 == 0:
            print(f"embedding cells {batch_start}-{batch_end} / {model_input.n_obs}", flush=True)
        gene_batch = np.full((batch_end - batch_start, max_length), pad_token_id, dtype=np.int64)
        value_batch = np.full((batch_end - batch_start, max_length), pad_value, dtype=np.float32)
        gene_batch[:, 0] = cls_token_id
        value_batch[:, 0] = pad_value

        for row_offset, cell_index in enumerate(range(batch_start, batch_end)):
            row = counts[cell_index]
            if sparse.issparse(row):
                row = row.tocsr()
                local_indices = row.indices
                values = row.data.astype(np.float32, copy=False)
            else:
                values = np.asarray(row, dtype=np.float32)
                local_indices = np.flatnonzero(values)
                values = values[local_indices]
            if len(local_indices) == 0:
                continue
            n_keep = min(max_length - 1, len(local_indices))
            if len(local_indices) > n_keep:
                top_positions = np.argpartition(values, -n_keep)[-n_keep:]
                order = top_positions[np.argsort(values[top_positions])[::-1]]
            else:
                order = np.argsort(values)[::-1]
            selected_values = values[order]
            selected_gene_ids = gene_ids[local_indices[order]]
            value_batch[row_offset, 1 : n_keep + 1] = bin_expression_values(selected_values)
            gene_batch[row_offset, 1 : n_keep + 1] = selected_gene_ids

        gene_tensor = torch.from_numpy(gene_batch).long()
        value_tensor = torch.from_numpy(value_batch).float()
        padding_mask = gene_tensor.eq(pad_token_id)
        with torch.no_grad():
            encoded = model._encode(gene_tensor, value_tensor, padding_mask)
        batch_embeddings = encoded[:, 0, :].detach().cpu().numpy()
        norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
        embeddings[batch_start:batch_end] = batch_embeddings / np.maximum(norms, 1e-12)

    save_embedding_cache(
        embeddings_path=EMBEDDINGS_PATH,
        embeddings=embeddings,
        cell_ids=model_input.obs_names.to_numpy(dtype=str),
        metadata=scgpt_embedding_metadata(max_length),
    )
    return embeddings


def bin_expression_values(values: np.ndarray, n_bins: int = 51) -> np.ndarray:
    if values.size == 0 or values.max() == 0:
        return np.zeros(values.shape, dtype=np.float32)
    bins = np.quantile(values, np.linspace(0, 1, n_bins - 1))
    return np.digitize(values, bins).astype(np.float32)


def load_or_compute_embeddings(
    adata: sc.AnnData,
    *,
    max_length: int,
    batch_size: int,
    force: bool,
    compute_if_missing: bool,
) -> np.ndarray:
    current_ids = adata.obs_names.to_numpy(dtype=str)
    expected_metadata = scgpt_embedding_metadata(max_length)
    if not force:
        cached = load_embedding_cache(
            embeddings_path=EMBEDDINGS_PATH,
            expected_cell_ids=current_ids,
            expected_metadata=expected_metadata,
        )
        if cached is not None:
            return cached

    if not compute_if_missing:
        raise RuntimeError(
            f"No matching scGPT embedding cache found for max_length={max_length}. "
            "Run without --probe-only, or use --force to recompute."
        )
    return compute_embeddings(adata, max_length=max_length, batch_size=batch_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate scGPT embeddings and run the shared linear-probe benchmark."
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=int(SCGPT_CONFIG["max_length"]),
        help="Total scGPT sequence length, including the <cls> token.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(SCGPT_CONFIG["batch_size"]),
        help="CPU inference batch size.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute embeddings even when a matching cache exists.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--embeddings-only",
        action="store_true",
        help="Stop after writing the scGPT embedding cache.",
    )
    mode.add_argument(
        "--probe-only",
        action="store_true",
        help="Use an existing matching embedding cache and only rerun the linear probe.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.force and args.probe_only:
        raise ValueError("--force and --probe-only cannot be used together.")

    adata = sc.read_h5ad(DATASET_PATH)
    split_df = pd.read_csv(SPLIT_PATH)
    split_lookup = split_df.set_index("cell_id")
    ordered = split_lookup.loc[adata.obs_names]
    embeddings = load_or_compute_embeddings(
        adata,
        max_length=args.max_length,
        batch_size=args.batch_size,
        force=args.force,
        compute_if_missing=not args.probe_only,
    )
    if args.embeddings_only:
        print(f"wrote {EMBEDDINGS_PATH}")
        return

    metrics = run_embedding_benchmark(
        adata=adata,
        ordered_splits=ordered,
        embeddings=embeddings,
        method_name=METHOD_NAME,
        batch_key=DATASET_CONFIG["batch_key"],
        benchmark_path=BENCHMARK_PATH,
        metrics_path=METRICS_PATH,
        predictions_path=PREDICTIONS_PATH,
        per_class_path=PER_CLASS_PATH,
        random_state=SPLIT_CONFIG["seed"],
    )

    print(metrics.to_string(index=False))
    print()
    print(f"wrote {EMBEDDINGS_PATH}")
    print(f"wrote {METRICS_PATH}")
    print(f"wrote {PREDICTIONS_PATH}")
    print(f"wrote {PER_CLASS_PATH}")
    print(f"updated {BENCHMARK_PATH}")


if __name__ == "__main__":
    main()
