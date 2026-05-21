from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "scFoundation" / "model"))

from load import gatherData, load_model_frommmf

from ica2.config import load_benchmark_config, resolve_from_root
from ica2.models.embedding_cache import load_embedding_cache, save_embedding_cache
from ica2.models.embedding_benchmark import run_embedding_benchmark


CONFIG = load_benchmark_config(ROOT)
DATASET_CONFIG = CONFIG["dataset"]
SPLIT_CONFIG = CONFIG["split"]
SCFOUNDATION_CONFIG = CONFIG["methods"]["scfoundation"]
METHOD_NAME = SCFOUNDATION_CONFIG["linear_probe_name"]
DATASET_PATH = resolve_from_root(ROOT, DATASET_CONFIG["path"])
SPLIT_PATH = ROOT / "data" / "splits" / f"pancreas_seed{SPLIT_CONFIG['seed']}_splits.csv"
EMBEDDINGS_PATH = ROOT / "data" / "processed" / "scfoundation_embeddings.npz"
METRICS_PATH = ROOT / "results" / "metrics" / f"{METHOD_NAME}_metrics.csv"
PREDICTIONS_PATH = ROOT / "results" / "metrics" / f"{METHOD_NAME}_predictions.csv"
PER_CLASS_PATH = ROOT / "results" / "metrics" / f"{METHOD_NAME}_per_class_f1.csv"
BENCHMARK_PATH = ROOT / "results" / "metrics" / "benchmark_summary.csv"
SCFOUNDATION_MODEL_DIR = ROOT / "third_party" / "scFoundation" / "model"
GENE_INDEX_PATH = SCFOUNDATION_MODEL_DIR / "OS_scRNA_gene_index.19264.tsv"


def scfoundation_embedding_metadata(max_genes: int) -> dict[str, object]:
    return {
        "model": "scfoundation",
        "model_path": str(Path(SCFOUNDATION_CONFIG["model_path"])),
        "counts_layer": DATASET_CONFIG["counts_layer"],
        "gene_index": str(GENE_INDEX_PATH),
        "max_genes": int(max_genes),
        "tgthighres": str(SCFOUNDATION_CONFIG["tgthighres"]),
        "sequence_policy": "top_expressed_genes_plus_count_and_resolution_tokens",
        "pooling": "last_two_tokens_gene_max_gene_mean",
    }


def align_counts_to_scfoundation(adata: sc.AnnData) -> np.ndarray:
    gene_list = pd.read_csv(GENE_INDEX_PATH, sep="\t")["gene_name"].astype(str).tolist()
    source = {gene: index for index, gene in enumerate(adata.var_names.astype(str))}
    source_indices = np.array([source.get(gene, -1) for gene in gene_list], dtype=np.int64)
    counts = adata.layers[DATASET_CONFIG["counts_layer"]]
    if sparse.issparse(counts):
        counts = counts.toarray()
    counts = np.asarray(counts, dtype=np.float32)
    aligned = np.zeros((adata.n_obs, len(gene_list)), dtype=np.float32)
    matched = source_indices >= 0
    aligned[:, matched] = counts[:, source_indices[matched]]
    print(f"scFoundation matched {int(matched.sum())}/{len(gene_list)} genes", flush=True)
    return aligned


def build_model_input(aligned_counts: np.ndarray, *, max_genes: int) -> torch.Tensor:
    total = aligned_counts.sum(axis=1, keepdims=True)
    lognorm = np.log1p(aligned_counts / np.maximum(total, 1e-12) * 1e4).astype(np.float32)
    for row_index in range(lognorm.shape[0]):
        nonzero = np.flatnonzero(lognorm[row_index])
        if len(nonzero) <= max_genes:
            continue
        keep = nonzero[np.argpartition(lognorm[row_index, nonzero], -max_genes)[-max_genes:]]
        keep_mask = np.zeros(lognorm.shape[1], dtype=bool)
        keep_mask[keep] = True
        lognorm[row_index, ~keep_mask] = 0

    target_token = float(str(SCFOUNDATION_CONFIG["tgthighres"])[1:])
    total_token = np.log10(np.maximum(total, 1e-12)).astype(np.float32)
    return torch.from_numpy(
        np.concatenate(
            [
                lognorm,
                np.full((aligned_counts.shape[0], 1), target_token, dtype=np.float32),
                total_token,
            ],
            axis=1,
        )
    )


def compute_embeddings(
    adata: sc.AnnData,
    *,
    max_genes: int,
    batch_size: int,
) -> np.ndarray:
    aligned_counts = align_counts_to_scfoundation(adata)
    model_input = build_model_input(aligned_counts, max_genes=max_genes)
    model, model_config = load_model_frommmf(
        SCFOUNDATION_CONFIG["model_path"],
        "cell",
        device="cpu",
    )
    model.eval()

    data_gene_ids = torch.arange(model_input.shape[1]).repeat(batch_size, 1)
    embeddings = np.zeros((adata.n_obs, 3072), dtype=np.float32)
    for batch_start in range(0, adata.n_obs, batch_size):
        batch_end = min(batch_start + batch_size, adata.n_obs)
        if batch_start == 0 or (batch_start // batch_size) % 10 == 0:
            print(f"embedding cells {batch_start}-{batch_end} / {adata.n_obs}", flush=True)
        batch = model_input[batch_start:batch_end]
        labels = batch > 0
        ids = data_gene_ids[: batch_end - batch_start]
        x, x_padding = gatherData(batch, labels, model_config["pad_token_id"])
        position_gene_ids, _ = gatherData(ids, labels, model_config["pad_token_id"])

        with torch.no_grad():
            x = model.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)
            x = x + model.pos_emb(position_gene_ids)
            gene_embeddings = model.encoder(x, x_padding)
            cell_embeddings = torch.concat(
                [
                    gene_embeddings[:, -1, :],
                    gene_embeddings[:, -2, :],
                    torch.max(gene_embeddings[:, :-2, :], dim=1)[0],
                    torch.mean(gene_embeddings[:, :-2, :], dim=1),
                ],
                dim=1,
            )
        embeddings[batch_start:batch_end] = cell_embeddings.detach().cpu().numpy()

    save_embedding_cache(
        embeddings_path=EMBEDDINGS_PATH,
        embeddings=embeddings,
        cell_ids=adata.obs_names.to_numpy(dtype=str),
        metadata=scfoundation_embedding_metadata(max_genes),
    )
    return embeddings


def load_or_compute_embeddings(
    adata: sc.AnnData,
    *,
    max_genes: int,
    batch_size: int,
    force: bool,
    compute_if_missing: bool,
) -> np.ndarray:
    current_ids = adata.obs_names.to_numpy(dtype=str)
    expected_metadata = scfoundation_embedding_metadata(max_genes)
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
            f"No matching scFoundation embedding cache found for max_genes={max_genes}. "
            "Run without --probe-only, or use --force to recompute."
        )
    return compute_embeddings(adata, max_genes=max_genes, batch_size=batch_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate scFoundation embeddings and run the shared linear-probe benchmark."
    )
    parser.add_argument(
        "--max-genes",
        type=int,
        default=int(SCFOUNDATION_CONFIG["max_genes"]),
        help="Maximum non-special genes retained per cell after log-normalization.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(SCFOUNDATION_CONFIG["batch_size"]),
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
        help="Stop after writing the scFoundation embedding cache.",
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
        max_genes=args.max_genes,
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
