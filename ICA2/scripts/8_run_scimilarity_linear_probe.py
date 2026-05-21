from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import scanpy as sc


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ica2.config import load_benchmark_config, resolve_from_root
from ica2.models.embedding_cache import load_embedding_cache, save_embedding_cache
from ica2.models.embedding_benchmark import run_embedding_benchmark


CONFIG = load_benchmark_config(ROOT)
DATASET_CONFIG = CONFIG["dataset"]
SPLIT_CONFIG = CONFIG["split"]
SCIMILARITY_CONFIG = CONFIG["methods"]["scimilarity"]
METHOD_NAME = SCIMILARITY_CONFIG["linear_probe_name"]
DATASET_PATH = resolve_from_root(ROOT, DATASET_CONFIG["path"])
SPLIT_PATH = ROOT / "data" / "splits" / f"pancreas_seed{SPLIT_CONFIG['seed']}_splits.csv"
EMBEDDINGS_PATH = ROOT / "data" / "processed" / "scimilarity_embeddings.npz"
METRICS_PATH = ROOT / "results" / "metrics" / f"{METHOD_NAME}_metrics.csv"
PREDICTIONS_PATH = ROOT / "results" / "metrics" / f"{METHOD_NAME}_predictions.csv"
PER_CLASS_PATH = ROOT / "results" / "metrics" / f"{METHOD_NAME}_per_class_f1.csv"
BENCHMARK_PATH = ROOT / "results" / "metrics" / "benchmark_summary.csv"


def scimilarity_embedding_metadata() -> dict[str, object]:
    return {
        "model": "scimilarity_annotation_model_v1",
        "model_path": str(Path(SCIMILARITY_CONFIG["model_path"])),
        "counts_layer": DATASET_CONFIG["counts_layer"],
        "preprocessing": "align_dataset_then_lognorm_counts",
    }


def compute_embeddings(adata: sc.AnnData) -> np.ndarray:
    from scimilarity import CellEmbedding
    from scimilarity.utils import align_dataset, lognorm_counts

    embedding_model = CellEmbedding(model_path=SCIMILARITY_CONFIG["model_path"])
    model_input = adata.copy()
    model_input.X = model_input.layers[DATASET_CONFIG["counts_layer"]].copy()
    model_input = align_dataset(model_input, embedding_model.gene_order)
    model_input = lognorm_counts(model_input)
    embeddings = embedding_model.get_embeddings(model_input.X)

    save_embedding_cache(
        embeddings_path=EMBEDDINGS_PATH,
        embeddings=embeddings,
        cell_ids=adata.obs_names.to_numpy(dtype=str),
        metadata=scimilarity_embedding_metadata(),
    )
    return embeddings


def load_or_compute_embeddings(
    adata: sc.AnnData,
    *,
    force: bool,
    compute_if_missing: bool,
) -> np.ndarray:
    current_ids = adata.obs_names.to_numpy(dtype=str)
    if not force:
        cached = load_embedding_cache(
            embeddings_path=EMBEDDINGS_PATH,
            expected_cell_ids=current_ids,
            expected_metadata=scimilarity_embedding_metadata(),
        )
        if cached is not None:
            return cached

    if not compute_if_missing:
        raise RuntimeError(
            "No matching SCimilarity embedding cache found. "
            "Run without --probe-only, or use --force to recompute."
        )
    return compute_embeddings(adata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SCimilarity embeddings and run the shared linear-probe benchmark."
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
        help="Stop after writing the SCimilarity embedding cache.",
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
