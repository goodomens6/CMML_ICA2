from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import scanpy as sc


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ica2.config import load_benchmark_config, resolve_from_root
from ica2.models.embedding_benchmark import run_embedding_benchmark
from ica2.models.embedding_cache import read_embedding_metadata


CONFIG = load_benchmark_config(ROOT)
DATASET_CONFIG = CONFIG["dataset"]
SPLIT_CONFIG = CONFIG["split"]
DATASET_PATH = resolve_from_root(ROOT, DATASET_CONFIG["path"])
SPLIT_PATH = ROOT / "data" / "splits" / f"pancreas_seed{SPLIT_CONFIG['seed']}_splits.csv"
BENCHMARK_PATH = ROOT / "results" / "metrics" / "benchmark_summary.csv"
SCFOUNDATION_GENE_INDEX_PATH = (
    ROOT / "third_party" / "scFoundation" / "model" / "OS_scRNA_gene_index.19264.tsv"
)


def scgpt_expected_metadata() -> dict[str, object]:
    scgpt_config = CONFIG["methods"]["scgpt"]
    return {
        "model": "scgpt_whole_human",
        "model_path": str(Path(scgpt_config["model_path"])),
        "counts_layer": DATASET_CONFIG["counts_layer"],
        "gene_col": scgpt_config["gene_col"],
        "max_length": int(scgpt_config["max_length"]),
        "sequence_policy": "cls_token_plus_top_expressed_genes",
        "value_policy": "per_cell_quantile_bins_51",
    }


def scimilarity_expected_metadata() -> dict[str, object]:
    scimilarity_config = CONFIG["methods"]["scimilarity"]
    return {
        "model": "scimilarity_annotation_model_v1",
        "model_path": str(Path(scimilarity_config["model_path"])),
        "counts_layer": DATASET_CONFIG["counts_layer"],
        "preprocessing": "align_dataset_then_lognorm_counts",
    }


def scfoundation_expected_metadata() -> dict[str, object]:
    scfoundation_config = CONFIG["methods"]["scfoundation"]
    return {
        "model": "scfoundation",
        "model_path": str(Path(scfoundation_config["model_path"])),
        "counts_layer": DATASET_CONFIG["counts_layer"],
        "gene_index": str(SCFOUNDATION_GENE_INDEX_PATH),
        "max_genes": int(scfoundation_config["max_genes"]),
        "tgthighres": str(scfoundation_config["tgthighres"]),
        "sequence_policy": "top_expressed_genes_plus_count_and_resolution_tokens",
        "pooling": "last_two_tokens_gene_max_gene_mean",
    }

METHODS = {
    "scgpt": {
        "method_name": CONFIG["methods"]["scgpt"]["linear_probe_name"],
        "embeddings_path": ROOT / "data" / "processed" / "scgpt_embeddings.npz",
        "expected_metadata": scgpt_expected_metadata(),
    },
    "scimilarity": {
        "method_name": CONFIG["methods"]["scimilarity"]["linear_probe_name"],
        "embeddings_path": ROOT / "data" / "processed" / "scimilarity_embeddings.npz",
        "expected_metadata": scimilarity_expected_metadata(),
    },
    "scfoundation": {
        "method_name": CONFIG["methods"]["scfoundation"]["linear_probe_name"],
        "embeddings_path": ROOT / "data" / "processed" / "scfoundation_embeddings.npz",
        "expected_metadata": scfoundation_expected_metadata(),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the shared linear probe on an existing foundation-model embedding cache."
    )
    parser.add_argument(
        "--method",
        choices=sorted(METHODS),
        required=True,
        help="Embedding cache to evaluate.",
    )
    return parser.parse_args()


def format_metadata_diff(
    stored_metadata: dict[str, object] | None,
    expected_metadata: dict[str, object],
) -> str:
    if stored_metadata is None:
        return "stored metadata is missing"
    keys = sorted(set(stored_metadata) | set(expected_metadata))
    differences = []
    for key in keys:
        stored_value = stored_metadata.get(key, "<missing>")
        expected_value = expected_metadata.get(key, "<missing>")
        if stored_value != expected_value:
            differences.append(
                f"{key}: stored={stored_value!r}, expected={expected_value!r}"
            )
    return "; ".join(differences)


def load_embeddings(
    path: Path,
    expected_cell_ids: np.ndarray,
    expected_metadata: dict[str, object],
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding cache: {path}")
    with np.load(path) as stored:
        stored_ids = stored["cell_ids"].astype(str)
        if not np.array_equal(stored_ids, expected_cell_ids.astype(str)):
            raise ValueError(f"Embedding cell order does not match dataset order: {path}")
        stored_metadata = read_embedding_metadata(path, stored)
        if stored_metadata != expected_metadata:
            raise ValueError(
                "Embedding metadata does not match the current benchmark configuration "
                f"for {path}: {format_metadata_diff(stored_metadata, expected_metadata)}"
            )
        return stored["embeddings"].copy()


def print_metadata(path: Path) -> None:
    metadata = read_embedding_metadata(path)
    print("embedding metadata:", json.dumps(metadata, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args()
    method_config = METHODS[args.method]
    embeddings_path = method_config["embeddings_path"]
    method_name = method_config["method_name"]

    adata = sc.read_h5ad(DATASET_PATH)
    split_df = pd.read_csv(SPLIT_PATH)
    split_lookup = split_df.set_index("cell_id")
    ordered = split_lookup.loc[adata.obs_names]
    embeddings = load_embeddings(
        embeddings_path,
        adata.obs_names.to_numpy(dtype=str),
        method_config["expected_metadata"],
    )

    metrics_path = ROOT / "results" / "metrics" / f"{method_name}_metrics.csv"
    predictions_path = ROOT / "results" / "metrics" / f"{method_name}_predictions.csv"
    per_class_path = ROOT / "results" / "metrics" / f"{method_name}_per_class_f1.csv"

    print_metadata(embeddings_path)
    metrics = run_embedding_benchmark(
        adata=adata,
        ordered_splits=ordered,
        embeddings=embeddings,
        method_name=method_name,
        batch_key=DATASET_CONFIG["batch_key"],
        benchmark_path=BENCHMARK_PATH,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        per_class_path=per_class_path,
        random_state=SPLIT_CONFIG["seed"],
    )

    print(metrics.to_string(index=False))
    print()
    print(f"read {embeddings_path}")
    print(f"wrote {metrics_path}")
    print(f"wrote {predictions_path}")
    print(f"wrote {per_class_path}")
    print(f"updated {BENCHMARK_PATH}")


if __name__ == "__main__":
    main()
