from pathlib import Path
import sys

import pandas as pd
import scanpy as sc
from scimilarity import CellAnnotation
from scimilarity.utils import align_dataset, lognorm_counts


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ica2.config import load_benchmark_config, resolve_from_root
from ica2.evaluation.label_harmonization import (
    harmonize_scimilarity_prediction,
    harmonize_true_label,
)
from ica2.evaluation.metrics import classification_summary, per_class_f1


CONFIG = load_benchmark_config(ROOT)
DATASET_CONFIG = CONFIG["dataset"]
SPLIT_CONFIG = CONFIG["split"]
SCIMILARITY_CONFIG = CONFIG["methods"]["scimilarity"]
METHOD_NAME = "scimilarity"
DATASET_PATH = resolve_from_root(ROOT, DATASET_CONFIG["path"])
SPLIT_PATH = ROOT / "data" / "splits" / f"pancreas_seed{SPLIT_CONFIG['seed']}_splits.csv"
PREDICTIONS_PATH = ROOT / "results" / "metrics" / "scimilarity_predictions.csv"
HARMONIZED_METRICS_PATH = ROOT / "results" / "metrics" / "scimilarity_harmonized_metrics.csv"
HARMONIZED_PER_CLASS_PATH = (
    ROOT / "results" / "metrics" / "scimilarity_harmonized_per_class_f1.csv"
)
HARMONIZED_BENCHMARK_PATH = ROOT / "results" / "metrics" / "benchmark_harmonized_summary.csv"


def main() -> None:
    adata = sc.read_h5ad(DATASET_PATH)
    split_df = pd.read_csv(SPLIT_PATH)
    split_lookup = split_df.set_index("cell_id")
    test_ids = split_lookup.index[split_lookup["split"].eq("test")]

    test_data = adata[test_ids].copy()
    test_data.X = test_data.layers[DATASET_CONFIG["counts_layer"]].copy()

    annotator = CellAnnotation(
        model_path=SCIMILARITY_CONFIG["model_path"],
        filenames={
            "knn": SCIMILARITY_CONFIG["knn_filename"],
            "celltype_labels": SCIMILARITY_CONFIG["celltype_labels_filename"],
        },
    )
    test_data = align_dataset(test_data, annotator.gene_order)
    test_data = lognorm_counts(test_data)
    embeddings = annotator.get_embeddings(test_data.X)
    predictions, _, _, stats = annotator.get_predictions_knn(
        embeddings,
        k=SCIMILARITY_CONFIG["prediction_k"],
        disable_progress=True,
    )

    true_labels = adata[test_ids].obs[DATASET_CONFIG["label_key"]].astype(str).to_numpy()
    raw_predictions = predictions.astype(str).to_numpy()
    harmonized_true = [harmonize_true_label(label) for label in true_labels]
    harmonized_predictions = [
        harmonize_scimilarity_prediction(label) for label in raw_predictions
    ]

    predictions_df = pd.DataFrame(
        {
            "cell_id": test_ids,
            "true_label": true_labels,
            "raw_prediction": raw_predictions,
            "harmonized_true_label": harmonized_true,
            "harmonized_prediction": harmonized_predictions,
            "batch": adata[test_ids].obs[DATASET_CONFIG["batch_key"]].astype(str).to_numpy(),
            "min_dist": stats["min_dist"].to_numpy(),
            "vs_all": stats["vsAll"].to_numpy(),
        }
    )
    harmonized_metrics = pd.DataFrame(
        [
            classification_summary(
                harmonized_true,
                harmonized_predictions,
                method=METHOD_NAME,
                split="test_harmonized",
            )
        ]
    )
    harmonized_per_class = per_class_f1(harmonized_true, harmonized_predictions)

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    harmonized_metrics.to_csv(HARMONIZED_METRICS_PATH, index=False)
    harmonized_per_class.to_csv(HARMONIZED_PER_CLASS_PATH, index=False)

    harmonized_benchmark = pd.read_csv(HARMONIZED_BENCHMARK_PATH)
    if "notes" not in harmonized_benchmark.columns:
        harmonized_benchmark["notes"] = ""
    else:
        harmonized_benchmark["notes"] = harmonized_benchmark["notes"].fillna("").astype("object")
    harmonized_benchmark.loc[
        harmonized_benchmark["method"].eq(METHOD_NAME),
        [
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "status",
            "notes",
        ],
    ] = [
        harmonized_metrics.loc[0, "accuracy"],
        harmonized_metrics.loc[0, "macro_precision"],
        harmonized_metrics.loc[0, "macro_recall"],
        harmonized_metrics.loc[0, "macro_f1"],
        "complete",
        "ontology-harmonized label space",
    ]
    harmonized_benchmark.to_csv(HARMONIZED_BENCHMARK_PATH, index=False)

    print(harmonized_metrics.to_string(index=False))
    print()
    print(predictions_df["raw_prediction"].value_counts().head(20).to_string())
    print()
    print(f"wrote {PREDICTIONS_PATH}")
    print(f"wrote {HARMONIZED_METRICS_PATH}")
    print(f"wrote {HARMONIZED_PER_CLASS_PATH}")
    print(f"updated {HARMONIZED_BENCHMARK_PATH}")


if __name__ == "__main__":
    main()
