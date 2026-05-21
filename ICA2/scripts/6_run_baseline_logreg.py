from pathlib import Path
import sys

import pandas as pd
import scanpy as sc
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ica2.evaluation.metrics import classification_summary, per_class_f1
from ica2.evaluation.label_harmonization import (
    harmonize_dataset_prediction,
    harmonize_true_label,
)
from ica2.config import load_benchmark_config, resolve_from_root
from ica2.models.linear_probe import build_embedding_linear_probe


CONFIG = load_benchmark_config(ROOT)
DATASET_CONFIG = CONFIG["dataset"]
SPLIT_CONFIG = CONFIG["split"]
METHOD_NAME = CONFIG["methods"]["baseline"]["name"]
DATASET_PATH = resolve_from_root(ROOT, DATASET_CONFIG["path"])
SPLIT_PATH = ROOT / "data" / "splits" / f"pancreas_seed{SPLIT_CONFIG['seed']}_splits.csv"
METRICS_PATH = ROOT / "results" / "metrics" / f"{METHOD_NAME}_metrics.csv"
PREDICTIONS_PATH = ROOT / "results" / "metrics" / f"{METHOD_NAME}_predictions.csv"
PER_CLASS_PATH = ROOT / "results" / "metrics" / f"{METHOD_NAME}_per_class_f1.csv"
BENCHMARK_PATH = ROOT / "results" / "metrics" / "benchmark_summary.csv"
HARMONIZED_METRICS_PATH = (
    ROOT / "results" / "metrics" / f"{METHOD_NAME}_harmonized_metrics.csv"
)
HARMONIZED_BENCHMARK_PATH = ROOT / "results" / "metrics" / "benchmark_harmonized_summary.csv"


def main() -> None:
    adata = sc.read_h5ad(DATASET_PATH)
    split_df = pd.read_csv(SPLIT_PATH)
    split_lookup = split_df.set_index("cell_id")

    ordered = split_lookup.loc[adata.obs_names]
    x = adata.layers[DATASET_CONFIG["normalized_layer"]]
    y = adata.obs[DATASET_CONFIG["label_key"]].astype(str).to_numpy()

    train_mask = ordered["split"].eq("train").to_numpy()
    validation_mask = ordered["split"].eq("validation").to_numpy()
    test_mask = ordered["split"].eq("test").to_numpy()

    shared_probe = build_embedding_linear_probe(random_state=SPLIT_CONFIG["seed"])
    model = Pipeline(
        steps=[
            ("svd", TruncatedSVD(n_components=50, random_state=SPLIT_CONFIG["seed"])),
            *shared_probe.steps,
        ]
    )
    model.fit(x[train_mask], y[train_mask])

    validation_pred = model.predict(x[validation_mask])
    test_pred = model.predict(x[test_mask])

    metrics = pd.DataFrame(
        [
            classification_summary(
                y[validation_mask],
                validation_pred,
                method=METHOD_NAME,
                split="validation",
            ),
            classification_summary(
                y[test_mask],
                test_pred,
                method=METHOD_NAME,
                split="test",
            ),
        ]
    )

    predictions = pd.DataFrame(
        {
            "cell_id": adata.obs_names[test_mask],
            "true_label": y[test_mask],
            "predicted_label": test_pred,
            "batch": adata.obs.loc[adata.obs_names[test_mask], DATASET_CONFIG["batch_key"]]
            .astype(str)
            .to_numpy(),
        }
    )
    per_class = per_class_f1(y[test_mask], test_pred)
    test_true_harmonized = [harmonize_true_label(label) for label in y[test_mask]]
    test_pred_harmonized = [
        harmonize_dataset_prediction(label) for label in test_pred
    ]
    harmonized_metrics = pd.DataFrame(
        [
            classification_summary(
                test_true_harmonized,
                test_pred_harmonized,
                method=METHOD_NAME,
                split="test_harmonized",
            )
        ]
    )

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    per_class.to_csv(PER_CLASS_PATH, index=False)
    harmonized_metrics.to_csv(HARMONIZED_METRICS_PATH, index=False)

    benchmark_rows = {
        "logistic_regression_svd50": metrics.loc[metrics["split"].eq("test")]
        .iloc[0]
        .to_dict(),
        "scgpt_embedding_logreg": {
            "method": "scgpt_embedding_logreg",
            "split": "test",
            "n_cells": int(test_mask.sum()),
            "accuracy": pd.NA,
            "macro_precision": pd.NA,
            "macro_recall": pd.NA,
            "macro_f1": pd.NA,
            "status": "pending",
            "notes": "",
        },
        "scimilarity_embedding_logreg": {
            "method": "scimilarity_embedding_logreg",
            "split": "test",
            "n_cells": int(test_mask.sum()),
            "accuracy": pd.NA,
            "macro_precision": pd.NA,
            "macro_recall": pd.NA,
            "macro_f1": pd.NA,
            "status": "pending",
            "notes": "",
        },
        "scfoundation_embedding_logreg": {
            "method": "scfoundation_embedding_logreg",
            "split": "test",
            "n_cells": int(test_mask.sum()),
            "accuracy": pd.NA,
            "macro_precision": pd.NA,
            "macro_recall": pd.NA,
            "macro_f1": pd.NA,
            "status": "pending",
            "notes": "",
        },
    }
    if BENCHMARK_PATH.exists():
        existing_benchmark = pd.read_csv(BENCHMARK_PATH)
        existing_rows = {
            row["method"]: row.to_dict()
            for _, row in existing_benchmark.iterrows()
        }
        for method, row in existing_rows.items():
            if method != METHOD_NAME and method in benchmark_rows:
                benchmark_rows[method] = row
    benchmark = pd.DataFrame(benchmark_rows.values())
    benchmark.to_csv(BENCHMARK_PATH, index=False)
    harmonized_rows = {
        METHOD_NAME: harmonized_metrics.iloc[0].to_dict(),
        "scimilarity": {
            "method": "scimilarity",
            "split": "test_harmonized",
            "n_cells": int(test_mask.sum()),
            "accuracy": pd.NA,
            "macro_precision": pd.NA,
            "macro_recall": pd.NA,
            "macro_f1": pd.NA,
            "status": "pending",
            "notes": "",
        },
    }
    if HARMONIZED_BENCHMARK_PATH.exists():
        existing_harmonized = pd.read_csv(HARMONIZED_BENCHMARK_PATH)
        existing_rows = {
            row["method"]: row.to_dict()
            for _, row in existing_harmonized.iterrows()
        }
        for method, row in existing_rows.items():
            if method != METHOD_NAME:
                harmonized_rows[method] = row
    harmonized_benchmark = pd.DataFrame(harmonized_rows.values())
    harmonized_benchmark.to_csv(HARMONIZED_BENCHMARK_PATH, index=False)

    print(metrics.to_string(index=False))
    print()
    print(f"wrote {METRICS_PATH}")
    print(f"wrote {PREDICTIONS_PATH}")
    print(f"wrote {PER_CLASS_PATH}")
    print(f"wrote {HARMONIZED_METRICS_PATH}")
    print(f"wrote {BENCHMARK_PATH}")
    print(f"wrote {HARMONIZED_BENCHMARK_PATH}")


if __name__ == "__main__":
    main()
