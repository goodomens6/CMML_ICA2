from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ica2.evaluation.metrics import classification_summary, per_class_f1
from ica2.models.linear_probe import build_embedding_linear_probe


def run_embedding_benchmark(
    *,
    adata,
    ordered_splits: pd.DataFrame,
    embeddings: np.ndarray,
    method_name: str,
    batch_key: str,
    benchmark_path: Path,
    metrics_path: Path,
    predictions_path: Path,
    per_class_path: Path,
    random_state: int,
) -> pd.DataFrame:
    y = adata.obs["cell_type"].astype(str).to_numpy()
    train_mask = ordered_splits["split"].eq("train").to_numpy()
    validation_mask = ordered_splits["split"].eq("validation").to_numpy()
    test_mask = ordered_splits["split"].eq("test").to_numpy()

    model = build_embedding_linear_probe(random_state=random_state)
    model.fit(embeddings[train_mask], y[train_mask])

    validation_pred = model.predict(embeddings[validation_mask])
    test_pred = model.predict(embeddings[test_mask])
    metrics = pd.DataFrame(
        [
            classification_summary(
                y[validation_mask],
                validation_pred,
                method=method_name,
                split="validation",
            ),
            classification_summary(
                y[test_mask],
                test_pred,
                method=method_name,
                split="test",
            ),
        ]
    )
    predictions = pd.DataFrame(
        {
            "cell_id": adata.obs_names[test_mask],
            "true_label": y[test_mask],
            "predicted_label": test_pred,
            "batch": adata.obs.loc[adata.obs_names[test_mask], batch_key]
            .astype(str)
            .to_numpy(),
        }
    )
    per_class = per_class_f1(y[test_mask], test_pred)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    per_class.to_csv(per_class_path, index=False)

    benchmark = pd.read_csv(benchmark_path)
    if "notes" not in benchmark.columns:
        benchmark["notes"] = ""
    else:
        benchmark["notes"] = benchmark["notes"].fillna("").astype("object")
    test_metrics = metrics.loc[metrics["split"].eq("test")].iloc[0]
    row = {
        "method": method_name,
        "split": "test",
        "n_cells": int(test_mask.sum()),
        "accuracy": test_metrics["accuracy"],
        "macro_precision": test_metrics["macro_precision"],
        "macro_recall": test_metrics["macro_recall"],
        "macro_f1": test_metrics["macro_f1"],
        "status": "complete",
        "notes": "shared embedding linear probe",
    }
    if benchmark["method"].eq(method_name).any():
        benchmark.loc[benchmark["method"].eq(method_name), row.keys()] = list(row.values())
    else:
        benchmark = pd.concat([benchmark, pd.DataFrame([row])], ignore_index=True)
    benchmark.to_csv(benchmark_path, index=False)
    return metrics
