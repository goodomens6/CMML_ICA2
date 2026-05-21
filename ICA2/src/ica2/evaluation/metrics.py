from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


def classification_summary(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    method: str,
    split: str,
) -> dict[str, object]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "method": method,
        "split": split,
        "n_cells": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "status": "complete",
        "notes": "",
    }


def per_class_f1(
    y_true: Iterable[str],
    y_pred: Iterable[str],
) -> pd.DataFrame:
    labels = sorted(set(y_true))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "cell_type": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
