from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ica2.evaluation.metrics import classification_summary


METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"
BENCHMARK_PATH = METRICS_DIR / "benchmark_summary.csv"
PER_BATCH_PATH = METRICS_DIR / "per_batch_metrics.csv"
PER_BATCH_MD_PATH = METRICS_DIR / "per_batch_metrics.md"
PER_CLASS_MATRIX_PATH = METRICS_DIR / "per_class_f1_matrix.csv"
BENCHMARK_FIGURE_PATH = FIGURES_DIR / "benchmark_accuracy_macro_f1.png"
PER_BATCH_FIGURE_PATH = FIGURES_DIR / "per_batch_macro_f1_heatmap.png"
PER_CLASS_FIGURE_PATH = FIGURES_DIR / "per_class_f1_heatmap.png"
CONFUSION_FIGURE_PATH = FIGURES_DIR / "confusion_matrices_normalized.png"

PREDICTION_FILES = {
    "logistic_regression_svd50": METRICS_DIR / "logistic_regression_svd50_predictions.csv",
    "scgpt_embedding_logreg": METRICS_DIR / "scgpt_embedding_logreg_predictions.csv",
    "scimilarity_embedding_logreg": METRICS_DIR / "scimilarity_embedding_logreg_predictions.csv",
    "scfoundation_embedding_logreg": METRICS_DIR
    / "scfoundation_embedding_logreg_predictions.csv",
}

DISPLAY_NAMES = {
    "logistic_regression_svd50": "SVD + logreg",
    "scgpt_embedding_logreg": "scGPT",
    "scimilarity_embedding_logreg": "SCimilarity",
    "scfoundation_embedding_logreg": "scFoundation",
}


def format_metric(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def load_predictions() -> dict[str, pd.DataFrame]:
    predictions = {}
    for method, path in PREDICTION_FILES.items():
        if path.exists():
            predictions[method] = pd.read_csv(path)
    return predictions


def write_per_batch_metrics(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for method, table in predictions.items():
        for batch, batch_df in table.groupby("batch", sort=True):
            summary = classification_summary(
                batch_df["true_label"],
                batch_df["predicted_label"],
                method=method,
                split=f"test_batch:{batch}",
            )
            summary["batch"] = batch
            rows.append(summary)
    per_batch = pd.DataFrame(rows)
    ordered_columns = [
        "method",
        "batch",
        "split",
        "n_cells",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "status",
        "notes",
    ]
    per_batch = per_batch[ordered_columns]
    per_batch.to_csv(PER_BATCH_PATH, index=False)
    write_per_batch_markdown(per_batch)
    return per_batch


def write_per_batch_markdown(per_batch: pd.DataFrame) -> None:
    lines = [
        "# Per-batch Metrics",
        "",
        "| Method | Batch | Cells | Accuracy | Macro F1 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in per_batch.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    DISPLAY_NAMES.get(row.method, row.method),
                    str(row.batch),
                    f"{int(row.n_cells):,}",
                    format_metric(row.accuracy),
                    format_metric(row.macro_f1),
                ]
            )
            + " |"
        )
    PER_BATCH_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_per_class_matrix() -> pd.DataFrame:
    matrices = []
    for method in PREDICTION_FILES:
        path = METRICS_DIR / f"{method}_per_class_f1.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path)[["cell_type", "f1"]].rename(columns={"f1": method})
        matrices.append(table)
    matrix = matrices[0]
    for table in matrices[1:]:
        matrix = matrix.merge(table, on="cell_type", how="outer")
    matrix = matrix.sort_values("cell_type")
    matrix.to_csv(PER_CLASS_MATRIX_PATH, index=False)
    return matrix


def confusion_labels(predictions: dict[str, pd.DataFrame]) -> list[str]:
    labels: set[str] = set()
    for table in predictions.values():
        labels.update(table["true_label"].astype(str))
        labels.update(table["predicted_label"].astype(str))
    return sorted(labels)


def confusion_matrix_for(table: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    grouped = (
        table.assign(
            true_label=table["true_label"].astype(str),
            predicted_label=table["predicted_label"].astype(str),
        )
        .groupby(["true_label", "predicted_label"], observed=True)
        .size()
    )
    for (true_label, predicted_label), count in grouped.items():
        matrix.loc[true_label, predicted_label] = int(count)
    matrix.index.name = "true_label"
    matrix.columns.name = "predicted_label"
    return matrix


def write_confusion_matrices(
    predictions: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    labels = confusion_labels(predictions)
    normalized_matrices = {}
    for method, table in predictions.items():
        counts = confusion_matrix_for(table, labels)
        normalized = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        counts.to_csv(METRICS_DIR / f"{method}_confusion_matrix_counts.csv")
        normalized.to_csv(METRICS_DIR / f"{method}_confusion_matrix_normalized.csv")
        normalized_matrices[method] = normalized
    return normalized_matrices


def plot_benchmark() -> None:
    benchmark = pd.read_csv(BENCHMARK_PATH)
    completed = benchmark[benchmark["status"].eq("complete")].copy()
    methods = completed["method"].tolist()
    x = np.arange(len(methods))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.bar(
        x - width / 2,
        completed["accuracy"].astype(float),
        width,
        label="Accuracy",
        color="#2F6F73",
    )
    ax.bar(
        x + width / 2,
        completed["macro_f1"].astype(float),
        width,
        label="Macro F1",
        color="#C9822B",
    )
    ax.set_ylim(0.82, 1.02)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES.get(method, method) for method in methods], rotation=20, ha="right")
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D7D7D7", linewidth=0.7, alpha=0.7)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(BENCHMARK_FIGURE_PATH, dpi=220)
    plt.close(fig)


def plot_heatmap(
    matrix: pd.DataFrame,
    path: Path,
    *,
    title: str,
    x_labels: list[str],
    y_labels: list[str],
    values: np.ndarray,
    figsize: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_title(title, pad=10)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_yticklabels(y_labels)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            if np.isfinite(value):
                ax.text(
                    x,
                    y,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if value < 0.72 else "white",
                )
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.78, label="Macro F1" if "batch" in title.lower() else "F1")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_per_batch(per_batch: pd.DataFrame) -> None:
    pivot = per_batch.pivot(index="batch", columns="method", values="macro_f1")
    ordered_methods = [method for method in PREDICTION_FILES if method in pivot.columns]
    pivot = pivot[ordered_methods]
    plot_heatmap(
        pivot,
        PER_BATCH_FIGURE_PATH,
        title="Macro F1 by Batch",
        x_labels=[DISPLAY_NAMES.get(method, method) for method in pivot.columns],
        y_labels=pivot.index.astype(str).tolist(),
        values=pivot.to_numpy(dtype=float),
        figsize=(8.2, max(4.2, 0.35 * len(pivot.index) + 1.8)),
    )


def plot_per_class(matrix: pd.DataFrame) -> None:
    method_columns = [method for method in PREDICTION_FILES if method in matrix.columns]
    values = matrix[method_columns].to_numpy(dtype=float)
    plot_heatmap(
        matrix,
        PER_CLASS_FIGURE_PATH,
        title="Per-cell-type F1",
        x_labels=[DISPLAY_NAMES.get(method, method) for method in method_columns],
        y_labels=matrix["cell_type"].astype(str).tolist(),
        values=values,
        figsize=(8.2, max(5.0, 0.32 * len(matrix.index) + 1.8)),
    )


def plot_confusion_matrices(normalized_matrices: dict[str, pd.DataFrame]) -> None:
    ordered_methods = [method for method in PREDICTION_FILES if method in normalized_matrices]
    if not ordered_methods:
        return

    n_cols = 2
    n_rows = int(np.ceil(len(ordered_methods) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(11.0, 5.3 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for method_index, (ax, method) in enumerate(zip(axes.ravel(), ordered_methods)):
        row_index = method_index // n_cols
        col_index = method_index % n_cols
        matrix = normalized_matrices[method]
        values = matrix.to_numpy(dtype=float)
        image = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=1, aspect="equal")
        ax.set_title(DISPLAY_NAMES.get(method, method), pad=8)
        ax.set_xticks(np.arange(len(matrix.columns)))
        ax.set_yticks(np.arange(len(matrix.index)))
        if row_index == n_rows - 1:
            ax.set_xticklabels(matrix.columns.astype(str).tolist(), rotation=90, fontsize=6)
            ax.set_xlabel("Predicted label")
        else:
            ax.set_xticklabels([])
            ax.set_xlabel("")
        if col_index == 0:
            ax.set_yticklabels(matrix.index.astype(str).tolist(), fontsize=6)
            ax.set_ylabel("True label")
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)

    for ax in axes.ravel()[len(ordered_methods) :]:
        ax.axis("off")
    if image is not None:
        fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            shrink=0.7,
            fraction=0.025,
            pad=0.02,
            label="Row-normalized count",
        )
    fig.suptitle("Normalized Confusion Matrices", fontsize=14)
    fig.savefig(CONFUSION_FIGURE_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions()
    per_batch = write_per_batch_metrics(predictions)
    per_class = write_per_class_matrix()
    normalized_confusion = write_confusion_matrices(predictions)
    plot_benchmark()
    plot_per_batch(per_batch)
    plot_per_class(per_class)
    plot_confusion_matrices(normalized_confusion)
    print(f"wrote {PER_BATCH_PATH}")
    print(f"wrote {PER_BATCH_MD_PATH}")
    print(f"wrote {PER_CLASS_MATRIX_PATH}")
    for method in normalized_confusion:
        print(f"wrote {METRICS_DIR / f'{method}_confusion_matrix_counts.csv'}")
        print(f"wrote {METRICS_DIR / f'{method}_confusion_matrix_normalized.csv'}")
    print(f"wrote {BENCHMARK_FIGURE_PATH}")
    print(f"wrote {PER_BATCH_FIGURE_PATH}")
    print(f"wrote {PER_CLASS_FIGURE_PATH}")
    print(f"wrote {CONFUSION_FIGURE_PATH}")


if __name__ == "__main__":
    main()
