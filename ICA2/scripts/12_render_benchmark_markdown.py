from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = [
    {
        "csv_path": ROOT / "results" / "metrics" / "benchmark_summary.csv",
        "md_path": ROOT / "results" / "metrics" / "benchmark_summary.md",
        "title": "Main Benchmark Summary",
        "include_notes": False,
    },
    {
        "csv_path": ROOT / "results" / "metrics" / "benchmark_harmonized_summary.csv",
        "md_path": ROOT / "results" / "metrics" / "benchmark_harmonized_summary.md",
        "title": "Harmonized Benchmark Summary",
        "include_notes": True,
    },
]


def format_metric(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def render_table(csv_path: Path, md_path: Path, title: str, include_notes: bool) -> None:
    table = pd.read_csv(csv_path)
    headers = [
        "Method",
        "Split",
        "Cells",
        "Accuracy",
        "Macro precision",
        "Macro recall",
        "Macro F1",
        "Status",
    ]
    alignments = ["---", "---", "---:", "---:", "---:", "---:", "---:", "---"]
    if include_notes and "notes" in table.columns:
        headers.append("Notes")
        alignments.append("---")

    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(alignments) + " |",
    ]

    for row in table.itertuples(index=False):
        values = [
            str(row.method),
            str(row.split),
            f"{int(row.n_cells):,}",
            format_metric(row.accuracy),
            format_metric(row.macro_precision),
            format_metric(row.macro_recall),
            format_metric(row.macro_f1),
            str(row.status),
        ]
        if include_notes and "notes" in table.columns:
            values.append("" if pd.isna(row.notes) else str(row.notes))
        lines.append("| " + " | ".join(values) + " |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {md_path}")


def main() -> None:
    for spec in TABLES:
        render_table(**spec)


if __name__ == "__main__":
    main()
