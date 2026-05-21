from pathlib import Path
import sys

import scanpy as sc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ica2.data.splits import SplitFractions, make_label_preserving_split, summarize_split
from ica2.config import load_benchmark_config, resolve_from_root


CONFIG = load_benchmark_config(ROOT)
DATASET_CONFIG = CONFIG["dataset"]
SPLIT_CONFIG = CONFIG["split"]
DATASET_PATH = resolve_from_root(ROOT, DATASET_CONFIG["path"])
SPLIT_PATH = ROOT / "data" / "splits" / f"pancreas_seed{SPLIT_CONFIG['seed']}_splits.csv"
SUMMARY_PATH = ROOT / "data" / "splits" / f"pancreas_seed{SPLIT_CONFIG['seed']}_summary.csv"


def main() -> None:
    adata = sc.read_h5ad(DATASET_PATH, backed="r")
    obs = adata.obs[[DATASET_CONFIG["label_key"], DATASET_CONFIG["batch_key"]]].copy()

    split_df = make_label_preserving_split(
        obs=obs,
        label_key=DATASET_CONFIG["label_key"],
        batch_key=DATASET_CONFIG["batch_key"],
        fractions=SplitFractions(
            train=SPLIT_CONFIG["train_fraction"],
            validation=SPLIT_CONFIG["validation_fraction"],
            test=SPLIT_CONFIG["test_fraction"],
        ),
        seed=SPLIT_CONFIG["seed"],
    )
    summary_df = summarize_split(split_df, label_key=DATASET_CONFIG["label_key"])

    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(SPLIT_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print(f"wrote {SPLIT_PATH}")
    print(f"wrote {SUMMARY_PATH}")
    print()
    print(split_df["split"].value_counts().reindex(["train", "validation", "test"]).to_string())
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
