from pathlib import Path
import sys

import scanpy as sc


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ica2.config import load_benchmark_config, resolve_from_root


CONFIG = load_benchmark_config(ROOT)
DATASET_PATH = resolve_from_root(ROOT, CONFIG["dataset"]["path"])


def main() -> None:
    adata = sc.read_h5ad(DATASET_PATH)

    print(f"shape: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"obs: {list(adata.obs.columns)}")
    print(f"var: {list(adata.var.columns)}")
    print(f"layers: {list(adata.layers.keys())}")
    print(f"obsm: {list(adata.obsm.keys())}")
    print(f"obsp: {list(adata.obsp.keys())}")
    print(f"varm: {list(adata.varm.keys())}")
    print(f"uns: {list(adata.uns.keys())}")

    if "cell_type" in adata.obs:
        print("\ncell_type counts:")
        print(adata.obs["cell_type"].value_counts().sort_index().to_string())

    if "batch" in adata.obs:
        print("\nbatch counts:")
        print(adata.obs["batch"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
