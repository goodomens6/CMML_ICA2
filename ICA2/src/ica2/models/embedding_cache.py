from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def metadata_path_for(embeddings_path: Path) -> Path:
    return embeddings_path.with_suffix(".meta.json")


def save_embedding_cache(
    *,
    embeddings_path: Path,
    embeddings: np.ndarray,
    cell_ids: np.ndarray,
    metadata: dict[str, object],
) -> None:
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(metadata, sort_keys=True)
    np.savez_compressed(
        embeddings_path,
        embeddings=embeddings,
        cell_ids=cell_ids,
        metadata_json=metadata_json,
    )
    metadata_path_for(embeddings_path).write_text(metadata_json + "\n", encoding="utf-8")


def load_embedding_cache(
    *,
    embeddings_path: Path,
    expected_cell_ids: np.ndarray,
    expected_metadata: dict[str, object],
) -> np.ndarray | None:
    if not embeddings_path.exists():
        return None

    with np.load(embeddings_path) as stored:
        stored_ids = stored["cell_ids"].astype(str)
        if not np.array_equal(stored_ids, expected_cell_ids.astype(str)):
            return None
        stored_metadata = read_embedding_metadata(embeddings_path, stored)
        if stored_metadata != expected_metadata:
            return None
        return stored["embeddings"].copy()


def read_embedding_metadata(
    embeddings_path: Path,
    stored: np.lib.npyio.NpzFile | None = None,
) -> dict[str, object] | None:
    if stored is not None and "metadata_json" in stored.files:
        return json.loads(str(stored["metadata_json"].item()))

    sidecar_path = metadata_path_for(embeddings_path)
    if sidecar_path.exists():
        return json.loads(sidecar_path.read_text(encoding="utf-8"))

    return None
