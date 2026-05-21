from __future__ import annotations

import os
import time
from pathlib import Path

import requests


SHARE_URL = (
    "https://hopebio2020.sharepoint.com/:f:/s/PublicSharedfiles/"
    "IgBlEJ72TBE5Q76AmgXbgjXiAR69fzcrgzqgUYdSThPLrqk"
)
FILE_URL = (
    "https://hopebio2020.sharepoint.com/sites/PublicSharedfiles/"
    "Shared%20Documents/Public%20Shared%20files/models.ckpt?download=1"
)
MODEL_DIR = Path(os.environ.get("CMML_MODEL_DIR", r"D:\cmml_models"))
TARGET = MODEL_DIR / "scfoundation" / "models.ckpt"
EXPECTED_SIZE = 1_432_587_886
CHUNK_SIZE = 1024 * 1024


def download_once() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    current_size = TARGET.stat().st_size if TARGET.exists() else 0

    with requests.Session() as session:
        session.get(SHARE_URL, timeout=120)
        headers = {"Range": f"bytes={current_size}-"} if current_size else {}
        with session.get(FILE_URL, headers=headers, stream=True, timeout=120) as response:
            response.raise_for_status()
            mode = "ab" if current_size and response.status_code == 206 else "wb"
            with TARGET.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        handle.write(chunk)


def main() -> None:
    while True:
        size = TARGET.stat().st_size if TARGET.exists() else 0
        if size >= EXPECTED_SIZE:
            return
        try:
            download_once()
        except Exception as exc:
            print(f"retrying after error: {exc}", flush=True)
            time.sleep(10)


if __name__ == "__main__":
    main()
