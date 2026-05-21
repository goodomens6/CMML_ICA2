from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import requests


ROOT = Path(__file__).resolve().parents[1]
URL = "https://openproblems-data.s3.amazonaws.com/resources/datasets/openproblems_v1/pancreas/log_cp10k/dataset.h5ad"
DEFAULT_TARGET = (
    ROOT / "data" / "raw" / "human_pancreas_openproblems_v1_log_cp10k.h5ad"
)
TOTAL_SIZE = 1_356_334_299
WORKERS = 8
CHUNK_SIZE = 1024 * 1024


def download_range(start: int, end: int, part_path: Path, lock: Lock, progress: list[int]) -> None:
    existing = part_path.stat().st_size if part_path.exists() else 0
    actual_start = start + existing
    if actual_start > end:
        return

    headers = {"Range": f"bytes={actual_start}-{end}"}
    with requests.get(URL, headers=headers, stream=True, timeout=60) as response:
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError(f"Expected HTTP 206 for range request, got {response.status_code}")

        with part_path.open("ab") as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                handle.write(chunk)
                with lock:
                    progress[0] += len(chunk)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Open Problems human pancreas h5ad.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_TARGET,
        help="Output .h5ad path. Defaults to the project data/raw directory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=WORKERS,
        help="Number of parallel HTTP range downloads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = args.output.expanduser()
    if not target.is_absolute():
        target = ROOT / target
    workers = int(args.workers)
    if workers < 1:
        raise ValueError("--workers must be at least 1")

    target.parent.mkdir(parents=True, exist_ok=True)
    existing_size = target.stat().st_size if target.exists() else 0
    if existing_size >= TOTAL_SIZE:
        print("dataset already complete")
        return

    remaining = TOTAL_SIZE - existing_size
    workers = min(workers, remaining)
    segment_size = remaining // workers
    ranges: list[tuple[int, int]] = []
    start = existing_size
    for idx in range(workers):
        end = TOTAL_SIZE - 1 if idx == workers - 1 else start + segment_size - 1
        ranges.append((start, end))
        start = end + 1

    part_specs = [
        (start, end, target.with_name(f"{target.name}.range{idx + 1}.part"))
        for idx, (start, end) in enumerate(ranges)
    ]
    lock = Lock()
    progress = [
        existing_size
        + sum(part_path.stat().st_size for _, _, part_path in part_specs if part_path.exists())
    ]
    print(f"resuming from {existing_size} bytes")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(download_range, start, end, part_path, lock, progress)
            for start, end, part_path in part_specs
            if start <= end
        ]
        for future in as_completed(futures):
            future.result()
            print(f"completed {progress[0]} / {TOTAL_SIZE} bytes")

    temp_path = target.with_name(f"{target.name}.complete.tmp")
    with temp_path.open("wb") as output:
        if existing_size > 0:
            with target.open("rb") as initial_part:
                shutil.copyfileobj(initial_part, output)
        for _, _, part_path in part_specs:
            with part_path.open("rb") as part_handle:
                shutil.copyfileobj(part_handle, output)

    final_size = temp_path.stat().st_size
    if final_size != TOTAL_SIZE:
        raise RuntimeError(f"unexpected final size: {final_size}")

    temp_path.replace(target)
    for _, _, part_path in part_specs:
        part_path.unlink(missing_ok=True)
    print("download complete")


if __name__ == "__main__":
    main()
