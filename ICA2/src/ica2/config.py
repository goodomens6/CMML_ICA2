from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand_env_defaults(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        env_value = os.environ.get(name)
        if env_value is not None:
            return env_value
        if default is not None:
            return default
        raise KeyError(f"Environment variable {name!r} is required by benchmark.yaml")

    return os.path.expanduser(_ENV_PATTERN.sub(replace, value))


def _expand_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_config(item) for item in value]
    if isinstance(value, str):
        return _expand_env_defaults(value)
    return value


def load_benchmark_config(root: Path) -> dict[str, Any]:
    config_path = root / "configs" / "benchmark.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return _expand_config(yaml.safe_load(handle))


def resolve_from_root(root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return root / path
