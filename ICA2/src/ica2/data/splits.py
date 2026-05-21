from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitFractions:
    train: float
    validation: float
    test: float

    def __post_init__(self) -> None:
        total = self.train + self.validation + self.test
        if not np.isclose(total, 1.0):
            raise ValueError(f"split fractions must sum to 1.0, got {total}")


def make_label_preserving_split(
    obs: pd.DataFrame,
    label_key: str,
    batch_key: str,
    fractions: SplitFractions,
    seed: int,
) -> pd.DataFrame:
    """Create a reproducible split while keeping every label represented when possible."""
    rng = np.random.default_rng(seed)
    assignments: list[pd.DataFrame] = []

    for label, group in obs.groupby(label_key, observed=True):
        indices = group.index.to_numpy(copy=True)
        rng.shuffle(indices)
        count = len(indices)

        if count < 3:
            raise ValueError(
                f"label {label!r} has only {count} cells; cannot place it in train/validation/test"
            )

        n_validation = max(1, int(round(count * fractions.validation)))
        n_test = max(1, int(round(count * fractions.test)))
        n_train = count - n_validation - n_test

        while n_train < 1:
            if n_validation >= n_test and n_validation > 1:
                n_validation -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                raise ValueError(f"label {label!r} cannot be split safely")
            n_train = count - n_validation - n_test

        split_values = np.array(
            ["train"] * n_train
            + ["validation"] * n_validation
            + ["test"] * n_test,
            dtype=object,
        )

        assignments.append(
            pd.DataFrame(
                {
                    "cell_id": indices,
                    label_key: label,
                    batch_key: obs.loc[indices, batch_key].astype(str).to_numpy(),
                    "split": split_values,
                }
            )
        )

    split_df = pd.concat(assignments, ignore_index=True)
    split_df = split_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return split_df


def summarize_split(split_df: pd.DataFrame, label_key: str) -> pd.DataFrame:
    """Return per-label counts across train/validation/test."""
    summary = (
        split_df.groupby([label_key, "split"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["train", "validation", "test"], fill_value=0)
        .reset_index()
    )
    summary["total"] = summary[["train", "validation", "test"]].sum(axis=1)
    return summary
