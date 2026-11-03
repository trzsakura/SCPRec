"""Shared train-derived validation protocol for small recommenders."""

from __future__ import annotations

import math
from typing import Any, Mapping


def split_expert_training_rows(
    dataset_dict: Mapping[str, Any],
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[dict[str, Any], Any]:
    """Reserve a deterministic row-level subset of ``train`` for validation.

    The returned mapping keeps the original item catalog and official
    validation/test splits, but replaces ``train`` with the fit subset. The
    second return value is the held-out subset used for early stopping and for
    the expert-quality prior in the Stage-II router.
    """

    fraction = float(validation_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must be finite and between 0 and 1.")
    if "train" not in dataset_dict:
        raise KeyError("DatasetDict must contain a train split.")
    train_split = dataset_dict["train"]
    if len(train_split) < 2:
        raise ValueError("At least two training rows are required for validation.")
    if not hasattr(train_split, "train_test_split"):
        raise TypeError("The train split must provide train_test_split().")

    split = train_split.train_test_split(
        test_size=fraction,
        seed=int(seed),
        shuffle=True,
    )
    fit_split = split["train"]
    validation_split = split["test"]
    if len(fit_split) == 0 or len(validation_split) == 0:
        raise ValueError("The requested validation split produced an empty subset.")

    fit_dataset = dict(dataset_dict)
    fit_dataset["train"] = fit_split
    return fit_dataset, validation_split
