from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split_indices(
    df: pd.DataFrame,
    stratify_col: str,
    seed: int,
    train_size: float,
    val_size: float,
    test_size: float,
) -> Dict[str, np.ndarray]:
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split sizes must sum to 1.0. Got {total}")

    idx = np.arange(len(df))
    y = df[stratify_col]

    train_idx, temp_idx = train_test_split(
        idx,
        test_size=(val_size + test_size),
        random_state=seed,
        stratify=y,
    )

    y_temp = y.iloc[temp_idx]
    val_ratio = val_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_ratio,
        random_state=seed,
        stratify=y_temp,
    )

    return {
        "train": np.asarray(train_idx, dtype=int),
        "val": np.asarray(val_idx, dtype=int),
        "test": np.asarray(test_idx, dtype=int),
    }
