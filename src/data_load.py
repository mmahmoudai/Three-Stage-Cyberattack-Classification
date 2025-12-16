from __future__ import annotations

import os
from typing import Dict, List, Tuple

import pandas as pd


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def validate_schema(
    df: pd.DataFrame,
    protocol_col: str,
    class_col: str,
    category_col: str,
) -> None:
    required = {protocol_col, class_col, category_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def remove_leakage_columns(
    df: pd.DataFrame,
    protocol_col: str,
    class_col: str,
    category_col: str,
) -> Tuple[pd.DataFrame, List[str]]:
    dropped: List[str] = []
    feature_cols = [c for c in df.columns if c not in {class_col, category_col}]

    class_series = df[class_col]
    category_series = df[category_col]

    for c in feature_cols:
        if c in {protocol_col}:
            continue
        try:
            s = df[c]
            if s.equals(class_series) or s.equals(category_series):
                dropped.append(c)
        except Exception:
            continue

    if dropped:
        df = df.drop(columns=dropped)

    return df, dropped


def ensure_dataset_exists(csv_path: str) -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)


def infer_feature_columns(
    df: pd.DataFrame,
    protocol_col: str,
    class_col: str,
    category_col: str,
) -> Tuple[List[str], List[str], List[str]]:
    feature_cols = [c for c in df.columns if c not in {class_col, category_col}]
    cat_cols = [protocol_col]
    num_cols = [c for c in feature_cols if c not in set(cat_cols)]
    return feature_cols, num_cols, cat_cols
