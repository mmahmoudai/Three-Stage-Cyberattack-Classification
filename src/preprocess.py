from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class CatBoostImputer:
    numeric_medians: Dict[str, float]
    categorical_fill_value: str

    def transform(self, df: pd.DataFrame, numeric_cols: List[str], cat_cols: List[str]) -> pd.DataFrame:
        out = df.copy()
        for c in numeric_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out[c] = out[c].fillna(self.numeric_medians[c])
        for c in cat_cols:
            out[c] = out[c].astype("object")
            out[c] = out[c].where(~out[c].isna(), self.categorical_fill_value)
            out[c] = out[c].astype(str)
        return out


def build_onehot_preprocessor(numeric_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def fit_catboost_imputer(df: pd.DataFrame, numeric_cols: List[str]) -> CatBoostImputer:
    medians: Dict[str, float] = {}
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        medians[c] = float(np.nanmedian(s.to_numpy()))
    return CatBoostImputer(numeric_medians=medians, categorical_fill_value="missing")


def cat_feature_indices(feature_cols: List[str], cat_cols: List[str]) -> List[int]:
    s = set(cat_cols)
    return [i for i, c in enumerate(feature_cols) if c in s]


def split_xy(df: pd.DataFrame, feature_cols: List[str], target_col: str):
    x = df[feature_cols]
    y = df[target_col]
    return x, y
