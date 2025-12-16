import json
import os
import platform
import random
import sys
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_csv(path: str, df, percent_cols=None) -> None:
    """Write DataFrame to CSV. If percent_cols specified, format those columns as percentages."""
    ensure_dir(os.path.dirname(path) or ".")
    if percent_cols:
        df_out = df.copy()
        for col in percent_cols:
            if col in df_out.columns:
                df_out[col] = df_out[col].apply(lambda x: f"{x*100:.2f}%" if isinstance(x, (int, float)) else x)
        df_out.to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


def format_metrics_percent(metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert decimal metrics to percentage strings for display."""
    result = {}
    percent_keys = ['macro_f1', 'weighted_f1', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'ci_low', 'ci_high']
    for k, v in metrics_dict.items():
        if k in percent_keys and isinstance(v, (int, float)):
            result[k] = f"{v*100:.2f}%"
        else:
            result[k] = v
    return result


def now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def get_env_metadata(extra_packages: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    import importlib

    pkgs = [
        "numpy",
        "pandas",
        "sklearn",
        "lightgbm",
        "catboost",
        "optuna",
        "matplotlib",
        "seaborn",
    ]
    if extra_packages:
        pkgs.extend(list(extra_packages))

    versions: Dict[str, Optional[str]] = {}
    for p in pkgs:
        try:
            m = importlib.import_module(p)
            versions[p] = getattr(m, "__version__", None)
        except Exception:
            versions[p] = None

    return {
        "timestamp_utc": now_utc_iso(),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": versions,
    }
