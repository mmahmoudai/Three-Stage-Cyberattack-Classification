from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import f1_score


def bootstrap_macro_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    n_resamples: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)

    scores = np.empty(n_resamples, dtype=float)
    idx = np.arange(n)
    for i in range(n_resamples):
        sample_idx = rng.choice(idx, size=n, replace=True)
        scores[i] = float(
            f1_score(
                y_true[sample_idx],
                y_pred[sample_idx],
                average="macro",
                labels=labels,
                zero_division=0,
            )
        )

    low = float(np.percentile(scores, 2.5))
    high = float(np.percentile(scores, 97.5))
    return {
        "macro_f1": float(np.mean(scores)),
        "ci_low": low,
        "ci_high": high,
    }
