"""
Utilities for evaluating classification decision thresholds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Evaluate model performance across multiple classification thresholds.

    Parameters
    ----------
    y_true
        Ground truth binary labels.

    y_prob
        Predicted probabilities for the positive class.

    thresholds
        Thresholds to evaluate.
        Defaults to 0.05–0.95.

    Returns
    -------
    pandas.DataFrame
        Performance metrics for every threshold.
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    results = []

    for threshold in thresholds:

        y_pred = (y_prob >= threshold).astype(int)

        results.append(
            {
                "Threshold": threshold,
                "Precision": precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "Recall": recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "F1": f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
            }
        )

    return pd.DataFrame(results)