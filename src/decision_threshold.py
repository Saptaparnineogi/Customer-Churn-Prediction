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


def find_best_threshold(
    threshold_results: pd.DataFrame,
    metric: str = "F1",
) -> pd.Series:
    """
    Return the threshold row with the highest selected metric.

    Parameters
    ----------
    threshold_results
        DataFrame returned by `evaluate_thresholds`.

    metric
        Metric to maximize. Supported values are:
        "Precision", "Recall", and "F1".

    Returns
    -------
    pandas.Series
        Row containing the best threshold and its metrics.
    """

    supported_metrics = {
        "Precision",
        "Recall",
        "F1",
    }

    if metric not in supported_metrics:
        raise ValueError(
            f"Unsupported metric '{metric}'. "
            f"Choose from {sorted(supported_metrics)}."
        )

    if threshold_results.empty:
        raise ValueError(
            "threshold_results cannot be empty."
        )

    best_index = threshold_results[metric].idxmax()

    return threshold_results.loc[best_index]