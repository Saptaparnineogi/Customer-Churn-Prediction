import numpy as np
import pandas as pd
import pytest

from src.decision_threshold import (
    evaluate_thresholds,
    find_best_threshold,
)

def test_evaluate_thresholds_returns_dataframe() -> None:
    """Threshold evaluation should return a pandas DataFrame."""

    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.10, 0.40, 0.60, 0.90])

    results = evaluate_thresholds(
        y_true=y_true,
        y_prob=y_prob,
    )

    assert isinstance(results, pd.DataFrame)


def test_evaluate_thresholds_contains_expected_columns() -> None:
    """The result should contain the required evaluation metrics."""

    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.10, 0.40, 0.60, 0.90])

    results = evaluate_thresholds(
        y_true=y_true,
        y_prob=y_prob,
    )

    expected_columns = {
        "Threshold",
        "Precision",
        "Recall",
        "F1",
    }

    assert expected_columns.issubset(results.columns)

def test_evaluate_thresholds_uses_given_thresholds() -> None:
    """The function should evaluate exactly the supplied thresholds."""

    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.10, 0.40, 0.60, 0.90])
    thresholds = np.array([0.30, 0.50, 0.70])

    results = evaluate_thresholds(
        y_true=y_true,
        y_prob=y_prob,
        thresholds=thresholds,
    )

    np.testing.assert_allclose(
        results["Threshold"].to_numpy(),
        thresholds,
    )

def test_threshold_metrics_are_correct() -> None:
    """Metrics should match the expected predictions at a known threshold."""

    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.10, 0.40, 0.60, 0.90])

    results = evaluate_thresholds(
        y_true=y_true,
        y_prob=y_prob,
        thresholds=np.array([0.50]),
    )

    result = results.iloc[0]

    assert result["Precision"] == pytest.approx(1.0)
    assert result["Recall"] == pytest.approx(1.0)
    assert result["F1"] == pytest.approx(1.0)

def test_find_best_threshold_selects_highest_metric() -> None:
    """The selected row should have the highest requested metric."""

    threshold_results = pd.DataFrame(
        {
            "Threshold": [0.30, 0.50, 0.70],
            "Precision": [0.50, 0.75, 0.90],
            "Recall": [0.90, 0.80, 0.40],
            "F1": [0.64, 0.77, 0.55],
        }
    )

    best_result = find_best_threshold(
        threshold_results=threshold_results,
        metric="F1",
    )

    assert best_result["Threshold"] == pytest.approx(0.50)
    assert best_result["F1"] == pytest.approx(0.77)