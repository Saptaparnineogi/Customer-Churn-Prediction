"""
Utilities for saving and loading trained models.
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any
import joblib
from sklearn.base import BaseEstimator


def save_model(
    model: BaseEstimator,
    model_path: str | Path,
) -> None:
    """
    Save a trained model to disk.

    Parameters
    ----------
    model:
        Trained model or pipeline.

    model_path:
        Destination path.
    """

    model_path = Path(model_path)

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        model_path,
    )


def load_model(
    model_path: str | Path,
) -> BaseEstimator:
    """
    Load a trained model.

    Parameters
    ----------
    model_path:
        Path to saved model.

    Returns
    -------
    BaseEstimator
        Loaded sklearn pipeline.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    return joblib.load(model_path)

def save_model_metadata(
    metadata: dict[str, Any],
    metadata_path: str | Path,
) -> None:
    """
    Save model metadata as a JSON file.

    Parameters
    ----------
    metadata:
        Dictionary containing model metadata.

    metadata_path:
        Destination path.
    """

    metadata_path = Path(metadata_path)

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )