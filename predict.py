"""Generate churn predictions for new customer data."""

import pandas as pd
from pathlib import Path
import argparse

from __future__ import annotations
from src.config import (
    CUSTOMER_ID_COLUMN,
    MODEL_DIR,
    TARGET_COLUMN,
)
from src.data_loader import load_data
from src.feature_engineering import engineer_features
from src.model_io import (
    load_model,
    load_model_metadata,
)


DEFAULT_MODEL_PATH = MODEL_DIR / "best_model.joblib"
DEFAULT_METADATA_PATH = MODEL_DIR / "model_metadata.json"


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Generate churn probabilities and predictions "
            "for new customer data."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the input customer CSV file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path where predictions will be saved.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=(
            "Path to the trained model pipeline. "
            f"Default: {DEFAULT_MODEL_PATH}"
        ),
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help=(
            "Path to the model metadata JSON file. "
            f"Default: {DEFAULT_METADATA_PATH}"
        ),
    )

    return parser.parse_args()