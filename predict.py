"""Generate churn predictions for new customer data."""

from __future__ import annotations
import pandas as pd
from pathlib import Path
import argparse


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


def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Prepare customer features for prediction.

    Parameters
    ----------
    df:
        Raw customer data.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series | None]
        Model feature matrix and optional customer IDs.
    """

    df = engineer_features(df.copy())

    customer_ids = None

    if CUSTOMER_ID_COLUMN in df.columns:
        customer_ids = df[CUSTOMER_ID_COLUMN].copy()

    columns_to_drop = [
        column
        for column in (
            CUSTOMER_ID_COLUMN,
            TARGET_COLUMN,
        )
        if column in df.columns
    ]

    features = df.drop(columns=columns_to_drop)

    if features.empty:
        raise ValueError(
            "No model features remain after removing "
            "identifier and target columns."
        )

    return features, customer_ids


def get_decision_threshold(
    metadata: dict,
) -> float:
    """
    Extract and validate the decision threshold.

    Parameters
    ----------
    metadata:
        Model metadata dictionary.

    Returns
    -------
    float
        Classification threshold.
    """

    threshold = metadata.get("decision_threshold")

    if threshold is None:
        raise KeyError(
            "The metadata file does not contain "
            "'decision_threshold'."
        )

    threshold = float(threshold)

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Decision threshold must be between 0 and 1."
        )

    return threshold


def make_predictions(
    model,
    features: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Generate churn probabilities and binary predictions.

    Parameters
    ----------
    model:
        Fitted sklearn classification pipeline.

    features:
        Customer feature matrix.

    threshold:
        Probability threshold used to classify churn.

    Returns
    -------
    pd.DataFrame
        Churn probabilities and predicted labels.
    """

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "The loaded model does not support predict_proba()."
        )

    probabilities = model.predict_proba(features)[:, 1]
    print(model.predict(features).sum())
    predictions = (
        probabilities >= threshold
    ).astype(int)

    return pd.DataFrame(
        {
            "churn_probability": probabilities,
            "predicted_churn": predictions,
            "predicted_churn_label": pd.Series(
                predictions
            ).map(
                {
                    0: "No",
                    1: "Yes",
                }
            ),
        }
    )


def build_output(
    predictions: pd.DataFrame,
    customer_ids: pd.Series | None,
) -> pd.DataFrame:
    """
    Add customer identifiers to prediction output.

    Parameters
    ----------
    predictions:
        Prediction results.

    customer_ids:
        Optional customer identifier series.

    Returns
    -------
    pd.DataFrame
        Final prediction output.
    """

    output = predictions.copy()

    if customer_ids is not None:
        output.insert(
            0,
            CUSTOMER_ID_COLUMN,
            customer_ids.reset_index(drop=True),
        )

    return output


def save_predictions(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save predictions to a CSV file.

    Parameters
    ----------
    predictions:
        Prediction output.

    output_path:
        Destination CSV path.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        output_path,
        index=False,
    )


def main() -> None:
    """Run the complete prediction workflow."""
    print("==================================================")
    print("Customer Churn Prediction")
    print("==================================================")
    args = parse_arguments()

    df = load_data(args.input)
    print(f"Loaded {len(df)} customer records from {args.input}")
    model = load_model(args.model)
    print("Model Loaded successfully.")
    metadata = load_model_metadata(args.metadata)

    threshold = get_decision_threshold(metadata)

    features, customer_ids = prepare_features(df)

    predictions = make_predictions(
        model=model,
        features=features,
        threshold=threshold,
    )

    output = build_output(
        predictions=predictions,
        customer_ids=customer_ids,
    )

    save_predictions(
        predictions=output,
        output_path=args.output,
    )

    print("Predictions generated successfully.")
    print(f"Customers processed: {len(output)}")
    print(f"Decision threshold: {threshold:.2f}")
    print(f"Predicted churners: {output['predicted_churn'].sum()}")
    print(f"Predictions saved to: {args.output}")
    print("Prediction completed successfully.")


if __name__ == "__main__":
    main()