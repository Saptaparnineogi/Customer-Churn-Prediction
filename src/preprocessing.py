"""
Preprocessing utilities for the Customer Churn project.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_types(X: pd.DataFrame):
    """
    Automatically identify numerical and categorical features.
    """

    categorical_features = (
        X.select_dtypes(include=["object", "category", "string"])
        .columns
        .tolist()
    )

    numerical_features = (
        X.select_dtypes(include=["number"])
        .columns
        .tolist()
    )

    return numerical_features, categorical_features


def build_preprocessor(
    numerical_features,
    categorical_features,
):
    """
    Build preprocessing pipeline.
    """

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numerical_features,
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features,
            ),
        ]
    )

    return preprocessor