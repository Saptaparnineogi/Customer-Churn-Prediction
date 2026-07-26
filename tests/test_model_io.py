"""Tests for model persistence utilities."""

import json
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.model_io import (
    load_model,
    save_model,
    save_model_metadata,
)


def test_save_and_load_model(tmp_path: Path) -> None:
    """A saved sklearn model should be loadable from disk."""

    model = DummyClassifier(strategy="most_frequent")
    model.fit(
        [[0], [1], [2]],
        [0, 1, 1],
    )

    model_path = tmp_path / "test_model.joblib"

    save_model(
        model=model,
        model_path=model_path,
    )

    loaded_model = load_model(model_path)

    assert model_path.exists()
    assert isinstance(loaded_model, DummyClassifier)
    assert loaded_model.strategy == "most_frequent"


def test_save_and_load_pipeline(tmp_path: Path) -> None:
    """The persistence functions should support complete sklearn pipelines."""

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]
    )

    pipeline.fit(
        [[0], [1], [2]],
        [0, 1, 1],
    )

    model_path = tmp_path / "test_pipeline.joblib"

    save_model(
        model=pipeline,
        model_path=model_path,
    )

    loaded_pipeline = load_model(model_path)

    assert isinstance(loaded_pipeline, Pipeline)
    assert list(loaded_pipeline.named_steps) == [
        "scaler",
        "model",
    ]


def test_save_model_creates_parent_directory(
    tmp_path: Path,
) -> None:
    """Saving should create missing parent directories."""

    model = DummyClassifier(strategy="most_frequent")
    model.fit(
        [[0], [1]],
        [0, 1],
    )

    model_path = (
        tmp_path
        / "nested"
        / "models"
        / "model.joblib"
    )

    save_model(
        model=model,
        model_path=model_path,
    )

    assert model_path.exists()


def test_load_model_raises_for_missing_file(
    tmp_path: Path,
) -> None:
    """Loading a nonexistent model should raise FileNotFoundError."""

    missing_path = tmp_path / "missing_model.joblib"

    try:
        load_model(missing_path)
    except FileNotFoundError as error:
        assert str(missing_path) in str(error)
    else:
        raise AssertionError(
            "Expected FileNotFoundError to be raised."
        )


def test_save_model_metadata(tmp_path: Path) -> None:
    """Model metadata should be saved as valid JSON."""

    metadata = {
        "model_name": "Random Forest",
        "selection_metric": "F1",
        "decision_threshold": 0.54,
        "precision": 0.554,
        "recall": 0.757,
        "f1_score": 0.640,
    }

    metadata_path = tmp_path / "model_metadata.json"

    save_model_metadata(
        metadata=metadata,
        metadata_path=metadata_path,
    )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        saved_metadata = json.load(file)

    assert metadata_path.exists()
    assert saved_metadata == metadata