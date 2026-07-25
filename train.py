"""Run the complete customer churn model-training pipeline."""

from sklearn.model_selection import train_test_split

from src.config import (
    CUSTOMER_ID_COLUMN,
    DATA_PATH,
    MODEL_DIR,
    MODEL_SELECTION_METRIC,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data_loader import load_data
from src.feature_engineering import engineer_features
from src.model_training import get_models, train_models
from src.preprocessing import build_preprocessor, get_feature_types
from src.evaluation import (
    evaluate_models,
    results_to_dataframe,
    print_results,
)
from src.model_io import save_model
from src.decision_threshold import evaluate_thresholds


def prepare_target(df):
    """
    Convert the churn target into binary values.
    """
    target = df[TARGET_COLUMN]
    if target.dtype == "object" or target.dtype == "str":
        target = target.map({"No": 0, "Yes": 1})

    if target.isna().any():
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' contains unmapped or missing values."
        )

    return target.astype(int)


def main() -> None:
    """Execute the end-to-end training workflow."""

    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Apply feature engineering
    df = engineer_features(df)

    # 3. Separate features and target
    y = prepare_target(df)

    columns_to_drop = [TARGET_COLUMN]

    if CUSTOMER_ID_COLUMN in df.columns:
        columns_to_drop.append(CUSTOMER_ID_COLUMN)

    X = df.drop(columns=columns_to_drop)
    
    # 4. Create stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    # 5. Detect feature types using training data only
    numerical_features, categorical_features = get_feature_types(X_train)

    # 6. Build preprocessing pipeline
    preprocessor = build_preprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features
    )

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    print(f"Transformed train shape: {X_train_transformed.shape}")
    print(f"Transformed test shape: {X_test_transformed.shape}")
    # 7. Define and train models
    models = get_models(random_state=RANDOM_STATE)

    trained_models = train_models(
        preprocessor=preprocessor,
        models=models,
        X_train=X_train,
        y_train=y_train,
    )

    print("Training completed successfully.")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Models trained: {', '.join(trained_models.keys())}")

    results = evaluate_models(
    trained_models,
    X_test,
    y_test,
)

    results_df = results_to_dataframe(results)
    results_df = results_df.sort_values(
    by=MODEL_SELECTION_METRIC,
    ascending=False,
    )
    print_results(results_df)
    best_model_name = results_df.iloc[0]["Model"]

    save_model(
    trained_models[best_model_name],
    MODEL_DIR / "best_model.joblib",
    )

    print(f"Saved {best_model_name}")
    best_model_result = results[best_model_name]

    threshold_results = evaluate_thresholds(
    y_true=y_test,
    y_prob=best_model_result["y_prob"],
    )

    print(threshold_results.head())
    print(threshold_results.tail())


if __name__ == "__main__":
    main()