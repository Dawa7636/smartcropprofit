from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from utils import (
    CROP_COLUMN,
    CROP_DATA_PATH,
    CROP_MODEL_PATH,
    FEATURE_COLUMNS,
    METADATA_PATH,
    MODELS_DIR,
    YIELD_DATA_PATH,
    YIELD_MODEL_PATH,
    YIELD_TARGET,
)


CLASSIFIER_FEATURES = FEATURE_COLUMNS
REGRESSOR_FEATURES = FEATURE_COLUMNS + [CROP_COLUMN]


def load_dataset(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Dataset is missing or empty: {path}")

    dataframe = pd.read_csv(path)
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(missing_columns)}")
    return dataframe


def preprocess_crop_data() -> pd.DataFrame:
    crop_df = load_dataset(CROP_DATA_PATH, FEATURE_COLUMNS + [CROP_COLUMN]).copy()
    crop_df[FEATURE_COLUMNS] = crop_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    crop_df[CROP_COLUMN] = crop_df[CROP_COLUMN].astype(str).str.strip().str.lower()
    crop_df = crop_df.dropna(subset=FEATURE_COLUMNS + [CROP_COLUMN]).reset_index(drop=True)
    return crop_df


def preprocess_yield_data() -> pd.DataFrame:
    yield_df = load_dataset(YIELD_DATA_PATH, FEATURE_COLUMNS + [CROP_COLUMN, YIELD_TARGET]).copy()
    numeric_columns = FEATURE_COLUMNS + [YIELD_TARGET]
    yield_df[numeric_columns] = yield_df[numeric_columns].apply(pd.to_numeric, errors="raise")
    yield_df[CROP_COLUMN] = yield_df[CROP_COLUMN].astype(str).str.strip().str.lower()
    yield_df = yield_df.dropna(subset=numeric_columns + [CROP_COLUMN]).reset_index(drop=True)
    return yield_df


def train_crop_classifier(crop_df: pd.DataFrame) -> tuple[Pipeline, dict[str, float]]:
    selected_features = CLASSIFIER_FEATURES
    x = crop_df[selected_features]
    y = crop_df[CROP_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    classifier = Pipeline(
        steps=[
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=14,
                    min_samples_split=4,
                    random_state=42,
                    class_weight="balanced",
                ),
            )
        ]
    )
    classifier.fit(x_train, y_train)

    predictions = classifier.predict(x_test)
    metrics = {
        "crop_accuracy": round(float(accuracy_score(y_test, predictions)), 4),
    }
    return classifier, metrics


def train_yield_regressor(yield_df: pd.DataFrame) -> tuple[Pipeline, dict[str, float]]:
    selected_features = REGRESSOR_FEATURES
    x = yield_df[selected_features]
    y = yield_df[YIELD_TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("crop_encoder", OneHotEncoder(handle_unknown="ignore"), [CROP_COLUMN]),
            ("numeric_features", "passthrough", FEATURE_COLUMNS),
        ]
    )

    regressor = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    max_depth=16,
                    min_samples_split=3,
                    random_state=42,
                ),
            ),
        ]
    )
    regressor.fit(x_train, y_train)

    predictions = regressor.predict(x_test)
    metrics = {
        "yield_mae": round(float(mean_absolute_error(y_test, predictions)), 2),
        "yield_rmse": round(float(mean_squared_error(y_test, predictions) ** 0.5), 2),
    }
    return regressor, metrics


def save_training_artifacts(
    crop_model: Pipeline,
    yield_model: Pipeline,
    metrics: dict[str, float],
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(crop_model, CROP_MODEL_PATH)
    joblib.dump(yield_model, YIELD_MODEL_PATH)
    joblib.dump(
        {
            "classifier_features": CLASSIFIER_FEATURES,
            "regressor_features": REGRESSOR_FEATURES,
            "target_columns": {
                "crop": CROP_COLUMN,
                "yield": YIELD_TARGET,
            },
            "metrics": metrics,
        },
        METADATA_PATH,
    )


def train_and_save_models() -> dict[str, float]:
    crop_df = preprocess_crop_data()
    yield_df = preprocess_yield_data()

    crop_model, crop_metrics = train_crop_classifier(crop_df)
    yield_model, yield_metrics = train_yield_regressor(yield_df)

    metrics = {
        **crop_metrics,
        **yield_metrics,
    }
    save_training_artifacts(crop_model, yield_model, metrics)
    return metrics


def main() -> None:
    metrics = train_and_save_models()
    print(f"Crop model saved to: {CROP_MODEL_PATH}")
    print(f"Yield model saved to: {YIELD_MODEL_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")
    print(f"Crop classifier accuracy: {metrics['crop_accuracy']}")
    print(f"Yield MAE (kg/ha): {metrics['yield_mae']}")
    print(f"Yield RMSE (kg/ha): {metrics['yield_rmse']}")


if __name__ == "__main__":
    main()
