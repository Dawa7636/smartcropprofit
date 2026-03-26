from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tsa.arima.model import ARIMA

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "sample_data"
MODELS_DIR = BASE_DIR / "models"

CROP_DATA_PATH = DATA_DIR / "crop_recommendation.csv"
YIELD_DATA_PATH = DATA_DIR / "yield_dataset.csv"
MARKET_DATA_PATH = DATA_DIR / "market_prices.csv"
CROP_MODEL_PATH = MODELS_DIR / "crop_classifier.pkl"
YIELD_MODEL_PATH = MODELS_DIR / "yield_regressor.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.pkl"

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "rainfall"]
YIELD_TARGET = "yield_kg_per_ha"
COST_COLUMN = "cost_per_ha"
MARKET_PRICE_COLUMN = "price_per_kg"
DATE_COLUMN = "date"
CROP_COLUMN = "crop"

MIN_NUMERIC_VALUE = 0.0
FORECAST_DAYS = 7


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Dataset is missing or empty: {path}")

    dataframe = pd.read_csv(path)
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(missing_columns)}")
    return dataframe


def load_crop_dataset() -> pd.DataFrame:
    crop_df = _read_csv(CROP_DATA_PATH, FEATURE_COLUMNS + [CROP_COLUMN])
    crop_df[FEATURE_COLUMNS] = crop_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    crop_df[CROP_COLUMN] = crop_df[CROP_COLUMN].astype(str).str.strip().str.lower()
    return crop_df


def load_yield_dataset() -> pd.DataFrame:
    yield_df = _read_csv(YIELD_DATA_PATH, FEATURE_COLUMNS + [CROP_COLUMN, YIELD_TARGET, COST_COLUMN])
    numeric_columns = FEATURE_COLUMNS + [YIELD_TARGET, COST_COLUMN]
    yield_df[numeric_columns] = yield_df[numeric_columns].apply(pd.to_numeric, errors="raise")
    yield_df[CROP_COLUMN] = yield_df[CROP_COLUMN].astype(str).str.strip().str.lower()
    return yield_df


def load_market_dataset() -> pd.DataFrame:
    market_df = _read_csv(MARKET_DATA_PATH, [DATE_COLUMN, CROP_COLUMN, MARKET_PRICE_COLUMN])
    market_df[DATE_COLUMN] = pd.to_datetime(market_df[DATE_COLUMN], errors="raise")
    market_df[MARKET_PRICE_COLUMN] = pd.to_numeric(market_df[MARKET_PRICE_COLUMN], errors="raise")
    market_df[CROP_COLUMN] = market_df[CROP_COLUMN].astype(str).str.strip().str.lower()
    return market_df.sort_values([CROP_COLUMN, DATE_COLUMN]).reset_index(drop=True)


def _augment_tabular_data(dataframe: pd.DataFrame, numeric_columns: list[str], repeats: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    augmented_frames = [dataframe]

    for _ in range(repeats):
        sampled = dataframe.sample(frac=1.0, replace=True, random_state=int(rng.integers(0, 1_000_000))).copy()
        for column in numeric_columns:
            noise_scale = max(sampled[column].std(ddof=0), 1.0) * 0.03
            sampled[column] = np.maximum(
                MIN_NUMERIC_VALUE,
                sampled[column] + rng.normal(0.0, noise_scale, size=len(sampled)),
            )
        augmented_frames.append(sampled)

    return pd.concat(augmented_frames, ignore_index=True)


def build_crop_classifier(crop_df: pd.DataFrame) -> tuple[Pipeline, dict[str, float]]:
    augmented_crop_df = _augment_tabular_data(crop_df, FEATURE_COLUMNS)
    x_train, x_test, y_train, y_test = train_test_split(
        augmented_crop_df[FEATURE_COLUMNS],
        augmented_crop_df[CROP_COLUMN],
        test_size=0.2,
        random_state=42,
        stratify=augmented_crop_df[CROP_COLUMN],
    )

    pipeline = Pipeline(
        steps=[
            (
                "classifier",
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
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    metrics = {
        "crop_accuracy": round(float(accuracy_score(y_test, predictions)), 4),
    }
    return pipeline, metrics


def build_yield_regressor(yield_df: pd.DataFrame) -> tuple[Pipeline, dict[str, float]]:
    augmented_yield_df = _augment_tabular_data(yield_df, FEATURE_COLUMNS + [YIELD_TARGET, COST_COLUMN])
    model_features = FEATURE_COLUMNS + [CROP_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        augmented_yield_df[model_features],
        augmented_yield_df[YIELD_TARGET],
        test_size=0.2,
        random_state=42,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("crop", OneHotEncoder(handle_unknown="ignore"), [CROP_COLUMN]),
            ("numeric", "passthrough", FEATURE_COLUMNS),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=400,
                    max_depth=16,
                    min_samples_split=3,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    metrics = {
        "yield_mae": round(float(mean_absolute_error(y_test, predictions)), 2),
        "yield_rmse": round(float(np.sqrt(mean_squared_error(y_test, predictions))), 2),
    }
    return pipeline, metrics


def train_and_save_models() -> dict[str, float]:
    ensure_directories()
    crop_df = load_crop_dataset()
    yield_df = load_yield_dataset()

    crop_model, crop_metrics = build_crop_classifier(crop_df)
    yield_model, yield_metrics = build_yield_regressor(yield_df)

    metadata = {
        "crop_costs": yield_df.groupby(CROP_COLUMN)[COST_COLUMN].mean().round(2).to_dict(),
        "metrics": {
            **crop_metrics,
            **yield_metrics,
        },
    }
    joblib.dump(crop_model, CROP_MODEL_PATH)
    joblib.dump(yield_model, YIELD_MODEL_PATH)
    joblib.dump(metadata, METADATA_PATH)
    return metadata["metrics"]


def load_models() -> dict[str, Any]:
    ensure_directories()
    required_files = [CROP_MODEL_PATH, YIELD_MODEL_PATH, METADATA_PATH]
    if any((not path.exists() or path.stat().st_size == 0) for path in required_files):
        train_and_save_models()

    return {
        "crop_model": joblib.load(CROP_MODEL_PATH),
        "yield_model": joblib.load(YIELD_MODEL_PATH),
        "metadata": joblib.load(METADATA_PATH),
    }


def validate_request_payload(payload: dict[str, Any]) -> dict[str, float]:
    missing = [column for column in FEATURE_COLUMNS if column not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    features: dict[str, float] = {}
    for column in FEATURE_COLUMNS:
        try:
            value = float(payload[column])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Field '{column}' must be numeric.") from exc
        if value < MIN_NUMERIC_VALUE:
            raise ValueError(f"Field '{column}' must be non-negative.")
        features[column] = value
    return features


def get_latest_market_price(market_history: pd.DataFrame, crop: str) -> float:
    crop_history = market_history.loc[market_history[CROP_COLUMN] == crop].sort_values(DATE_COLUMN)
    if crop_history.empty:
        raise ValueError(f"No market prices available for crop '{crop}'.")
    return round(float(crop_history.iloc[-1][MARKET_PRICE_COLUMN]), 2)


def estimate_cost(features: dict[str, float], crop: str, crop_costs: dict[str, float]) -> float:
    base_cost = float(crop_costs.get(crop, np.mean(list(crop_costs.values()))))
    nutrient_factor = (features["N"] + features["P"] + features["K"]) * 2.15
    climate_factor = (features["temperature"] * 9.0) + (features["humidity"] * 3.2) + (features["rainfall"] * 1.8)
    return round(base_cost + nutrient_factor + climate_factor * 0.12, 2)


def _generate_sample_price_series(crop: str, days: int = 14) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(crop)) % (2**32))
    base_price = 25.0 + float(rng.integers(5, 75))
    drift = float(rng.uniform(-0.3, 0.6))
    seasonal = np.sin(np.linspace(0, 2 * np.pi, days)) * float(rng.uniform(0.2, 1.0))
    noise = rng.normal(0.0, 0.35, size=days)
    values = np.maximum(1.0, base_price + drift * np.arange(days) + seasonal + noise)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")
    return pd.DataFrame(
        {
            DATE_COLUMN: dates,
            CROP_COLUMN: crop,
            MARKET_PRICE_COLUMN: values.round(2),
        }
    )


def get_crop_price_history(market_history: pd.DataFrame, crop: str) -> pd.DataFrame:
    crop_history = market_history.loc[market_history[CROP_COLUMN] == crop].sort_values(DATE_COLUMN).copy()
    if crop_history.empty or len(crop_history) < 3:
        return _generate_sample_price_series(crop)
    return crop_history


def forecast_prices(market_history: pd.DataFrame, crop: str, forecast_days: int = FORECAST_DAYS) -> list[dict[str, Any]]:
    crop_history = get_crop_price_history(market_history, crop)
    price_series = crop_history[MARKET_PRICE_COLUMN].astype(float)
    last_date = pd.to_datetime(crop_history.iloc[-1][DATE_COLUMN])

    try:
        fitted_model = ARIMA(price_series, order=(1, 1, 1)).fit()
        forecast_values = fitted_model.forecast(steps=forecast_days)
    except Exception:
        try:
            fitted_model = ARIMA(price_series, order=(1, 0, 0)).fit()
            forecast_values = fitted_model.forecast(steps=forecast_days)
        except Exception:
            forecast_values = np.repeat(float(price_series.iloc[-1]), forecast_days)

    forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days, freq="D")
    return [
        {
            "date": forecast_date.strftime("%Y-%m-%d"),
            "predicted_price_per_kg": round(max(0.0, float(price)), 2),
        }
        for forecast_date, price in zip(forecast_dates, forecast_values, strict=False)
    ]


def predict_recommendation(payload: dict[str, Any]) -> dict[str, Any]:
    loaded = load_models()
    validated_features = validate_request_payload(payload)
    market_history = load_market_dataset()
    crop_model = loaded["crop_model"]
    yield_model = loaded["yield_model"]
    metadata = loaded["metadata"]

    input_frame = pd.DataFrame([validated_features], columns=FEATURE_COLUMNS)
    recommended_crop = str(crop_model.predict(input_frame)[0])

    yield_frame = pd.DataFrame([{**validated_features, CROP_COLUMN: recommended_crop}])
    predicted_yield_kg_per_ha = round(float(yield_model.predict(yield_frame)[0]), 2)

    market_price = get_latest_market_price(market_history, recommended_crop)
    estimated_cost = estimate_cost(validated_features, recommended_crop, metadata["crop_costs"])
    expected_profit = round((predicted_yield_kg_per_ha * market_price) - estimated_cost, 2)
    price_forecast = forecast_prices(market_history, recommended_crop)

    return {
        "recommended_crop": recommended_crop,
        "predicted_yield_kg_per_ha": predicted_yield_kg_per_ha,
        "market_price_per_kg": market_price,
        "cost_per_ha": estimated_cost,
        "expected_profit": expected_profit,
        "price_forecast": price_forecast,
        "model_metrics": metadata["metrics"],
    }
