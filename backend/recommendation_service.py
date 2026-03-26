from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils import (
    CROP_COLUMN,
    COST_COLUMN,
    FEATURE_COLUMNS,
    estimate_cost,
    forecast_prices,
    get_latest_market_price,
    load_market_dataset,
    load_models,
    load_yield_dataset,
    validate_request_payload,
)


def _predict_top_crops(
    crop_model: Any,
    input_frame: pd.DataFrame,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """
    Return top crop candidates with model confidence.
    Falls back to the single predicted class when probabilities are unavailable.
    """
    if hasattr(crop_model, "predict_proba") and hasattr(crop_model, "classes_"):
        probabilities = np.asarray(crop_model.predict_proba(input_frame)[0], dtype=float)
        classes = np.asarray(crop_model.classes_, dtype=str)
        sorted_indexes = np.argsort(probabilities)[::-1]
        top_indexes = sorted_indexes[: min(top_k, len(sorted_indexes))]
        return [(str(classes[index]), round(float(probabilities[index]), 4)) for index in top_indexes]

    predicted_crop = str(crop_model.predict(input_frame)[0])
    return [(predicted_crop, 1.0)]


def _build_crop_summary(
    crop: str,
    confidence: float,
    validated_features: dict[str, float],
    yield_model: Any,
    market_history: pd.DataFrame,
    crop_costs: dict[str, float],
) -> dict[str, Any]:
    # Yield model expects all environmental features plus one categorical crop field.
    yield_frame = pd.DataFrame([{**validated_features, CROP_COLUMN: crop}])
    predicted_yield_kg_per_ha = round(float(yield_model.predict(yield_frame)[0]), 2)

    market_price, price_source = get_latest_market_price(market_history, crop)
    estimated_cost = estimate_cost(validated_features, crop, crop_costs)
    expected_profit = round((predicted_yield_kg_per_ha * market_price) - estimated_cost, 2)

    return {
        "crop": crop,
        "confidence": confidence,
        "predicted_yield_kg_per_ha": predicted_yield_kg_per_ha,
        "market_price_per_kg": market_price,
        "market_price_source": price_source,
        "cost_per_ha": estimated_cost,
        "expected_profit": expected_profit,
    }


def predict_recommendation(payload: dict[str, Any]) -> dict[str, Any]:
    loaded = load_models()
    validated_features = validate_request_payload(payload)
    market_history = load_market_dataset()
    crop_model = loaded["crop_model"]
    yield_model = loaded["yield_model"]
    metadata = loaded["metadata"]
    crop_costs = metadata.get("crop_costs")
    if not crop_costs:
        yield_df = load_yield_dataset()
        crop_costs = yield_df.groupby(CROP_COLUMN)[COST_COLUMN].mean().round(2).to_dict()

    input_frame = pd.DataFrame([validated_features], columns=FEATURE_COLUMNS)
    top_candidates = _predict_top_crops(crop_model, input_frame, top_k=3)
    ranked_recommendations = [
        _build_crop_summary(
            crop=crop,
            confidence=confidence,
            validated_features=validated_features,
            yield_model=yield_model,
            market_history=market_history,
            crop_costs=crop_costs,
        )
        for crop, confidence in top_candidates
    ]

    primary = ranked_recommendations[0]
    price_forecast = forecast_prices(market_history, primary["crop"])

    return {
        "recommended_crop": primary["crop"],
        "predicted_yield_kg_per_ha": primary["predicted_yield_kg_per_ha"],
        "market_price_per_kg": primary["market_price_per_kg"],
        "market_price_source": primary["market_price_source"],
        "cost_per_ha": primary["cost_per_ha"],
        "expected_profit": primary["expected_profit"],
        "top_recommendations": ranked_recommendations,
        "price_forecast": price_forecast,
        "model_metrics": metadata["metrics"],
    }
