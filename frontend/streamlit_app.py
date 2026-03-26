from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st


DEFAULT_API_URL = "http://localhost:5000/api/recommend"


def call_recommendation_api(api_url: str, payload: dict[str, float]) -> dict:
    response = requests.post(api_url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def build_forecast_dataframe(forecast: list[dict]) -> pd.DataFrame:
    if not forecast:
        return pd.DataFrame(columns=["date", "predicted_price_per_kg"])

    forecast_df = pd.DataFrame(forecast)
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
    forecast_df["predicted_price_per_kg"] = pd.to_numeric(
        forecast_df["predicted_price_per_kg"], errors="coerce"
    )
    return forecast_df.dropna().reset_index(drop=True)


def render_forecast_chart(forecast_df: pd.DataFrame) -> None:
    st.subheader("7-Day Price Forecast")
    if forecast_df.empty:
        st.info("No forecast data available for plotting.")
        return

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(
        forecast_df["date"],
        forecast_df["predicted_price_per_kg"],
        color="#2E8B57",
        linewidth=2.5,
        marker="o",
        markersize=6,
    )
    ax.fill_between(
        forecast_df["date"],
        forecast_df["predicted_price_per_kg"],
        color="#2E8B57",
        alpha=0.12,
    )
    ax.set_title("Forecasted Market Prices", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price per kg")
    ax.grid(alpha=0.25, linestyle="--")
    plt.xticks(rotation=30)
    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)
    plt.close(fig)


def render_result_cards(result: dict) -> None:
    st.subheader("Recommendation Summary")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Recommended Crop", result["recommended_crop"].title())
    metric_col2.metric("Predicted Yield", f"{result['predicted_yield_kg_per_ha']:,.2f} kg/ha")
    metric_col3.metric("Market Price", f"{result['market_price_per_kg']:,.2f} per kg")
    metric_col4.metric("Expected Profit", f"{result['expected_profit']:,.2f}")


def render_details(result: dict) -> None:
    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:
        st.markdown("### Farm Economics")
        st.write(f"Estimated cost per hectare: `{result['cost_per_ha']:,.2f}`")
        st.write(f"Expected profit: `{result['expected_profit']:,.2f}`")
        st.write(f"Current market price: `{result['market_price_per_kg']:,.2f}` per kg")

    with detail_col2:
        st.markdown("### Model Snapshot")
        metrics = result.get("model_metrics", {})
        st.write(f"Crop accuracy: `{metrics.get('crop_accuracy', 'n/a')}`")
        st.write(f"Yield MAE: `{metrics.get('yield_mae', 'n/a')}` kg/ha")
        st.write(f"Yield RMSE: `{metrics.get('yield_rmse', 'n/a')}` kg/ha")


def main() -> None:
    st.set_page_config(page_title="SmartCropProfit", layout="wide")
    st.title("SmartCropProfit")
    st.caption("Crop recommendation, yield prediction, profit analysis, and market price forecasting.")

    with st.sidebar:
        st.header("Input Parameters")
        api_url = st.text_input("Flask API URL", value=DEFAULT_API_URL)
        nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, value=80.0, step=1.0)
        phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, value=40.0, step=1.0)
        potassium = st.number_input("Potassium (K)", min_value=0.0, value=40.0, step=1.0)
        temperature = st.number_input("Temperature (deg C)", min_value=0.0, value=26.0, step=0.1)
        humidity = st.number_input("Humidity (%)", min_value=0.0, value=78.0, step=0.1)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=180.0, step=0.1)
        submitted = st.button("Get Recommendation", use_container_width=True)

    intro_col1, intro_col2 = st.columns([1.3, 1.0])
    with intro_col1:
        st.markdown("### What this app does")
        st.write(
            "Enter soil nutrients and weather conditions to get a crop recommendation, expected yield, "
            "current market price, estimated profit, and a 7-day price forecast."
        )
    with intro_col2:
        st.markdown("### Required Inputs")
        st.write("N, P, K, temperature, humidity, and rainfall are sent to the Flask backend for inference.")

    if not submitted:
        st.info("Set the values in the sidebar and click `Get Recommendation`.")
        return

    payload = {
        "N": nitrogen,
        "P": phosphorus,
        "K": potassium,
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
    }

    try:
        result = call_recommendation_api(api_url, payload)
    except requests.HTTPError as exc:
        detail = exc.response.json() if exc.response is not None else {"error": str(exc)}
        st.error(detail.get("error", "The Flask API returned an unexpected error."))
        return
    except requests.RequestException as exc:
        st.error(f"Unable to reach the Flask API: {exc}")
        return

    render_result_cards(result)
    render_details(result)
    render_forecast_chart(build_forecast_dataframe(result.get("price_forecast", [])))


if __name__ == "__main__":
    main()
