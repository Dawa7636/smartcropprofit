from __future__ import annotations

import requests
import streamlit as st

from api_client import DEFAULT_API_URL, call_recommendation_api
from charts import build_forecast_dataframe, render_forecast_chart, render_profit_comparison_chart
from ui import (
    apply_custom_theme,
    render_details,
    render_hero,
    render_result_cards,
    render_top_recommendations_table,
)


def main() -> None:
    st.set_page_config(page_title="SmartCropProfit", layout="wide")
    apply_custom_theme()
    render_hero()

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

    top_recommendations = result.get("top_recommendations", [])

    render_result_cards(result)
    render_top_recommendations_table(top_recommendations)
    render_profit_comparison_chart(top_recommendations)
    render_details(result)
    render_forecast_chart(build_forecast_dataframe(result.get("price_forecast", [])))


if __name__ == "__main__":
    main()
