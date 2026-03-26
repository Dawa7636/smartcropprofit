from __future__ import annotations

import pandas as pd
import streamlit as st


def apply_custom_theme() -> None:
    # CSS is injected once to create a more distinctive visual identity.
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at 10% 10%, #dcfce7 0%, #f0fdf4 35%, #ecfeff 100%);
        }
        .hero {
            padding: 1.1rem 1.2rem;
            border-radius: 14px;
            background: linear-gradient(135deg, #064e3b, #065f46);
            color: #ecfdf5;
            box-shadow: 0 12px 28px rgba(6, 95, 70, 0.24);
            margin-bottom: 1rem;
        }
        .card {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid #bbf7d0;
            border-radius: 12px;
            padding: 0.8rem 1rem;
        }
        .card-title {
            font-size: 0.85rem;
            color: #166534;
            margin-bottom: 0.25rem;
        }
        .card-value {
            font-size: 1.25rem;
            font-weight: 700;
            color: #052e16;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h2 style="margin:0;">SmartCropProfit</h2>
            <p style="margin:0.35rem 0 0 0;">
                AI-based crop ranking with yield, profit analysis, and price outlook.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_cards(result: dict) -> None:
    st.subheader("Primary Recommendation")
    card_col1, card_col2, card_col3, card_col4 = st.columns(4)
    card_col1.markdown(
        f"<div class='card'><div class='card-title'>Recommended Crop</div><div class='card-value'>{result['recommended_crop'].title()}</div></div>",
        unsafe_allow_html=True,
    )
    card_col2.markdown(
        "<div class='card'><div class='card-title'>Predicted Yield</div>"
        f"<div class='card-value'>{result['predicted_yield_kg_per_ha']:,.2f} kg/ha</div></div>",
        unsafe_allow_html=True,
    )
    card_col3.markdown(
        "<div class='card'><div class='card-title'>Market Price</div>"
        f"<div class='card-value'>{result['market_price_per_kg']:,.2f} /kg</div></div>",
        unsafe_allow_html=True,
    )
    card_col4.markdown(
        "<div class='card'><div class='card-title'>Expected Profit</div>"
        f"<div class='card-value'>{result['expected_profit']:,.2f}</div></div>",
        unsafe_allow_html=True,
    )


def render_details(result: dict) -> None:
    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:
        st.markdown("### Farm Economics")
        st.write(f"Estimated cost per hectare: `{result['cost_per_ha']:,.2f}`")
        st.write(f"Expected profit: `{result['expected_profit']:,.2f}`")
        st.write(f"Current market price: `{result['market_price_per_kg']:,.2f}` per kg")
        if result.get("market_price_source"):
            st.write(f"Price source: `{result['market_price_source']}`")

    with detail_col2:
        st.markdown("### Model Snapshot")
        metrics = result.get("model_metrics", {})
        st.write(f"Crop accuracy: `{metrics.get('crop_accuracy', 'n/a')}`")
        st.write(f"Yield MAE: `{metrics.get('yield_mae', 'n/a')}` kg/ha")
        st.write(f"Yield RMSE: `{metrics.get('yield_rmse', 'n/a')}` kg/ha")


def render_top_recommendations_table(top_recommendations: list[dict]) -> None:
    st.subheader("Top 3 Crop Recommendations")
    if not top_recommendations:
        st.info("No ranked recommendations were returned by the API.")
        return

    ranked_df = pd.DataFrame(top_recommendations).copy()
    ranked_df["crop"] = ranked_df["crop"].astype(str).str.title()
    ranked_df["confidence"] = (ranked_df["confidence"] * 100).round(2)
    ranked_df = ranked_df.rename(
        columns={
            "crop": "Crop",
            "confidence": "Confidence (%)",
            "predicted_yield_kg_per_ha": "Predicted Yield (kg/ha)",
            "market_price_per_kg": "Market Price (/kg)",
            "cost_per_ha": "Estimated Cost",
            "expected_profit": "Expected Profit",
        }
    )

    st.dataframe(ranked_df, use_container_width=True, hide_index=True)
