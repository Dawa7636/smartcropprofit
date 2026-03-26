from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


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
        color="#0f766e",
        linewidth=2.5,
        marker="o",
        markersize=6,
    )
    ax.fill_between(
        forecast_df["date"],
        forecast_df["predicted_price_per_kg"],
        color="#14b8a6",
        alpha=0.18,
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


def render_profit_comparison_chart(top_recommendations: list[dict]) -> None:
    st.subheader("Top 3 Profit Comparison")
    if not top_recommendations:
        st.info("No recommendation data available for comparison.")
        return

    ranked_df = pd.DataFrame(top_recommendations).copy()
    ranked_df["crop"] = ranked_df["crop"].astype(str).str.title()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(
        ranked_df["crop"],
        ranked_df["expected_profit"],
        color=["#14532d", "#15803d", "#22c55e"][: len(ranked_df)],
        alpha=0.9,
    )
    ax.set_title("Expected Profit by Crop", fontsize=14, fontweight="bold")
    ax.set_ylabel("Profit per hectare")
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:,.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#1f2937",
        )

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
