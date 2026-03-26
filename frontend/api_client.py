from __future__ import annotations

from typing import Any

import requests


DEFAULT_API_URL = "http://localhost:5000/api/recommend"


def call_recommendation_api(api_url: str, payload: dict[str, float]) -> dict[str, Any]:
    """
    Centralized API client for the Streamlit app.
    Keeping this isolated helps us test/replace backend calls later.
    """
    response = requests.post(api_url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()
