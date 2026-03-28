# SmartCropProfit Project Explanation

## 1. What This Project Does
SmartCropProfit is an end-to-end AI application that:
- Takes farm/environment inputs (`N`, `P`, `K`, `temperature`, `humidity`, `rainfall`)
- Recommends the best crop
- Predicts expected yield
- Estimates cost and profit per hectare
- Forecasts crop price for the next 7 days

It has two parts:
- Backend API using Flask + ML models
- Frontend dashboard using Streamlit

---

## 2. Folder Structure
- `backend/`
  - `app.py`: Flask API entry point
  - `recommendation_service.py`: main recommendation pipeline
  - `utils.py`: data loading, model training/loading, pricing, forecasting helpers
  - `train_models.py`: optional manual training script
  - `sample_data/`: CSV datasets
  - `models/`: saved model artifacts (`.pkl`)
- `frontend/`
  - `streamlit_app.py`: Streamlit app entry point
  - `api_client.py`: HTTP call to backend API
  - `ui.py`: page theme/cards/tables/details
  - `charts.py`: forecast and profit charts

---

## 3. Full End-to-End Flow
1. User enters inputs in Streamlit sidebar.
2. Frontend sends `POST /api/recommend` to Flask backend.
3. Backend validates input fields and types.
4. Backend loads models; if missing, it trains and saves them automatically.
5. Crop classifier predicts top crop candidates.
6. For each candidate crop:
   - Yield regressor predicts yield (`kg/ha`)
   - Latest price is fetched (API if configured, else CSV fallback)
   - Cost is estimated
   - Profit is computed
7. Backend returns:
   - Primary recommendation
   - Top-3 ranked recommendations
   - 7-day price forecast
   - model metrics
8. Frontend renders cards, table, and charts.

---

## 4. Backend Details

### `backend/app.py`
- Creates Flask app.
- Exposes:
  - `GET /health`: API + model status and metrics
  - `POST /api/recommend`: main prediction endpoint
- Handles common errors:
  - invalid JSON -> `400`
  - bad input -> `400`
  - unexpected errors -> `500`

### `backend/recommendation_service.py`
- Main orchestration module.
- `_predict_top_crops(...)`
  - Uses classifier probability (`predict_proba`) to rank top crops.
  - Falls back to one predicted class if probability is unavailable.
- `_build_crop_summary(...)`
  - Predicts yield for one crop candidate.
  - Gets market price and source.
  - Estimates cost and expected profit.
- `predict_recommendation(payload)`
  - Validates features
  - Loads models + market data
  - Builds top recommendations
  - Forecasts 7-day price for top crop
  - Returns final response JSON

### `backend/utils.py`
This is the utility backbone of the backend.

- **Config/Constants**
  - Data paths, model paths, column names, API settings, forecast days.

- **Dataset loaders**
  - `load_crop_dataset()`
  - `load_yield_dataset()`
  - `load_market_dataset()`
  These enforce required columns and numeric conversion.

- **Model training**
  - `build_crop_classifier(...)`: RandomForestClassifier
  - `build_yield_regressor(...)`: RandomForestRegressor with one-hot crop encoding
  - `_augment_tabular_data(...)`: synthetic augmentation with controlled noise
  - `train_and_save_models()`: trains and writes model files + metadata
  - `load_models()`: auto-trains if artifacts are missing

- **Validation**
  - `validate_request_payload(...)`: ensures all fields exist, numeric, non-negative.

- **Market price lookup**
  - Tries custom Agmarknet-style API if configured.
  - Tries data.gov Agmarknet resource if configured.
  - Falls back to local `market_prices.csv`.

- **Cost & profit**
  - `estimate_cost(...)`: base crop cost + nutrient/climate adjustments.

- **Forecasting**
  - `forecast_prices(...)`: ARIMA forecast for 7 days.
  - Fallback logic exists for model failures and sparse history.

### `backend/train_models.py`
- Optional standalone script for manual training and saving artifacts.
- Prints model save locations and evaluation metrics.

---

## 5. Frontend Details

### `frontend/streamlit_app.py`
- Main UI flow.
- Sidebar input controls for all required features.
- Calls backend API on button click.
- Handles network/API errors and displays useful messages.
- Renders:
  - Primary recommendation cards
  - Top-3 table
  - Profit comparison chart
  - 7-day forecast chart

### `frontend/api_client.py`
- Keeps API call logic in one place:
  - `call_recommendation_api(api_url, payload)`

### `frontend/ui.py`
- Injects CSS for visual style.
- Renders hero section.
- Renders metric cards and details.
- Builds and displays top recommendation table.

### `frontend/charts.py`
- Converts forecast list to DataFrame.
- Draws:
  - forecast line chart
  - top-crop profit bar chart

---

## 6. Input and Output Contracts

### Input JSON (to backend)
```json
{
  "N": 80,
  "P": 40,
  "K": 40,
  "temperature": 26,
  "humidity": 78,
  "rainfall": 180
}
```

### Output JSON (simplified)
```json
{
  "recommended_crop": "rice",
  "predicted_yield_kg_per_ha": 3400.25,
  "market_price_per_kg": 32.2,
  "cost_per_ha": 47010.4,
  "expected_profit": 62357.65,
  "top_recommendations": [...],
  "price_forecast": [...],
  "model_metrics": {
    "crop_accuracy": 0.95,
    "yield_mae": 120.45,
    "yield_rmse": 168.33
  }
}
```

---

## 7. Dependencies
From `backend/requirements.txt`:
- Flask
- streamlit
- requests
- pandas
- numpy
- scikit-learn
- statsmodels
- matplotlib
- joblib

---

## 8. How to Run (Typical)
1. Install dependencies:
   - `pip install -r backend/requirements.txt`
2. Start backend:
   - `python backend/app.py`
3. Start frontend (new terminal):
   - `streamlit run frontend/streamlit_app.py`
4. Open Streamlit URL and test with inputs.

---

## 9. Notes About Current Repo State
- Backend expects separate model artifacts:
  - `crop_classifier.pkl`
  - `yield_regressor.pkl`
  - `model_metadata.pkl`
- If missing, backend auto-trains on first request.
- `backend/models/crop_profit_models.pkl` exists but is not used by current loading logic.

