from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                "https://land-price-predication-b-2.vercel.app",
                "http://localhost:5173"
            ]
        }
    }
)

# ── Load ML model & encoder ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "bestmodel.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)

# Feature order must match training
FEATURE_COLUMNS = [
    "city", "state", "city_tier", "zoning", "land_area_sqft",
    "dist_city_center_km", "dist_highway_km", "dist_transport_km",
    "dist_amenities_km", "historical_growth_pct", "population_growth_pct",
    "road_quality_score", "utility_access", "govt_dev_plan", "flood_risk"
]

CATEGORICAL_COLS = ["city", "state", "zoning"]


# ── Prediction Endpoint ───────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Validate required fields
        missing = [f for f in FEATURE_COLUMNS + ["current_price"] if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        current_price = float(data["current_price"])

        # Build input dataframe
        row = {col: data[col] for col in FEATURE_COLUMNS}
        input_df = pd.DataFrame([row])

        # Enforce numeric types
        numeric_cols = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLS]
        for col in numeric_cols:
            input_df[col] = pd.to_numeric(input_df[col])

        # Apply ordinal encoding to categorical columns
        input_df[CATEGORICAL_COLS] = encoder.transform(input_df[CATEGORICAL_COLS])

        # Predict
        predicted_price = float(model.predict(input_df)[0])
        profit = predicted_price - current_price
        roi = (profit / current_price) * 100 if current_price != 0 else 0

        return jsonify({
            "current_price_per_sqft": round(current_price, 2),
            "future_price_per_sqft": round(predicted_price, 2),
            "expected_profit_per_sqft": round(profit, 2),
            "roi_percent": round(roi, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Health Check ──────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "XGBoost Land Price Predictor"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
# Trigger reload
