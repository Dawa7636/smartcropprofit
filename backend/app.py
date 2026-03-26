from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, request


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from utils import FEATURE_COLUMNS, load_models, predict_recommendation


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    @app.get("/health")
    def health_check():
        models = load_models()
        return jsonify(
            {
                "status": "ok",
                "available_features": FEATURE_COLUMNS,
                "model_metrics": models["metadata"]["metrics"],
            }
        )

    @app.post("/api/recommend")
    def recommend():
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Request body must be valid JSON."}), 400

        try:
            result = predict_recommendation(payload)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:  # pragma: no cover
            return jsonify({"error": f"Internal server error: {exc}"}), 500

        return jsonify(result), 200

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
