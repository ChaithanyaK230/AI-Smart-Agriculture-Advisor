"""Flask API for crop and fertilizer recommendations."""

import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add project root to path so utils can be imported
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.fertilizer_recommender import recommend_fertilizer, recommend_for_crop
from utils.data_loader import get_feature_columns

app = Flask(__name__)
CORS(app)

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "crop_model.pkl")
model = None


def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Model not found. Run 'python train_model.py' first."
            )
        model = joblib.load(MODEL_PATH)
    return model


@app.route("/")
def home():
    return jsonify({"message": "AI Smart Agriculture Advisor API is running."})


@app.route("/predict-crop", methods=["POST"])
def predict_crop():
    """Predict the best crop based on soil and environmental conditions."""
    data = request.get_json()
    features = get_feature_columns()

    # Validate input
    missing = [f for f in features if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        values = np.array([[float(data[f]) for f in features]])
    except (ValueError, TypeError):
        return jsonify({"error": "All input values must be numbers."}), 400

    clf = get_model()
    prediction = clf.predict(values)[0]

    # Get top 3 predictions with probabilities
    probabilities = clf.predict_proba(values)[0]
    classes = clf.classes_
    top_indices = probabilities.argsort()[-3:][::-1]
    top_crops = [
        {"crop": classes[i], "probability": round(float(probabilities[i]) * 100, 2)}
        for i in top_indices
    ]

    return jsonify({
        "recommended_crop": prediction,
        "top_3": top_crops,
    })


@app.route("/recommend-fertilizer", methods=["POST"])
def fertilizer():
    """Recommend fertilizer based on N, P, K values."""
    data = request.get_json()
    required = ["N", "P", "K"]

    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        n = float(data["N"])
        p = float(data["P"])
        k = float(data["K"])
    except (ValueError, TypeError):
        return jsonify({"error": "N, P, K must be numbers."}), 400

    result = recommend_fertilizer(n, p, k)
    return jsonify(result)


@app.route("/recommend-fertilizer-for-crop", methods=["POST"])
def fertilizer_for_crop():
    """Recommend fertilizer based on N, P, K values for a specific crop."""
    data = request.get_json()
    required = ["N", "P", "K", "crop"]

    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        n = float(data["N"])
        p = float(data["P"])
        k = float(data["K"])
    except (ValueError, TypeError):
        return jsonify({"error": "N, P, K must be numbers."}), 400

    result = recommend_for_crop(n, p, k, data["crop"])
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
