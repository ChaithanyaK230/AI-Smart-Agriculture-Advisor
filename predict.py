"""
Crop Prediction Script
=======================
This script loads the trained RandomForest model and predicts the best crop
based on soil and environmental conditions you provide.

How prediction works (behind the scenes):
  1. You provide 7 input values (N, P, K, temperature, humidity, ph, rainfall)
  2. The saved model is loaded from disk (no re-training needed)
  3. Your inputs are passed to all 100 decision trees inside the RandomForest
  4. Each tree independently votes for a crop based on the input values
  5. The crop with the most votes wins (majority voting)
  6. The model also outputs probability scores -- how confident it is about
     each crop, calculated as (trees that voted for crop X) / (total trees)
"""

# ─── Step 1: Import Libraries ───────────────────────────────────────────────────
# joblib   -- loads the saved model file (.pkl) back into memory
# numpy    -- creates the numeric array that the model expects as input
# os       -- constructs file paths that work on any operating system

import joblib
import numpy as np
import pandas as pd
import os


def load_model():
    """
    Load the trained model from the models/ folder.

    joblib.load() reverses what joblib.dump() did during training --
    it reads the .pkl file and reconstructs the exact same RandomForest
    object, with all 100 trees and their learned rules intact.
    """
    model_path = os.path.join(os.path.dirname(__file__), "models", "crop_model.pkl")

    if not os.path.exists(model_path):
        print("ERROR: Model file not found!")
        print("Run 'python train_model.py' first to train and save the model.")
        return None

    model = joblib.load(model_path)
    return model


def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    """
    Predict the best crop for the given soil and weather conditions.

    Parameters:
        nitrogen    (float): Nitrogen content in soil (kg/ha)
        phosphorus  (float): Phosphorus content in soil (kg/ha)
        potassium   (float): Potassium content in soil (kg/ha)
        temperature (float): Average temperature in Celsius
        humidity    (float): Relative humidity in percentage
        ph          (float): pH value of the soil (0-14 scale)
        rainfall    (float): Annual rainfall in mm

    How the input flows through the model:
        1. The 7 values are packed into a 2D array: [[N, P, K, temp, hum, ph, rain]]
           The model expects a 2D array because it can handle multiple samples at once.
           We have 1 sample, so it's a list inside a list.

        2. model.predict() returns the single best crop (highest vote count).

        3. model.predict_proba() returns a probability for every crop.
           Example: [0.01, 0.02, ..., 0.85, ..., 0.01]
           Each number = what fraction of the 100 trees voted for that crop.

        4. model.classes_ maps each position in the probability array to a crop name.
           Example: classes_[0] = "apple", classes_[1] = "banana", etc.
    """
    model = load_model()
    if model is None:
        return None

    # Pack the 7 input values into a DataFrame with the same column names used
    # during training. This avoids sklearn warnings about missing feature names.
    input_data = pd.DataFrame([[nitrogen, phosphorus, potassium,
                                temperature, humidity, ph, rainfall]],
                              columns=["N", "P", "K", "temperature",
                                       "humidity", "ph", "rainfall"])

    # Get the single best prediction (the crop with the most tree votes)
    prediction = model.predict(input_data)[0]

    # Get probability scores for ALL crops
    # predict_proba returns shape (1, 22) -- 1 sample, 22 crop probabilities
    probabilities = model.predict_proba(input_data)[0]

    # model.classes_ is an array of crop names in the same order as probabilities
    # Example: if classes_ = ["apple", "banana", ...] and probabilities = [0.01, 0.85, ...]
    #          then banana has 85% probability
    classes = model.classes_

    # Sort by probability (highest first) and take the top 5
    # argsort() returns indices that would sort the array in ascending order
    # [-5:] takes the last 5 (highest), [::-1] reverses to descending order
    top_indices = probabilities.argsort()[-5:][::-1]

    top_crops = []
    for idx in top_indices:
        top_crops.append({
            "crop": classes[idx],
            "confidence": round(float(probabilities[idx]) * 100, 2)
        })

    return {
        "best_crop": prediction,
        "top_5": top_crops,
    }


# ─── Step 2: Main Script ────────────────────────────────────────────────────────
# When you run this file directly (python predict.py), it asks for your
# soil and weather values, then shows the prediction.

if __name__ == "__main__":
    print("=" * 60)
    print("CROP PREDICTION")
    print("=" * 60)
    print("Enter your soil and environmental conditions:\n")

    # input() reads text from the terminal
    # float() converts that text to a decimal number
    try:
        n    = float(input("  Nitrogen (N)      : "))
        p    = float(input("  Phosphorus (P)    : "))
        k    = float(input("  Potassium (K)     : "))
        temp = float(input("  Temperature (C)   : "))
        hum  = float(input("  Humidity (%)      : "))
        ph   = float(input("  pH                : "))
        rain = float(input("  Rainfall (mm)     : "))
    except ValueError:
        print("\nERROR: Please enter valid numbers only.")
        exit(1)

    print("\nPredicting...")
    result = predict_crop(n, p, k, temp, hum, ph, rain)

    if result is None:
        exit(1)

    # Display the result
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"\n  Best crop: {result['best_crop'].upper()}")
    print(f"\n  Top 5 predictions:")

    for i, crop_info in enumerate(result["top_5"], start=1):
        bar = "#" * int(crop_info["confidence"] / 2.5)  # scale bar to ~40 chars max
        print(f"    {i}. {crop_info['crop']:<14} {crop_info['confidence']:>6.2f}%  {bar}")

    print()
