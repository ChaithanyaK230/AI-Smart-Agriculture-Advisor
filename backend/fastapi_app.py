"""
FastAPI Backend for AI Smart Agriculture Advisor
==================================================
This file creates a REST API using FastAPI that serves two endpoints:
  1. POST /predict-crop          -- predict the best crop for given conditions
  2. POST /recommend-fertilizer  -- get fertilizer advice based on N, P, K

WHY FASTAPI OVER FLASK?
-----------------------
- Automatic docs    -- visit /docs to get an interactive Swagger UI for free
- Type validation   -- Pydantic models validate input before your code runs
- Async support     -- can handle more concurrent requests efficiently
- Faster            -- built on Starlette, one of the fastest Python frameworks

API STRUCTURE:
--------------
  Client (frontend/Postman)
       |
       |  sends JSON via POST request
       v
  FastAPI receives the request
       |
       |  Pydantic validates the input (are all fields present? correct types?)
       |  If invalid --> automatic 422 error with details
       v
  Your function runs (predict crop / recommend fertilizer)
       |
       v
  FastAPI returns JSON response to the client
"""

# ─── Step 1: Import Libraries ───────────────────────────────────────────────────
# FastAPI       -- the web framework that handles HTTP requests/responses
# CORSMiddleware -- allows the frontend (different origin) to call this API
#                   Without CORS, browsers block cross-origin requests
# BaseModel     -- Pydantic base class for defining request/response schemas
#                  It auto-validates that incoming JSON has the right fields & types
# Field         -- adds validation rules (min/max values) and descriptions
# Optional      -- marks a field as not required

import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path so utils/ can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.fertilizer_recommender import recommend_fertilizer, recommend_for_crop
from utils.data_loader import get_feature_columns


# ─── Step 2: Define Request & Response Schemas ──────────────────────────────────
# Pydantic models define the SHAPE of data going in and out of the API.
# Think of them as contracts:
#   "I expect JSON with these exact fields and these types"
#
# Benefits:
#   - If a field is missing or wrong type, FastAPI returns a clear error
#   - The /docs page auto-generates example values from these models
#   - Your code can trust that data is valid (no manual checking needed)


class CropPredictionRequest(BaseModel):
    """
    Input schema for crop prediction.
    All 7 soil/weather parameters are required.

    Field() adds constraints:
      ge=0   --> value must be >= 0 (greater than or equal)
      le=14  --> value must be <= 14 (for pH scale)
      description --> shows up in the /docs page
    """
    N: float = Field(..., ge=0, description="Nitrogen content in soil (kg/ha)")
    P: float = Field(..., ge=0, description="Phosphorus content in soil (kg/ha)")
    K: float = Field(..., ge=0, description="Potassium content in soil (kg/ha)")
    temperature: float = Field(..., description="Average temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity (%)")
    ph: float = Field(..., ge=0, le=14, description="Soil pH value (0-14)")
    rainfall: float = Field(..., ge=0, description="Annual rainfall in mm")

    # This inner class provides example values for the /docs page
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "N": 90, "P": 42, "K": 43,
                    "temperature": 21.0, "humidity": 82.0,
                    "ph": 6.5, "rainfall": 203.0
                }
            ]
        }
    }


class CropPredictionItem(BaseModel):
    """One crop with its confidence percentage."""
    crop: str
    probability: float


class CropPredictionResponse(BaseModel):
    """Output schema for crop prediction."""
    recommended_crop: str
    top_3: list[CropPredictionItem]


class FertilizerRequest(BaseModel):
    """
    Input schema for fertilizer recommendation.
    Only N, P, K are required. Optionally specify a crop name for
    crop-specific advice.
    """
    N: float = Field(..., ge=0, description="Nitrogen content in soil (kg/ha)")
    P: float = Field(..., ge=0, description="Phosphorus content in soil (kg/ha)")
    K: float = Field(..., ge=0, description="Potassium content in soil (kg/ha)")
    crop: Optional[str] = Field(None, description="Crop name for crop-specific advice (optional)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"N": 30, "P": 20, "K": 25, "crop": "rice"}
            ]
        }
    }


# ─── Step 3: Create the FastAPI App ─────────────────────────────────────────────
# FastAPI() creates the application instance.
# title, description, version appear on the /docs page.

app = FastAPI(
    title="AI Smart Agriculture Advisor",
    description="Predict the best crop and get fertilizer recommendations based on soil conditions.",
    version="1.0.0",
)

# ─── Step 4: Configure CORS ─────────────────────────────────────────────────────
# CORS (Cross-Origin Resource Sharing) controls which websites can call this API.
# allow_origins=["*"] means ANY website can call it (fine for development).
# In production, you'd list specific domains: ["https://yoursite.com"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # which origins (websites) can access
    allow_credentials=True,    # allow cookies to be sent
    allow_methods=["*"],       # allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],       # allow all headers
)


# ─── Step 5: Load the ML Model ──────────────────────────────────────────────────
# We load the model once when the server starts, not on every request.
# This is much faster than loading from disk on each prediction.

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "crop_model.pkl")
model = None


def get_model():
    """
    Load the trained model (lazy loading).
    Lazy = load only when first needed, then keep in memory.
    """
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Model not found. Run 'python train_model.py' first."
            )
        model = joblib.load(MODEL_PATH)
    return model


# ─── Step 6: Define API Endpoints ───────────────────────────────────────────────
#
# ANATOMY OF A FASTAPI ENDPOINT:
#
#   @app.post("/predict-crop")        <-- route decorator (URL + HTTP method)
#   def predict_crop(                  <-- function name (becomes operation ID)
#       data: CropPredictionRequest    <-- Pydantic model auto-validates the JSON body
#   ) -> CropPredictionResponse:       <-- return type (for docs generation)
#
# When a POST request hits /predict-crop:
#   1. FastAPI reads the JSON body
#   2. Pydantic validates it against CropPredictionRequest
#   3. If valid, your function runs with validated data
#   4. If invalid, FastAPI returns 422 with error details (you write zero validation code)


@app.get("/")
def home():
    """Health check endpoint. Returns a simple status message."""
    return {"message": "AI Smart Agriculture Advisor API is running."}


@app.post("/predict-crop", response_model=CropPredictionResponse)
def predict_crop(data: CropPredictionRequest):
    """
    Predict the best crop based on soil and environmental conditions.

    How it works:
      1. Takes your 7 input values (N, P, K, temperature, humidity, ph, rainfall)
      2. Passes them to the trained RandomForest model
      3. Each of the 100 trees votes for a crop
      4. Returns the top-voted crop + top 3 predictions with confidence scores
    """
    # Build feature array in the same column order the model was trained on
    features = get_feature_columns()  # ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

    # Create a DataFrame with named columns so sklearn doesn't warn about
    # missing feature names (the model was trained with named columns)
    input_data = pd.DataFrame(
        [[data.N, data.P, data.K, data.temperature,
          data.humidity, data.ph, data.rainfall]],
        columns=features
    )

    # Get the model and make predictions
    clf = get_model()
    prediction = clf.predict(input_data)[0]  # single best crop

    # predict_proba returns probability for each crop class
    # Example: [0.01, 0.02, ..., 0.85, ...] across all 22 crops
    probabilities = clf.predict_proba(input_data)[0]
    classes = clf.classes_

    # Get indices of top 3 highest probabilities
    # argsort() sorts ascending, [-3:] takes last 3, [::-1] reverses to descending
    top_indices = probabilities.argsort()[-3:][::-1]
    top_crops = [
        CropPredictionItem(
            crop=classes[i],
            probability=round(float(probabilities[i]) * 100, 2)
        )
        for i in top_indices
    ]

    return CropPredictionResponse(
        recommended_crop=prediction,
        top_3=top_crops,
    )


@app.post("/recommend-fertilizer")
def fertilizer(data: FertilizerRequest):
    """
    Recommend fertilizer based on soil nutrient levels (N, P, K).

    Two modes:
      1. General mode (no crop specified):
         Compares N, P, K against universal thresholds and suggests fertilizers.

      2. Crop-specific mode (crop name provided):
         Compares N, P, K against that crop's ideal range.
         Calculates exact deficits and recommends targeted fertilizers.

    How it works:
      - Each nutrient is classified as "low", "adequate", or "high"
      - For deficiencies, specific fertilizer products are suggested
      - For excesses, reduction strategies are recommended
    """
    # If a crop name is provided, use crop-specific analysis
    if data.crop:
        result = recommend_for_crop(data.N, data.P, data.K, data.crop)
    else:
        result = recommend_fertilizer(data.N, data.P, data.K)

    return result


# ─── Step 7: Run the Server ─────────────────────────────────────────────────────
# uvicorn is the ASGI server that runs FastAPI (like a waiter serving requests).
#
# host="0.0.0.0" -- listen on all network interfaces (accessible from other devices)
# port=8000      -- the port number (default for FastAPI, Flask uses 5000)
# reload=True    -- auto-restart when you edit code (development only)
#
# After running, visit:
#   http://localhost:8000       -- home endpoint
#   http://localhost:8000/docs  -- interactive Swagger UI (try your API in the browser!)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
