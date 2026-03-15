# AI Smart Agriculture Advisor

An intelligent crop and fertilizer recommendation system powered by machine learning. The system analyzes soil nutrients (N, P, K), weather conditions (temperature, humidity, rainfall), and soil pH to recommend the best crop to grow and the right fertilizers to apply.

Built with Python, scikit-learn, FastAPI, and a clean HTML/CSS/JS frontend -- designed to be beginner-friendly and easy to understand.

## Project Overview

Farmers often face the challenge of choosing the right crop for their soil and climate conditions. Poor crop selection leads to low yields, while incorrect fertilizer use wastes money and harms the environment.

This project solves both problems:

1. **Crop Prediction** -- A RandomForestClassifier trained on 2200 real agricultural samples predicts the best crop with 99.3% accuracy across 22 different crops.

2. **Fertilizer Recommendation** -- A rule-based system analyzes soil nutrient levels and compares them against crop-specific optimal ranges, then suggests specific fertilizer products to correct any deficiencies.

## Features

- **ML-Powered Crop Prediction** -- Predicts the best crop from 22 options using RandomForestClassifier (100 decision trees). Returns top 3 predictions with confidence percentages.
- **Smart Fertilizer Advice** -- Two modes: general analysis against universal thresholds, or crop-specific analysis that calculates exact nutrient deficits and suggests targeted fertilizers.
- **Exploratory Data Analysis** -- EDA script generates correlation heatmaps, feature distribution histograms, and box plots to understand the dataset.
- **Dual Backend Support** -- Both Flask (port 5000) and FastAPI (port 8000) backends included. FastAPI provides auto-generated Swagger docs at `/docs`.
- **Interactive Web Interface** -- Responsive HTML/CSS/JS frontend with forms, animated probability bars, nutrient status chips, and fertilizer cards.
- **Combined Workflow** -- "Get Crop + Fertilizer" button predicts the best crop and automatically fetches tailored fertilizer advice in one click.
- **Standalone Prediction Script** -- `predict.py` allows terminal-based predictions or can be imported into other Python code.
- **Feature Importance Analysis** -- Training script reveals which soil/weather factors matter most (rainfall and humidity dominate at ~44%).

## Technologies Used

| Category       | Technology                                          |
|----------------|-----------------------------------------------------|
| Language       | Python 3                                            |
| ML Framework   | scikit-learn (RandomForestClassifier)                |
| Data Handling  | pandas, NumPy                                       |
| Visualization  | matplotlib, seaborn                                 |
| Backend (v1)   | Flask, Flask-CORS                                   |
| Backend (v2)   | FastAPI, Uvicorn, Pydantic                          |
| Frontend       | HTML5, CSS3, JavaScript (Vanilla)                   |
| Model Storage  | joblib (.pkl serialization)                         |
| Dataset        | Crop Recommendation Dataset (Kaggle, 2200 samples) |

## Project Structure

```
AI-Smart-Agriculture-Advisor/
|
|-- data/                          # Dataset and EDA output
|   |-- Crop_recommendation.csv   # Main dataset (2200 rows, 22 crops)
|   |-- correlation_heatmap.png   # Feature correlation visualization
|   |-- feature_distributions.png # Histogram of each feature
|   |-- boxplots_by_crop.png      # Box plots showing feature ranges per crop
|
|-- models/                        # Trained ML models
|   |-- crop_model.pkl            # Saved RandomForest model
|
|-- backend/                       # API servers
|   |-- app.py                    # Flask backend (port 5000)
|   |-- fastapi_app.py            # FastAPI backend (port 8000, recommended)
|
|-- frontend/                      # Web interface
|   |-- index.html                # Single-page app (HTML + CSS + JS)
|
|-- utils/                         # Shared helper modules
|   |-- data_loader.py            # Load dataset and define feature columns
|   |-- fertilizer_recommender.py # Fertilizer logic + crop nutrient database
|
|-- train_model.py                 # Train and evaluate the ML model
|-- predict.py                     # Standalone prediction script (terminal/import)
|-- eda.py                         # Exploratory data analysis with visualizations
|-- requirements.txt               # Python dependencies
|-- README.md                      # Project documentation
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Steps

1. **Clone the repository**

```bash
git clone https://github.com/ChaithanyaK230/AI-Smart-Agriculture-Advisor.git
cd AI-Smart-Agriculture-Advisor
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Verify the dataset**

Ensure `data/Crop_recommendation.csv` is present. If not, download it from [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) and place it in the `data/` folder.

## How to Run the Project

### Step 1: Train the Model

```bash
python train_model.py
```

This will:
- Load the dataset (2200 samples, 22 crops)
- Split into 80% training / 20% testing
- Train a RandomForestClassifier with 100 trees
- Print accuracy (99.3%) and feature importance rankings
- Save the model to `models/crop_model.pkl`

### Step 2: Start the Backend

**Option A: FastAPI (recommended)**

```bash
cd backend
python fastapi_app.py
```

- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs

**Option B: Flask**

```bash
python backend/app.py
```

- API: http://localhost:5000

### Step 3: Open the Frontend

Open `frontend/index.html` in your browser. Enter soil parameters and click the buttons to get recommendations.

> **Note:** The frontend connects to port 8000 (FastAPI) by default. If using Flask, change `API_URL` in `index.html` to `http://localhost:5000`.

### Optional: Run EDA

```bash
python eda.py
```

Generates three charts in `data/` showing feature correlations, distributions, and per-crop patterns.

### Optional: Terminal Prediction

```bash
python predict.py
```

Enter soil values when prompted to get predictions directly in the terminal.

## API Endpoints

### `POST /predict-crop`

Predict the best crop based on soil and weather conditions.

**Request:**

```json
{
  "N": 90, "P": 42, "K": 43,
  "temperature": 21.0, "humidity": 82.0,
  "ph": 6.5, "rainfall": 203.0
}
```

**Response:**

```json
{
  "recommended_crop": "rice",
  "top_3": [
    { "crop": "rice", "probability": 99.0 },
    { "crop": "jute", "probability": 1.0 },
    { "crop": "pomegranate", "probability": 0.0 }
  ]
}
```

### `POST /recommend-fertilizer`

Get fertilizer advice based on N, P, K values. Optionally specify a crop for targeted advice.

**Request (general):**

```json
{ "N": 30, "P": 20, "K": 25 }
```

**Request (crop-specific):**

```json
{ "N": 30, "P": 20, "K": 25, "crop": "rice" }
```

**Response (crop-specific):**

```json
{
  "crop": "rice",
  "nutrient_analysis": {
    "N": { "soil_value": 30, "ideal_range": "60-100", "status": "deficit", "gap": 30 },
    "P": { "soil_value": 20, "ideal_range": "35-60", "status": "deficit", "gap": 15 },
    "K": { "soil_value": 25, "ideal_range": "35-50", "status": "deficit", "gap": 10 }
  },
  "fertilizer_plan": [
    {
      "nutrient": "N",
      "action": "Increase N by ~30 kg/ha",
      "fertilizers": [
        { "name": "Urea", "nutrient": "Nitrogen (46% N)", "description": "..." }
      ]
    }
  ]
}
```

## Dataset

The project uses the [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) from Kaggle.

| Property | Value |
|----------|-------|
| Total samples | 2200 |
| Crops | 22 (100 samples each, perfectly balanced) |
| Features | N, P, K, temperature, humidity, pH, rainfall |
| Target | Crop label |
| Missing values | None |

### Feature Importance (from trained model)

| Rank | Feature     | Importance |
|------|-------------|------------|
| 1    | Rainfall    | 22.7%      |
| 2    | Humidity    | 21.1%      |
| 3    | Potassium   | 18.1%      |
| 4    | Phosphorus  | 14.4%      |
| 5    | Nitrogen    | 10.9%      |
| 6    | Temperature | 7.6%       |
| 7    | pH          | 5.2%       |

## Future Improvements

- **Deep Learning Models** -- Experiment with neural networks (TensorFlow/PyTorch) and compare accuracy with RandomForest.
- **More Crops and Regions** -- Expand the dataset to include region-specific crop varieties and local soil data.
- **Weather API Integration** -- Auto-fetch real-time temperature, humidity, and rainfall from weather APIs instead of manual input.
- **Soil Sensor Integration** -- Connect IoT soil sensors (Arduino/Raspberry Pi) to feed live N, P, K, pH readings directly into the system.
- **User Accounts and History** -- Let farmers save past predictions and track soil health over time with a database backend.
- **Mobile App** -- Build a React Native or Flutter mobile app for field use where laptops are impractical.
- **Multilingual Support** -- Add regional language translations for farmers who don't use English.
- **Yield Prediction** -- Predict expected crop yield (kg/hectare) in addition to crop type, helping farmers estimate revenue.
- **Disease Detection** -- Add image-based plant disease detection using CNN models on leaf photos.
- **Deployment** -- Deploy the API to cloud platforms (AWS/GCP/Heroku) and host the frontend on GitHub Pages or Vercel.

## License

This project is open source and available for educational purposes.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.
