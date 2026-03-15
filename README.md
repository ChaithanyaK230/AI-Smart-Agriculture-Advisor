# AI Smart Agriculture Advisor

A machine learning project that recommends the best crop and fertilizer based on soil and environmental conditions.

## Features

- **Crop Prediction** — Recommends the best crop using a RandomForestClassifier trained on soil (N, P, K, pH) and weather (temperature, humidity, rainfall) data
- **Fertilizer Recommendation** — Suggests fertilizers based on soil nutrient levels (N, P, K)
- **Web Interface** — Simple browser-based UI to interact with the system
- **REST API** — Flask backend with endpoints for predictions

## Project Structure

```
├── data/               # Dataset and EDA output charts
├── models/             # Trained model files (.pkl)
├── backend/            # Flask API server
├── frontend/           # HTML/CSS/JS web interface
├── utils/              # Shared helper modules
├── train_model.py      # Model training script
├── eda.py              # Exploratory data analysis script
└── requirements.txt    # Python dependencies
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train_model.py

# Start the backend server
python backend/app.py
```

Then open `frontend/index.html` in your browser.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict-crop` | POST | Predict best crop from soil & weather data |
| `/recommend-fertilizer` | POST | Get fertilizer advice from N, P, K values |

## Dataset

Uses the [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) with 2200 samples across 22 crops.

## Tech Stack

- Python, scikit-learn, pandas, NumPy
- Flask (backend)
- HTML/CSS/JS (frontend)
