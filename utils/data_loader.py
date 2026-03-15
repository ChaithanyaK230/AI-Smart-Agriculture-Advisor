import pandas as pd
import os


def load_crop_data(filepath=None):
    """Load the crop recommendation dataset."""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "..", "data", "Crop_recommendation.csv")

    df = pd.read_csv(filepath)
    return df


def get_feature_columns():
    """Return the feature column names used for prediction."""
    return ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def get_target_column():
    """Return the target column name."""
    return "label"
