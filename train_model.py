"""Train a RandomForestClassifier for crop recommendation and save the model."""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from utils.data_loader import load_crop_data, get_feature_columns, get_target_column


def train():
    # Load data
    df = load_crop_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Crops in dataset: {df[get_target_column()].nunique()}")
    print(f"Samples per crop:\n{df[get_target_column()].value_counts()}\n")

    # Split features and target
    X = df[get_feature_columns()]
    y = df[get_target_column()]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    model_path = os.path.join("models", "crop_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train()
