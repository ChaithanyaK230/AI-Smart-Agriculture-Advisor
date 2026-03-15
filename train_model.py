"""
Train a RandomForestClassifier for Crop Recommendation
=======================================================
This script loads the crop dataset, trains a machine learning model,
evaluates its performance, and saves it for later use by the API.

Every step is explained for beginners.
"""

# ─── Step 1: Import Libraries ───────────────────────────────────────────────────
# os           → work with file paths (to save the model in the right folder)
# joblib       → serialize (save) and deserialize (load) Python objects to disk
# pandas       → load and manipulate the CSV dataset as a table (DataFrame)
# train_test_split → split data into training set (model learns from) and
#                    test set (model is evaluated on, never seen during training)
# RandomForestClassifier → an ensemble model that builds many decision trees
#                          and combines their votes for a final prediction
# classification_report  → shows precision, recall, f1-score per crop
# accuracy_score         → percentage of correct predictions out of all predictions

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Our own helper functions from the utils/ folder
from utils.data_loader import load_crop_data, get_feature_columns, get_target_column


def train():
    # ─── Step 2: Load the Dataset ────────────────────────────────────────────────
    # load_crop_data() reads "data/Crop_recommendation.csv" into a DataFrame.
    # Each row = one soil sample, each column = a feature or the crop label.
    df = load_crop_data()

    print("=" * 60)
    print("STEP 1: LOADING DATA")
    print("=" * 60)
    # .shape returns (rows, columns) → tells us the size of our dataset
    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    # .nunique() counts how many different crop types exist in the label column
    print(f"Unique crops: {df[get_target_column()].nunique()}")
    # .value_counts() shows how many samples we have for each crop
    # A balanced dataset (equal counts) prevents the model from being biased
    print(f"\nSamples per crop:\n{df[get_target_column()].value_counts()}")

    # ─── Step 3: Separate Features (X) and Target (y) ───────────────────────────
    # Features (X) = the INPUT columns the model uses to make predictions
    #   → N, P, K, temperature, humidity, ph, rainfall
    # Target (y) = the OUTPUT column the model tries to predict
    #   → label (the crop name)
    #
    # Think of it like a math function: y = f(X)
    # The model learns the function f() from training data.

    print("\n" + "=" * 60)
    print("STEP 2: PREPARING FEATURES AND TARGET")
    print("=" * 60)

    features = get_feature_columns()  # ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    target = get_target_column()      # "label"

    X = df[features]  # DataFrame with only the 7 input columns
    y = df[target]     # Series with only the crop names

    print(f"Feature columns (X): {features}")
    print(f"Target column (y): {target}")
    print(f"X shape: {X.shape}  --  {X.shape[0]} samples, {X.shape[1]} features")
    print(f"y shape: {y.shape}  --  {y.shape[0]} labels")

    # ─── Step 4: Split into Training and Testing Sets ────────────────────────────
    # WHY split? If we train and test on the same data, the model might just
    # memorize answers instead of learning patterns. This is called "overfitting".
    #
    # test_size=0.2  → 20% of data is held back for testing (440 samples)
    #                  80% is used for training (1760 samples)
    # random_state=42 → a fixed seed so the split is the same every time you run
    #                   this script (makes results reproducible)

    print("\n" + "=" * 60)
    print("STEP 3: SPLITTING DATA (80% train / 20% test)")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set:  {X_test.shape[0]} samples")

    # ─── Step 5: Train the RandomForestClassifier ────────────────────────────────
    # How RandomForest works (simplified):
    #   1. It creates 100 decision trees (n_estimators=100)
    #   2. Each tree is trained on a random subset of the data
    #   3. Each tree independently predicts the crop
    #   4. The final prediction = the crop that most trees voted for (majority vote)
    #
    # Why RandomForest?
    #   - Works well with tabular data like ours
    #   - Handles both numeric features naturally
    #   - Resistant to overfitting (thanks to averaging many trees)
    #   - Gives feature importance for free
    #
    # .fit(X_train, y_train) = the actual training step
    #   The model looks at the training data and learns which feature values
    #   correspond to which crops.

    print("\n" + "=" * 60)
    print("STEP 4: TRAINING THE MODEL")
    print("=" * 60)
    print("Training RandomForestClassifier with 100 trees...")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Training complete!")

    # ─── Step 6: Evaluate the Model ──────────────────────────────────────────────
    # .predict(X_test) → feed the test features to the model and get predictions
    # We then compare these predictions to the actual answers (y_test)
    #
    # accuracy_score → (correct predictions) / (total predictions)
    #   Example: 437 correct out of 440 = 99.3% accuracy
    #
    # classification_report → detailed breakdown per crop:
    #   precision = of all samples predicted as crop X, how many were actually X?
    #   recall    = of all actual crop X samples, how many did we predict correctly?
    #   f1-score  = harmonic mean of precision and recall (overall quality per crop)
    #   support   = how many test samples existed for that crop

    print("\n" + "=" * 60)
    print("STEP 5: EVALUATING THE MODEL")
    print("=" * 60)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # ─── Step 7: Feature Importance ──────────────────────────────────────────────
    # RandomForest can tell us which features mattered most for predictions.
    # .feature_importances_ returns a score (0 to 1) for each feature.
    # Higher score = that feature had more influence on the model's decisions.
    #
    # This helps us understand:
    #   - Which soil/weather factors matter most for choosing a crop
    #   - Whether any features are redundant (very low importance)
    #
    # We sort them from most to least important for readability.

    print("=" * 60)
    print("STEP 6: FEATURE IMPORTANCE")
    print("=" * 60)
    print("(Which features matter most for predicting the right crop?)\n")

    importances = model.feature_importances_
    # Create a DataFrame pairing each feature name with its importance score
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    })
    # Sort by importance, highest first
    importance_df = importance_df.sort_values("Importance", ascending=False)
    importance_df = importance_df.reset_index(drop=True)

    # Display as a formatted table with rank numbers
    for rank, row in importance_df.iterrows():
        bar = "#" * int(row["Importance"] * 40)  # visual bar scaled to 40 chars
        print(f"  {rank + 1}. {row['Feature']:<14} {row['Importance']:.4f}  {bar}")

    # ─── Step 8: Save the Trained Model ──────────────────────────────────────────
    # joblib.dump() serializes the model object to a .pkl file on disk.
    # Later, the backend API loads this file with joblib.load() to make
    # predictions without re-training.
    #
    # Think of it like "saving your game" — you can pick up exactly where
    # you left off without replaying everything.

    print("\n" + "=" * 60)
    print("STEP 7: SAVING THE MODEL")
    print("=" * 60)

    model_path = os.path.join("models", "crop_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    print("You can now run the backend: python backend/app.py")


if __name__ == "__main__":
    train()
