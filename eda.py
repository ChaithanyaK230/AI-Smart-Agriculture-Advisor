
"""
Exploratory Data Analysis (EDA) for Crop Recommendation Dataset
================================================================
This script helps you understand the dataset before building a model.
EDA answers: What does the data look like? Are there problems? What patterns exist?
"""

# ─── Step 1: Import Libraries ───────────────────────────────────────────────────
# pandas       → for loading and manipulating tabular data (like Excel in Python)
# matplotlib   → for creating charts and plots
# seaborn      → built on top of matplotlib, makes prettier statistical plots
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # use non-interactive backend so plots save without opening windows
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Step 2: Load the Dataset ───────────────────────────────────────────────────
# pd.read_csv() reads a CSV file and stores it as a DataFrame (a table).
# A DataFrame has rows (samples) and columns (features).
df = pd.read_csv("data/Crop_recommendation.csv")

# ─── Step 3: Basic Overview ─────────────────────────────────────────────────────
# .shape returns (number_of_rows, number_of_columns) as a tuple.
# Rows = how many data samples we have.
# Columns = how many features (inputs) + target (output) we have.
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

# .columns gives the name of every column in the dataset.
# These are the variables we'll use for prediction.
print(f"\nColumn Names: {list(df.columns)}")

# .dtypes shows the data type of each column.
# float64/int64 = numbers, object = text/categories.
print(f"\nData Types:\n{df.dtypes}")

# .head() shows the first 5 rows so you can see what the data looks like.
print(f"\nFirst 5 Rows:")
print(df.head())

# ─── Step 4: Check for Missing Values ───────────────────────────────────────────
# .isnull() marks each cell as True (missing) or False (has value).
# .sum() counts the True values per column → total missing values.
# Missing values can break models or give wrong results.
print("\n" + "=" * 60)
print("MISSING VALUES CHECK")
print("=" * 60)
missing = df.isnull().sum()
print(missing)

# Check if any column has missing values at all.
if missing.sum() == 0:
    print("\nNo missing values found! The dataset is clean.")
else:
    print(f"\nTotal missing values: {missing.sum()}")

# ─── Step 5: Statistical Summary ────────────────────────────────────────────────
# .describe() calculates statistics for each numeric column:
#   count = number of non-null values
#   mean  = average value
#   std   = standard deviation (how spread out values are)
#   min   = smallest value
#   25%   = value below which 25% of data falls (1st quartile)
#   50%   = median (middle value)
#   75%   = 3rd quartile
#   max   = largest value
print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)
print(df.describe())

# ─── Step 6: Target Variable Analysis ───────────────────────────────────────────
# The "label" column is our target — the crop we want to predict.
# .value_counts() counts how many samples exist for each crop.
# A balanced dataset (equal counts) is easier for models to learn from.
print("\n" + "=" * 60)
print("CROP DISTRIBUTION (Target Variable)")
print("=" * 60)
crop_counts = df["label"].value_counts()
print(crop_counts)
print(f"\nTotal unique crops: {df['label'].nunique()}")

# ─── Step 7: Correlation Heatmap ────────────────────────────────────────────────
# Correlation measures how two features move together (-1 to +1):
#   +1 = when one goes up, the other always goes up (strong positive)
#    0 = no relationship
#   -1 = when one goes up, the other always goes down (strong negative)
#
# .select_dtypes("number") picks only numeric columns (not the crop label).
# .corr() computes the correlation between every pair of numeric columns.
#
# Why it matters: highly correlated features carry redundant information.

print("\n" + "=" * 60)
print("CORRELATION BETWEEN FEATURES")
print("=" * 60)
numeric_df = df.select_dtypes("number")  # exclude the text "label" column
correlation = numeric_df.corr()
print(correlation)

# Plot the correlation as a color-coded heatmap.
# annot=True  → show the numbers inside each cell
# cmap        → color scheme (coolwarm = blue for negative, red for positive)
# fmt=".2f"   → round numbers to 2 decimal places
# linewidths  → thin white lines between cells for readability
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=16)
plt.tight_layout()  # prevent labels from being cut off
plt.savefig("data/correlation_heatmap.png", dpi=150)
print("\nSaved: data/correlation_heatmap.png")
plt.show()

# ─── Step 8: Feature Distributions ──────────────────────────────────────────────
# A histogram shows how values are spread for each feature.
# This helps spot:
#   - Skewed data (most values on one side)
#   - Outliers (extreme values far from the rest)
#   - The range of values each feature takes
#
# We create one subplot per feature in a grid layout.
# bins=20     → divide the value range into 20 bars
# edgecolor   → outline color for each bar

features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 8))
# flatten() converts the 2D grid of axes into a 1D list so we can loop easily.
axes = axes.flatten()

for i, feature in enumerate(features):
    axes[i].hist(df[feature], bins=20, color="#66bb6a", edgecolor="black")
    axes[i].set_title(feature, fontsize=14, fontweight="bold")
    axes[i].set_xlabel("Value")       # x-axis label
    axes[i].set_ylabel("Frequency")   # y-axis label (how often a value appears)

# Hide the 8th subplot (we only have 7 features but a 2x4 grid = 8 slots).
axes[7].set_visible(False)

plt.suptitle("Distribution of Each Feature", fontsize=18)
plt.tight_layout()
plt.savefig("data/feature_distributions.png", dpi=150)
print("Saved: data/feature_distributions.png")
plt.show()

# ─── Step 9: Box Plots by Crop ──────────────────────────────────────────────────
# A box plot shows the spread of a feature for each crop:
#   - The box = middle 50% of values (interquartile range)
#   - The line inside = median
#   - The whiskers = range of most data
#   - Dots outside = outliers
#
# This reveals which features distinguish certain crops.

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(22, 10))
axes = axes.flatten()

for i, feature in enumerate(features):
    sns.boxplot(x="label", y=feature, data=df, ax=axes[i], palette="Greens")
    axes[i].set_title(feature, fontsize=14, fontweight="bold")
    axes[i].set_xlabel("")
    # Rotate crop names 90 degrees so they don't overlap each other.
    axes[i].tick_params(axis="x", rotation=90)

axes[7].set_visible(False)

plt.suptitle("Feature Distribution by Crop", fontsize=18)
plt.tight_layout()
plt.savefig("data/boxplots_by_crop.png", dpi=150)
print("Saved: data/boxplots_by_crop.png")
plt.show()

print("\n" + "=" * 60)
print("EDA COMPLETE!")
print("=" * 60)
print("Charts saved in the data/ folder:")
print("  1. data/correlation_heatmap.png")
print("  2. data/feature_distributions.png")
print("  3. data/boxplots_by_crop.png")
