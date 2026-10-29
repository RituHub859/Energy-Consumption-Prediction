
# ============================================================
# PROJECT: SMART ENERGY CONSUMPTION PREDICTION AND OPTIMIZATION
# IDE: Visual Studio Code
# Dataset: energydata_complete.csv
# Models: Linear Regression, Random Forest, Gradient Boosting
# ============================================================

import os
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. PROJECT SETTINGS
# ------------------------------------------------------------

FILE_NAME = "energydata_complete.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

# ------------------------------------------------------------
# 2. LOAD THE DATASET
# ------------------------------------------------------------

print("=" * 60)
print("SMART ENERGY CONSUMPTION PREDICTION SYSTEM")
print("=" * 60)

if not os.path.exists(FILE_NAME):
    raise FileNotFoundError(
        f"\nDataset '{FILE_NAME}' was not found.\n"
        "Place energydata_complete.csv in the same folder "
        "as main.py."
    )

df = pd.read_csv(FILE_NAME)

# Remove accidental spaces from column names
df.columns = df.columns.str.strip()

print("\nDataset loaded successfully!")
print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# Check the required columns
required_columns = ["date", "Appliances", "lights"]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

# ------------------------------------------------------------
# 3. CLEAN AND PREPARE THE DATA
# ------------------------------------------------------------

print("\nPreparing hourly energy data...")

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df.dropna(subset=["date"])

df = df.sort_values("date")
df = df.set_index("date")

# Support both common outdoor temperature column names
temperature_candidates = ["To", "T_out"]

temperature_col = next(
    (col for col in temperature_candidates if col in df.columns),
    None
)

if temperature_col is None:
    raise ValueError(
        "Outdoor temperature column not found. "
        "Expected 'To' or 'T_out'. Check the dataset columns."
    )

# Ensure required measurement columns are numeric
for col in ["Appliances", "lights", temperature_col]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=["Appliances", "lights", temperature_col]
)

# Aggregate 10-minute readings into hourly observations
hourly = df.resample("h").agg({
    "Appliances": "sum",
    "lights": "sum",
    temperature_col: "mean"
})

hourly = hourly.rename(
    columns={temperature_col: "temperature"}
)

# Combined measured appliance and lighting energy in kWh
hourly["energy_kwh"] = (
    hourly["Appliances"] + hourly["lights"]
) / 1000.0

hourly = hourly.dropna()

print("\nHourly data prepared successfully!")
print(hourly.head())
print("\nNumber of hourly records:", len(hourly))

# Save the prepared dataset
hourly.to_csv(
    os.path.join(OUTPUT_DIR, "hourly_energy_data.csv")
)

# ------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ------------------------------------------------------------

print("\nCreating machine learning features...")

hourly["hour"] = hourly.index.hour
hourly["day_of_week"] = hourly.index.dayofweek

hourly["is_weekend"] = (
    hourly["day_of_week"] >= 5
).astype(int)

# Previous hour's energy consumption
hourly["lag_1"] = hourly["energy_kwh"].shift(1)

# Energy consumption 24 hours earlier
hourly["lag_24"] = hourly["energy_kwh"].shift(24)

# Remove rows without sufficient historical observations
hourly = hourly.dropna()

features = [
    "temperature",
    "hour",
    "day_of_week",
    "is_weekend",
    "lag_1",
    "lag_24"
]

target = "energy_kwh"

X = hourly[features]
y = hourly[target]

print("\nFeatures used:")
print(features)

print("\nNumber of usable records:", len(X))

if len(X) < 50:
    raise ValueError(
        "Not enough usable hourly records to train the models."
    )

# ------------------------------------------------------------
# 5. CHRONOLOGICAL TRAIN-TEST SPLIT
# ------------------------------------------------------------

print("\nSplitting data chronologically...")

split_index = int(len(X) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("First test timestamp:", X_test.index.min())
print("Last test timestamp:", X_test.index.max())

# ------------------------------------------------------------
# 6. CREATE MACHINE LEARNING MODELS
# ------------------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=150,
        max_leaf_nodes=20,
        random_state=42
    )
}

results = []
predictions = {}

# ------------------------------------------------------------
# 7. TRAIN AND EVALUATE THE MODELS
# ------------------------------------------------------------

print("\nTraining machine learning models...")

for name, model in models.items():

    print(f"\nTraining {name}...")

    # Train the model
    model.fit(X_train, y_train)

    # Predict the test period
    y_pred = model.predict(X_test)

    predictions[name] = y_pred

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE (kWh)": mae,
        "RMSE (kWh)": rmse,
        "R2 Score": r2
    })

# Sort models by lowest MAE
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAE (kWh)",
    ascending=True
).reset_index(drop=True)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 60)

print(results_df.round(4).to_string(index=False))

# Save evaluation results
results_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison.csv"),
    index=False
)

# ------------------------------------------------------------
# 8. SELECT THE BEST MODEL
# ------------------------------------------------------------

best_name = results_df.iloc[0]["Model"]
best_model = models[best_name]
best_predictions = predictions[best_name]

print("\nBest model based on test MAE:", best_name)

# Save actual and predicted values
comparison_df = pd.DataFrame({
    "Actual_kWh": y_test,
    "Predicted_kWh": best_predictions
}, index=y_test.index)

comparison_df.index.name = "date"

comparison_df.to_csv(
    os.path.join(OUTPUT_DIR, "actual_vs_predicted.csv")
)

# ------------------------------------------------------------
# 9. GRAPH: MODEL PERFORMANCE
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results_df,
    x="MAE (kWh)",
    y="Model",
    color="steelblue"
)

plt.title("Machine Learning Model Comparison")
plt.xlabel("Mean Absolute Error (kWh) - Lower is Better")
plt.ylabel("Model")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "model_comparison.png"),
    dpi=300
)

plt.show()

# ------------------------------------------------------------
# 10. GRAPH: ACTUAL VS PREDICTED ENERGY
# ------------------------------------------------------------

n = min(168, len(y_test))

plt.figure(figsize=(14, 6))

plt.plot(
    y_test.index[:n],
    y_test.iloc[:n].values,
    label="Actual Consumption",
    linewidth=2
)

plt.plot(
    y_test.index[:n],
    best_predictions[:n],
    label="Predicted Consumption",
    linewidth=2
)

plt.title(
    f"Actual vs Predicted Hourly Energy ({best_name})"
)
plt.xlabel("Date and Time")
plt.ylabel("Energy Consumption (kWh)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"),
    dpi=300
)

plt.show()

# ------------------------------------------------------------
# 11. FEATURE IMPORTANCE: RANDOM FOREST
# ------------------------------------------------------------

rf_model = models["Random Forest"]

importance = pd.Series(
    rf_model.feature_importances_,
    index=features
).sort_values(ascending=True)

plt.figure(figsize=(9, 6))

importance.plot(
    kind="barh",
    color="teal"
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "feature_importance.png"),
    dpi=300
)

plt.show()

print("\nFeature importance, highest first:")
print(importance.sort_values(ascending=False).round(4))

importance_df = importance.rename(
    "Importance"
).reset_index()

importance_df.columns = ["Feature", "Importance"]

importance_df.to_csv(
    os.path.join(OUTPUT_DIR, "feature_importance.csv"),
    index=False
)

# ------------------------------------------------------------
# 12. PREDICT ENERGY FOR THE NEXT HOUR
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("NEXT-HOUR ENERGY FORECAST")
print("=" * 60)

last_time = hourly.index[-1]

next_time = last_time + pd.Timedelta(hours=1)

last_row = hourly.iloc[-1]

# The previous-day value is taken from 24 hourly rows earlier.
# This assumes the hourly series has no missing hours.
if len(hourly) < 25:
    raise ValueError(
        "At least 25 hourly records are required for forecasting."
    )

previous_day_usage = hourly["energy_kwh"].iloc[-24]

# For this beginner version, assume temperature remains unchanged.
# A real deployment should use a weather forecast.
next_hour_features = pd.DataFrame([{
    "temperature": last_row["temperature"],
    "hour": next_time.hour,
    "day_of_week": next_time.dayofweek,
    "is_weekend": int(next_time.dayofweek >= 5),
    "lag_1": last_row["energy_kwh"],
    "lag_24": previous_day_usage
}], columns=features)

next_prediction = best_model.predict(
    next_hour_features
)[0]

# Energy cannot be negative
next_prediction = max(0.0, float(next_prediction))

print("Best model:", best_name)
print("Forecast time:", next_time)
print(
    f"Predicted energy consumption: "
    f"{next_prediction:.3f} kWh"
)

# Save the forecast
forecast_df = pd.DataFrame([{
    "Forecast_Time": next_time,
    "Predicted_Energy_kWh": next_prediction,
    "Model": best_name
}])

forecast_df.to_csv(
    os.path.join(OUTPUT_DIR, "next_hour_forecast.csv"),
    index=False
)

# ------------------------------------------------------------
# 13. HIGH-CONSUMPTION ALERT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("ENERGY-SAVING ALERT")
print("=" * 60)

# Flag a prediction above the 90th percentile
# of the training target values.
high_usage_threshold = y_train.quantile(0.90)

print(
    f"High-consumption threshold: "
    f"{high_usage_threshold:.3f} kWh"
)

if next_prediction > high_usage_threshold:

    print("\nENERGY ALERT: Higher-than-usual usage predicted!")

    print("Suggested actions:")
    print("- Check unnecessary appliances.")
    print("- Review heating or air-conditioning settings.")
    print("- Check equipment schedules and occupancy.")

else:

    print("\nNo high-consumption alert for this forecast.")

    print("Continue monitoring electricity usage.")

print(
    "\nNote: High usage does not automatically mean "
    "electricity is being wasted."
)

# ------------------------------------------------------------
# 14. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PROJECT COMPLETED")
print("=" * 60)

print("Dataset records used:", len(hourly))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Best model:", best_name)

best_result = results_df.iloc[0]

print(f"MAE: {best_result['MAE (kWh)']:.4f} kWh")
print(f"RMSE: {best_result['RMSE (kWh)']:.4f} kWh")
print(f"R2 Score: {best_result['R2 Score']:.4f}")

print(f"\nAll output files are saved in '{OUTPUT_DIR}/'.")