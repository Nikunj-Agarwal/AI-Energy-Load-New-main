# ---------- 0. Imports ----------
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Create output directory
output_dir = Path("outlier_results")
output_dir.mkdir(parents=True, exist_ok=True)

# ---------- 1. Load data ----------
BASE = Path("OUTPUT_DIR/merged_data")
df = pd.read_csv(BASE / "gkg_energy.csv", index_col=0, parse_dates=True)
df.index.name = "timestamp"         # give the unnamed index a name

# load the two prediction files (assumes they also carry the same timestamp index)
pred_gkg = pd.read_csv("results/predicted_gkg_energy.csv",
                       index_col=0, parse_dates=True).rename(columns={"Predicted": "Predicted_gkg"})
pred_gkg = pred_gkg[["Predicted_gkg"]] # Select only the prediction column

pred_weather = pd.read_csv("results/predicted_weather.csv",
                           index_col=0, parse_dates=True).rename(columns={"Predicted": "Predicted_weather"})
pred_weather = pred_weather[["Predicted_weather"]] # Select only the prediction column

# sanity check: make sure indices align; if they don’t, inner-merge below will drop mismatches
# print(df.index.difference(pred_gkg.index))  # uncomment to debug mis-alignments

# ---------- 2. Detect outliers on Power demand_sum (IQR method) ----------
q1, q3 = df["Power demand_sum"].quantile([0.25, 0.75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outlier_mask = (df["Power demand_sum"] < lo) | (df["Power demand_sum"] > hi)

# keep only outlier timestamps
outlier_df = df[outlier_mask][["Power demand_sum"]]

# ---------- 3. Merge predictions ----------
combo = (
    outlier_df          # actuals on outlier points
    .join(pred_gkg, how="left")
    .join(pred_weather, how="left")
    .dropna(subset=["Predicted_gkg", "Predicted_weather"])  # drop rows where either prediction is missing
)

# ---------- 4. Compute metrics ----------
def metrics(y_true, y_pred):
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
    }

actual = combo["Power demand_sum"].values
m_gkg = metrics(actual, combo["Predicted_gkg"].values)
m_weather = metrics(actual, combo["Predicted_weather"].values)

metrics_df = (pd.DataFrame([m_gkg, m_weather],
                           index=["News + Weather model", "Weather-only model"])
                .round(3))
print("\nOutlier-only error metrics:")
print(metrics_df)
# Save metrics
metrics_df.to_csv(output_dir / "outlier_metrics.csv")

# ---------- 5. Quick visualization ----------
# Plot 1: Time series of Actual vs. Predictions (Scatter Plot)
plt.figure(figsize=(12, 6))
plt.scatter(combo.index, combo["Power demand_sum"], label="Actual", s=50, marker='o') # s is marker size
plt.scatter(combo.index, combo["Predicted_gkg"], label="Predicted_GKG", alpha=.7, s=50, marker='x')
plt.scatter(combo.index, combo["Predicted_weather"], label="Predicted_Weather", alpha=.7, s=50, marker='^')
plt.title("Model behaviour on demand outliers (Time Series Scatter)")
plt.ylabel("Power demand (same units as original)")
plt.xlabel("Timestamp")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / "outliers_timeseries_comparison.png")
plt.close() # Close the figure to free memory

# Removed other plot generation (scatter plots, residual plots)
# Removed all plt.show() calls

# ---------- 6. Optional: save outlier timestamps for later ----------
combo.reset_index().to_csv(output_dir / "outliers_with_predictions.csv", index=False)
print(f"\nAll outputs saved to '{output_dir.resolve()}' directory.")
