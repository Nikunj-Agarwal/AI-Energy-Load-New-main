#!/usr/bin/env python3
"""
Orchestrates walk-forward evaluation using Optuna and XGBoost with LSTM-style expanding actual training.
"""

import os
import json
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === Setup path to import from Model_pedictor/ ===
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from Model_pedictor.preparation import (
    load_data,
    handle_missing_values,
    handle_outliers
)
from run_xgb_optuna import run_optuna_on_data
from Model_pedictor.eval_utlis import compute_metrics

# === CONFIGURATION ===
TARGET_COL = "Power demand_sum"
INITIAL_TRAIN_PCT = 0.80
TEST_PCT = 0.02
N_TRIALS = 25

# === PATHS ===
OUTPUT_DIR = BASE_DIR / "OUTPUT_DIR"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RUNNING_EVAL_DIR = OUTPUT_DIR / "running_eval"
RUNNING_EVAL_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = RUNNING_EVAL_DIR / "walk_forward_per_split_summary_weather.csv"

PREDICTED_OUTPUT_PATH = RESULTS_DIR / "predicted_weather.csv"
PREDICTION_PLOT_PATH = RESULTS_DIR / "prediction_vs_actual_weather.png"

def main():
    print("=== Walk-Forward Evaluation with Optuna ===")

    data_path = BASE_DIR / "Weather_Energy" / "weather_energy_15min.csv"
    df = load_data(str(data_path))
    if df is None:
        print(f"❌ ERROR: Failed to load data from {data_path}")
        return
    df = handle_missing_values(df)
    df = handle_outliers(df, target_col=TARGET_COL)
    df = df.sort_index()

    predictions = []
    all_results = []
    split_id = 1

    total_len = len(df)
    train_end = int(INITIAL_TRAIN_PCT * total_len)
    test_len = int(TEST_PCT * total_len)

    while train_end + test_len <= total_len:
        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:train_end + test_len]

        if len(test_df) < 5 or len(train_df) < 100:
            print(f"⚠️  Skipping split {split_id} due to insufficient data")
            break

        print(f"\n➡️ Split {split_id} — Train: {len(train_df)}, Test: {len(test_df)}")

        X_train = train_df.drop(columns=[TARGET_COL])
        y_train = train_df[TARGET_COL]
        X_test = test_df.drop(columns=[TARGET_COL])
        y_test = test_df[TARGET_COL]

        model, best_params, _ = run_optuna_on_data(X_train, y_train, X_test, y_test, n_trials=N_TRIALS)

        X_test_clean = X_test.select_dtypes(include=[np.number])
        y_pred = model.predict(X_test_clean)

        metrics = compute_metrics(y_test, y_pred)

        result = {
            "split_id": split_id,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "mean_error": metrics["mean_error"],
            "mean_percent_error": metrics["mean_percent_error"],
            "best_params": best_params
        }

        all_results.append(result)

        temp = test_df[[TARGET_COL]].copy()
        temp["Predicted"] = y_pred
        predictions.append(temp)

        train_end += test_len
        split_id += 1

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\n✅ Walk-forward summary saved: {SUMMARY_CSV}")
    print(summary_df.describe(include='all'))

    combined_pred_df = pd.concat(predictions)
    combined_pred_df.to_csv(PREDICTED_OUTPUT_PATH)
    print(f"✅ Predictions saved: {PREDICTED_OUTPUT_PATH}")

    plt.figure(figsize=(14, 6))
    plt.plot(combined_pred_df.index, combined_pred_df[TARGET_COL], label="Actual", alpha=0.7)
    plt.plot(combined_pred_df.index, combined_pred_df["Predicted"], label="Predicted", alpha=0.7)
    plt.title("Actual vs Predicted Power Demand (Weather Only)")
    plt.xlabel("Timestamp")
    plt.ylabel("Power Demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PREDICTION_PLOT_PATH)
    print(f"✅ Prediction disparity plot saved: {PREDICTION_PLOT_PATH}")

if __name__ == "__main__":
    main()
