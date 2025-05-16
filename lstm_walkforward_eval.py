#!/usr/bin/env python3
"""
Run Optuna hyperparameter optimization for LSTM using walk-forward strategy.
Saves best models and evaluation metrics (RMSE, MAE, R2, MAPE) at each split.
"""

import os
import json
import joblib
import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

# Local path configuration (no dependency on config.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "OUTPUT_DIR")
MODEL_RESULTS_DIR = os.path.join(BASE_DIR, "MODEL_RESULTS")

EXPECTED_ENERGY_COL = 'Power demand_sum'
RESULTS_DIR = os.path.join(MODEL_RESULTS_DIR, 'lstm_walkforward')
os.makedirs(RESULTS_DIR, exist_ok=True)

def create_lstm_model(input_shape, trial):
    model = Sequential()
    model.add(LSTM(
        units=trial.suggest_int("units", 32, 128, step=32),
        input_shape=input_shape,
        return_sequences=False
    ))
    model.add(Dropout(trial.suggest_float("dropout", 0.1, 0.5)))
    model.add(Dense(1))
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=trial.suggest_float("lr", 1e-4, 1e-2, log=True))
    )
    return model

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

def evaluate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

def walk_forward_lstm_optuna(df, initial_train_pct=0.8, val_pct=0.02, seq_len=24, n_trials=10):
    df = df.dropna()
    feature_cols = [col for col in df.columns if col != EXPECTED_ENERGY_COL]

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    df[feature_cols] = feature_scaler.fit_transform(df[feature_cols])
    df[[EXPECTED_ENERGY_COL]] = target_scaler.fit_transform(df[[EXPECTED_ENERGY_COL]])

    all_results = []
    split_id = 1
    total_len = len(df)

    train_end = int(initial_train_pct * total_len)
    val_len = int(val_pct * total_len)

    while train_end + val_len <= total_len:
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:train_end + val_len]

        X_train, y_train = create_sequences(train_df[feature_cols].values, train_df[EXPECTED_ENERGY_COL].values, seq_len)
        X_val, y_val = create_sequences(val_df[feature_cols].values, val_df[EXPECTED_ENERGY_COL].values, seq_len)

        def objective(trial):
            model = create_lstm_model((X_train.shape[1], X_train.shape[2]), trial)
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=30,
                batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                verbose=0
            )
            val_preds = model.predict(X_val)
            return np.sqrt(mean_squared_error(y_val, val_preds))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        best_trial = study.best_trial

        best_model = create_lstm_model((X_train.shape[1], X_train.shape[2]), best_trial)
        best_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=best_trial.params['batch_size'],
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0
        )

        y_pred = best_model.predict(X_val)
        y_val_inv = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
        y_pred_inv = target_scaler.inverse_transform(y_pred).flatten()

        metrics = evaluate_metrics(y_val_inv, y_pred_inv)
        print(f"\n➡️ Split {split_id} — Train: {len(train_df)}, Val: {len(val_df)}")
        print(f"RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R2: {metrics['r2']:.3f}, MAPE: {metrics['mape']:.2f}%")

        result = {
            "split_id": split_id,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "metrics": metrics,
            "best_params": best_trial.params
        }

        model_path = os.path.join(RESULTS_DIR, f"lstm_split_{split_id}.h5")
        best_model.save(model_path)

        with open(os.path.join(RESULTS_DIR, f"metrics_split_{split_id}.json"), 'w') as f:
            json.dump(result, f, indent=2)

        all_results.append(result)
        split_id += 1
        train_end += val_len

    summary_df = pd.DataFrame([r['metrics'] | {"split_id": r['split_id']} for r in all_results])
    summary_df.to_csv(os.path.join(RESULTS_DIR, "lstm_walkforward_summary.csv"), index=False)
    print("\n✅ Walk-forward LSTM evaluation complete. Summary saved.")


if __name__ == "__main__":
    data_path = os.path.join(OUTPUT_DIR, 'merged_data', 'gkg_energy.csv')
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    walk_forward_lstm_optuna(df)
