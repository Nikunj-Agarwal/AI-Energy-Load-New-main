#!/usr/bin/env python3
"""
Run Optuna hyperparameter optimization for XGBoost across 9 data splits.
Saves best models and a summary of metrics.
"""
import os
import json
import joblib
import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from config import PATHS

EXPECTED_ENERGY_COL = 'Power demand_sum'

def load_split(data_type, strategy):
    base = PATHS['SPLIT_DATA_DIR']
    folder = os.path.join(base, data_type, strategy)
    train_path = os.path.join(folder, 'train.csv')
    test_path = os.path.join(folder, 'test.csv')
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Split files not found for {data_type}/{strategy}")
    train = pd.read_csv(train_path, index_col=0, parse_dates=True)
    test = pd.read_csv(test_path, index_col=0, parse_dates=True)
    print(f"Loaded {train_path}: columns={train.columns.tolist()}, shape={train.shape}")
    train = train.drop(columns=["Unnamed: 0"], errors="ignore")
    test = test.drop(columns=["Unnamed: 0"], errors="ignore")
    mask_train = train[EXPECTED_ENERGY_COL].notna() & np.isfinite(train[EXPECTED_ENERGY_COL])
    if not mask_train.all():
        print(f"Warning: Dropping {(~mask_train).sum()} rows with NaN/inf in train energy for {data_type}/{strategy}")
    train = train[mask_train]
    mask_test = test[EXPECTED_ENERGY_COL].notna() & np.isfinite(test[EXPECTED_ENERGY_COL])
    if not mask_test.all():
        print(f"Warning: Dropping {(~mask_test).sum()} rows with NaN/inf in test energy for {data_type}/{strategy}")
    test = test[mask_test]
    X_train = train.drop(columns=[EXPECTED_ENERGY_COL])
    y_train = train[EXPECTED_ENERGY_COL]
    X_test = test.drop(columns=[EXPECTED_ENERGY_COL])
    y_test = test[EXPECTED_ENERGY_COL]
    if len(X_train) == 0 or len(y_train) == 0:
        print(f"Error: No training samples left after cleaning for {data_type}/{strategy}. Skipping this split.")
        return None, None, None, None
    if len(X_test) == 0 or len(y_test) == 0:
        print(f"Warning: No test samples left after cleaning for {data_type}/{strategy}.")
    return X_train, y_train, X_test, y_test

def run_optuna_on_data(X_train, y_train, X_val, y_val, n_trials=25):
    def objective(trial):
        X_train_clean = X_train.select_dtypes(include=[np.number])
        X_val_clean = X_val.select_dtypes(include=[np.number])
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0)
        }
        model = XGBRegressor(**params, random_state=42, verbosity=0)
        model.fit(X_train_clean, y_train)
        preds = model.predict(X_val_clean)
        return np.sqrt(mean_squared_error(y_val, preds))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    X_train_clean = X_train.select_dtypes(include=[np.number])
    model = XGBRegressor(**best_params, random_state=42, verbosity=0)
    model.fit(X_train_clean, y_train)
    return model, best_params, study

def run_optuna_for_split(data_type, strategy, n_trials=25):
    print(f"Optimizing XGBoost for {data_type}/{strategy}")
    X_train, y_train, X_test, y_test = load_split(data_type, strategy)
    if X_train is None or y_train is None:
        print(f"Skipping {data_type}/{strategy} due to empty training set.")
        return None
    X_train = X_train.select_dtypes(include=[np.number])
    X_test_clean = X_test.select_dtypes(include=[np.number])
    model, best_params, study = run_optuna_on_data(X_train, y_train, X_test_clean, y_test, n_trials=n_trials)
    preds = model.predict(X_test_clean)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    out_dir = os.path.join(PATHS['MODELS_DIR'], 'xgb_optuna', data_type, strategy)
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, 'model.joblib'))
    study.trials_dataframe().to_csv(os.path.join(out_dir, 'optuna_trials.csv'), index=False)
    result = {
        'data_type': data_type,
        'strategy': strategy,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'best_params': best_params
    }
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(result, f, indent=2)
    return result

def main():
    data_types = ['weather', 'news', 'combined']
    strategies = ['fully_random', 'month_based', 'seasonal_block']
    all_results = []
    for dt in data_types:
        for strat in strategies:
            res = run_optuna_for_split(dt, strat)
            if res is not None:
                all_results.append(res)
    summary = pd.DataFrame(all_results)
    out_path = os.path.join(PATHS['MODEL_RESULTS_DIR'], 'xgb_optuna_summary.csv')
    os.makedirs(PATHS['MODEL_RESULTS_DIR'], exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"Summary saved to {out_path}")

if __name__ == '__main__':
    main()
