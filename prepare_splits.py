#!/usr/bin/env python3
"""
Generate train/test splits for three data types (weather, news, combined)
and three split strategies (fully_random, month_based, seasonal_block),
producing 9 datasets under DATA/split_data/{data_type}/{strategy}/
"""
import os
import glob
import json
import pandas as pd
import numpy as np # Import numpy
from config import PATHS
from Model_pedictor.preparation import split_time_series_data

# Define the expected energy column name
EXPECTED_ENERGY_COL = 'Power demand_sum'

def find_energy_column(columns):
    """Find the energy column, prioritizing the expected name."""
    if EXPECTED_ENERGY_COL in columns:
        return EXPECTED_ENERGY_COL
    # Fallback to pattern matching if exact name not found
    candidates = [c for c in columns if any(x in c.lower() for x in ['power', 'load', 'demand'])]
    if candidates:
        print(f"[WARN] Expected energy column '{EXPECTED_ENERGY_COL}' not found. Using '{candidates[0]}'.")
        return candidates[0]
    return None

def load_weather_energy():
    # Find 15-minute aggregated weather+energy file
    wd = os.path.join(os.path.dirname(__file__), 'Weather_Energy')
    files = glob.glob(os.path.join(wd, '*15min.csv'))
    if not files:
        raise FileNotFoundError('No 15-min weather CSV found in Weather_Energy/')
    # Determine which datetime column exists
    with open(files[0], 'r') as f:
        header = f.readline().strip().split(',')
    if 'timestamp' in header:
        date_col = 'timestamp'
    elif 'datetime' in header:
        date_col = 'datetime'
    else:
        raise ValueError('Weather file missing timestamp/datetime column')
    # Read and parse only that column
    df = pd.read_csv(files[0], parse_dates=[date_col])
    # Drop any extraneous index columns (e.g., 'Unnamed: 0' or similar)
    df = df.loc[:, ~df.columns.str.contains(r'^Unnamed')]
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    df.set_index(date_col, inplace=True)
    # ensure index is a true DatetimeIndex
    df.index = pd.to_datetime(df.index, errors='raise')

    # Verify energy column exists
    energy_col = find_energy_column(df.columns)
    if not energy_col:
        raise ValueError(f"Could not find energy column (expected '{EXPECTED_ENERGY_COL}' or similar) in weather data.")
    print(f"[DIAG] Identified energy column in weather data: {energy_col}")

    print(f"[DIAG] Weather DF columns: {df.columns.tolist()}")
    print(f"[DIAG] Weather DF sample:\n{df.head()}")
    return df

def load_gkg_aggregated():
    # Load aggregated GKG features
    agg_file = os.path.join(PATHS['AGGREGATED_DIR'], 'aggregated_gkg_15min.csv')
    if not os.path.exists(agg_file):
        raise FileNotFoundError(f'Aggregated GKG file not found: {agg_file}')
    df = pd.read_csv(agg_file)
    # Drop any unnamed index column
    df = df.loc[:, ~df.columns.str.contains(r'^Unnamed')]
    # Set index on timestamp or time_bucket
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    elif 'time_bucket' in df.columns:
        df['time_bucket'] = pd.to_datetime(df['time_bucket'])
        df.set_index('time_bucket', inplace=True)
    else:
        raise ValueError('GKG aggregated file missing timestamp/time_bucket column')
    print(f"[DIAG] GKG DF columns: {df.columns.tolist()}")
    print(f"[DIAG] GKG DF sample:\n{df.head()}")
    return df

def merge_combined(weather_df, gkg_df):
    """Inner join on datetime index with more robust handling"""
    # Reset indices to columns for explicit merge
    w_df = weather_df.reset_index()
    g_df = gkg_df.reset_index()
    
    # Get the column names containing datetime values
    w_time_col = 'timestamp' if 'timestamp' in w_df.columns else 'datetime'
    g_time_col = 'time_bucket' if 'time_bucket' in g_df.columns else 'timestamp'

    # Log the time ranges to help with debugging
    print(f"Weather data time range: {w_df[w_time_col].min()} to {w_df[w_time_col].max()}")
    print(f"GKG data time range: {g_df[g_time_col].min()} to {g_df[g_time_col].max()}")
    
    # Merge on datetime columns
    combined = pd.merge(w_df, g_df, left_on=w_time_col, right_on=g_time_col, how='inner')
    
    # Set the merged datetime as index
    if not combined.empty:
        combined.set_index(w_time_col, inplace=True)
        # Drop the redundant datetime column
        if g_time_col in combined.columns and g_time_col != w_time_col:
             combined.drop(columns=[g_time_col], inplace=True)
    print(f"[DIAG] Combined DF columns after merge: {combined.columns.tolist()}")
    # Check for energy column
    energy_col = find_energy_column(combined.columns)
    print(f"[DIAG] Energy column in combined: {energy_col}")
    if not energy_col:
        print("[ERROR] No energy column found in combined dataframe after merge!")
    else:
        print(f"[DIAG] Sample energy column values:\n{combined[energy_col].head()}")
    return combined

def merge_news_energy(gkg_df, weather_df):
    """Merge news (gkg_df) with only energy demand from weather_df"""
    # Reset indices to columns for explicit merge
    w_df = weather_df.reset_index()
    g_df = gkg_df.reset_index()
    
    # Get the column names containing datetime values
    w_time_col = 'timestamp' if 'timestamp' in w_df.columns else 'datetime'
    g_time_col = 'time_bucket' if 'time_bucket' in g_df.columns else 'timestamp'
    
    # Identify energy column in weather data
    energy_col = find_energy_column(w_df.columns)
    print(f"[DIAG] Energy column in weather_df for news merge: {energy_col}")
    if not energy_col:
        raise ValueError(f"No energy demand column (expected '{EXPECTED_ENERGY_COL}' or similar) found in weather data")
    
    # Select only the datetime and energy columns from weather data
    w_df_subset = w_df[[w_time_col, energy_col]]
    
    # Merge on datetime columns
    merged = pd.merge(g_df, w_df_subset, left_on=g_time_col, right_on=w_time_col, how='inner')
    
    # Set the datetime as index
    if not merged.empty:
        merged.set_index(g_time_col, inplace=True)
        # Drop the redundant datetime column
        if w_time_col != g_time_col and w_time_col in merged.columns:
            merged.drop(columns=[w_time_col], inplace=True)
    print(f"[DIAG] News+Energy merged columns: {merged.columns.tolist()}")
    if energy_col in merged.columns:
        print(f"[DIAG] Sample energy column values:\n{merged[energy_col].head()}")
    else:
        print("[ERROR] Energy column missing after news-energy merge!")
    return merged

def make_splits(df, data_type):
    print(f"[DIAG] Columns before splitting for {data_type}: {df.columns.tolist()}")
    print(f"[DIAG] Sample data before splitting:\n{df.head()}")
    # Determine energy target column
    target_col = find_energy_column(df.columns)
    print(f"[DIAG] Target column identified for {data_type}: {target_col}")
    if not target_col:
        raise ValueError(f"No target column (expected '{EXPECTED_ENERGY_COL}' or similar) in {data_type} data")

    # Prepare output base
    base_dir = PATHS['SPLIT_DATA_DIR']
    strategies = ['fully_random', 'month_based', 'seasonal_block']
    
    # Make a copy and reset index to avoid potential issues
    df_reset = df.copy()
    
    # Check if we have a datetime index to preserve
    has_datetime_index = isinstance(df.index, pd.DatetimeIndex)
    if has_datetime_index:
        # Save original index as a column
        df_reset['original_datetime'] = df_reset.index
        # Reset index to avoid issues with split_time_series_data
        df_reset = df_reset.reset_index(drop=True)
        datetime_col_name = 'original_datetime'
    else:
        datetime_col_name = None # No datetime index to preserve
    
    for strat in strategies:
        print(f'Creating split for {data_type} - {strat}')
        split_kwargs = dict(
            target_col=target_col, test_fraction=0.2, strategy=strat
        )
        # Pass datetime_col if it was preserved
        if datetime_col_name:
            split_kwargs['datetime_col'] = datetime_col_name

        X_train, y_train, X_test, y_test, split_info = split_time_series_data(
            df_reset, **split_kwargs
        )
        
        # Restore datetime index if it was present and preserved
        if datetime_col_name in X_train.columns:
            X_train = X_train.set_index(datetime_col_name)
            X_train.index.name = df.index.name # Restore original index name
        if datetime_col_name in X_test.columns:
            X_test = X_test.set_index(datetime_col_name)
            X_test.index.name = df.index.name # Restore original index name
        
        # Align indices before saving
        train_df = X_train.copy()
        # Ensure y_train/y_test have the correct index before assignment
        y_train.index = X_train.index
        y_test.index = X_test.index
        train_df[target_col] = y_train # Use the identified target_col name
        test_df = X_test.copy()
        test_df[target_col] = y_test # Use the identified target_col name

        out_dir = os.path.join(base_dir, data_type, strat)
        os.makedirs(out_dir, exist_ok=True)
        train_df.to_csv(os.path.join(out_dir, 'train.csv'), index=True)
        test_df.to_csv(os.path.join(out_dir, 'test.csv'), index=True)
        # Save split info
        with open(os.path.join(out_dir, 'split_info.json'), 'w') as f:
            # Convert numpy types to standard types for JSON serialization
            serializable_info = {}
            for k, v in split_info.items():
                if isinstance(v, dict):
                    serializable_info[k] = {str(sk): sv.tolist() if isinstance(sv, np.ndarray) else sv for sk, sv in v.items()}
                elif isinstance(v, np.ndarray):
                     serializable_info[k] = v.tolist()
                elif isinstance(v, (np.int64, np.int32, np.float64, np.float32)):
                     serializable_info[k] = v.item()
                else:
                     serializable_info[k] = v
            json.dump(serializable_info, f, default=str, indent=2)

def main():
    # Load datasets
    weather_df = load_weather_energy()
    gkg_df = load_gkg_aggregated()

    # --- DIAGNOSTIC PRINTS ---
    print("\n--- DataFrame Diagnostics ---")
    print(f"Weather DF Info: Index Type={type(weather_df.index)}, Length={len(weather_df)}")
    if isinstance(weather_df.index, pd.DatetimeIndex):
        print(f"Weather DF Time Range: {weather_df.index.min()} to {weather_df.index.max()}")
    print(f"\nGKG DF Info: Index Type={type(gkg_df.index)}, Length={len(gkg_df)}")
    if isinstance(gkg_df.index, pd.DatetimeIndex):
        print(f"GKG DF Time Range: {gkg_df.index.min()} to {gkg_df.index.max()}")

    # Check for index overlap
    if isinstance(weather_df.index, pd.DatetimeIndex) and isinstance(gkg_df.index, pd.DatetimeIndex):
        common_index = weather_df.index.intersection(gkg_df.index)
        print(f"\nNumber of common timestamps in index: {len(common_index)}")
        if len(common_index) == 0:
            print("WARNING: No common timestamps found between weather and GKG dataframes. Merges will result in empty dataframes!")
        else:
            print(f"Example common timestamps: {common_index[:5].tolist()}")
    else:
        print("\nWARNING: One or both dataframes do not have a DatetimeIndex. Cannot check for overlap.")
    print("--- End Diagnostics ---\n")
    # --- END DIAGNOSTIC PRINTS ---

    combined_df = merge_combined(weather_df, gkg_df)
    news_energy_df = merge_news_energy(gkg_df, weather_df)

    # No need to rename columns here, functions above should handle it.
    # energy_col = find_energy_column(weather_df.columns) # Already found in load_weather_energy
    # if energy_col and energy_col != 'target':
    #     print(f"Target column identified as '{energy_col}'. No renaming needed.")
        # weather_df.rename(columns={energy_col: "target"}, inplace=True)
        # news_energy_df.rename(columns={energy_col: "target"}, inplace=True)
        # combined_df.rename(columns={energy_col: "target"}, inplace=True)

    # --- DIAGNOSTIC PRINTS AFTER MERGE ---
    print(f"\n--- Post-Merge Diagnostics ---")
    print(f"Combined DF Length: {len(combined_df)}")
    print(f"News+Energy DF Length: {len(news_energy_df)}")
    print("--- End Post-Merge Diagnostics ---\n")
    # --- END DIAGNOSTIC PRINTS ---

    # Prepare splits for each data type
    datasets = {
        'weather': weather_df,
        'news': news_energy_df,
        'combined': combined_df
    }
    for name, df in datasets.items():
        # Add a check for empty dataframe before splitting
        if df.empty:
            print(f"SKIPPING split for '{name}': Input DataFrame is empty.")
            # Optionally create empty files/dirs if needed by downstream processes
            base_dir = PATHS['SPLIT_DATA_DIR']
            strategies = ['fully_random', 'month_based', 'seasonal_block']
            for strat in strategies:
                 out_dir = os.path.join(base_dir, name, strat)
                 os.makedirs(out_dir, exist_ok=True)
                 # Create empty files
                 pd.DataFrame().to_csv(os.path.join(out_dir, 'train.csv'))
                 pd.DataFrame().to_csv(os.path.join(out_dir, 'test.csv'))
                 with open(os.path.join(out_dir, 'split_info.json'), 'w') as f:
                     json.dump({'strategy': strat, 'status': 'skipped_empty_input'}, f)
            continue # Skip to the next dataset type

        make_splits(df, name)
    print('All splits generated successfully.')

if __name__ == '__main__':
    main()