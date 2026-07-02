"""
Feature engineering module.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def add_temporal_features(df):
    """Extracts temporal features from timestamp."""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df.timestamp.dt.hour
    df['month'] = df.timestamp.dt.month
    df['weekday'] = df.timestamp.dt.weekday
    df['dayofyear'] = df.timestamp.dt.dayofyear
    return df

def transform_target(df):
    """Applies log1p transformation to meter_reading."""
    df['meter_reading'] = np.log1p(df['meter_reading'])
    return df

def handle_missing(df):
    """Fills NaN values with column medians for numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df

def encode_categoricals(df, all_possible_uses=None):
    """One-hot encodes primary_use."""
    df = pd.get_dummies(df, columns=['primary_use'])
    if all_possible_uses is not None:
        # Ensure all columns exist, even if not present in this split
        for use in all_possible_uses:
            col_name = f'primary_use_{use}'
            if col_name not in df.columns:
                df[col_name] = 0
                
    return df

def prepare_features(df, all_possible_uses=None):
    """Executes the full feature engineering pipeline and splits X, y."""
    df = add_temporal_features(df.copy())
    df = transform_target(df)
    df = handle_missing(df)
    df = encode_categoricals(df, all_possible_uses)
    
    # Drop columns not used for training
    cols_to_drop = ['meter_reading', 'timestamp', 'building_id', 'site_id', 'meter']
    # Also drop year_built, floor_count as they often have too many missing values in ASHRAE dataset
    if 'year_built' in df.columns:
        cols_to_drop.append('year_built')
    if 'floor_count' in df.columns:
        cols_to_drop.append('floor_count')
        
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    y = df['meter_reading']
    return X, y, list(X.columns)

def align_features(X_source, X_target, target_feature_names=None):
    """Ensures X_target has identical columns to X_source, in the exact same order."""
    if target_feature_names is None:
        target_feature_names = X_source.columns

    for col in target_feature_names:
        if col not in X_target.columns:
            X_target[col] = 0
            
    # Drop extra columns not in source, and reorder to match source
    X_target = X_target[target_feature_names]
    return X_target

def scale_features(X_train, X_test, scaler=None):
    """Scales features using StandardScaler."""
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)
        
    X_test_scaled = scaler.transform(X_test)
    
    # Return as DataFrames to keep column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, X_test_scaled, scaler
