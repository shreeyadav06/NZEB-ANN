"""
Data loading and validation module.
"""

import pandas as pd
import numpy as np

def mount_drive():
    """Mounts Google Drive in Colab environment."""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive mounted successfully.")
    except ImportError:
        print("Not running in Google Colab. Skipping Drive mount.")

def load_merged_data(path):
    """Loads the pre-merged CSV data from the given path."""
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    return df

def validate_data(df, expected_sites=[2, 4, 15]):
    """Validates the loaded DataFrame for required columns and target sites."""
    required_cols = [
        'building_id', 'site_id', 'timestamp', 'meter', 'meter_reading',
        'primary_use', 'square_feet', 'air_temperature'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")
    
    missing_sites = [site for site in expected_sites if site not in df.site_id.unique()]
    if missing_sites:
        raise ValueError(f"Missing required sites in dataset: {missing_sites}")
        
    print(f"Data validation passed. Shape: {df.shape}")
    print(f"Sites present: {df.site_id.unique()}")
    return True

def split_by_site(df):
    """Splits the dataset into a dictionary of DataFrames keyed by site_id."""
    site_dfs = {}
    for site in df.site_id.unique():
        site_dfs[site] = df[df.site_id == site].copy()
    return site_dfs
