"""
Metrics and evaluation module.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import os

def evaluate(y_true, y_pred):
    """Calculates R2, MAPE, and RMSE."""
    return {
        'R2': r2_score(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))
    }

def evaluate_original_scale(y_true_log, y_pred_log):
    """Inverts the log1p transform and evaluates."""
    y_true_orig = np.expm1(y_true_log)
    y_pred_orig = np.expm1(y_pred_log)
    return evaluate(y_true_orig, y_pred_orig)

def summarize_kfold(fold_results):
    """Computes mean and std deviation across k-fold results."""
    summary = {}
    metrics = fold_results[0].keys()
    
    for metric in metrics:
        values = [res[metric] for res in fold_results]
        summary[f'{metric}_mean'] = np.mean(values)
        summary[f'{metric}_std'] = np.std(values)
        
    return summary

def log_experiment(result_dict, csv_path):
    """Appends experiment results to a CSV file."""
    df_new = pd.DataFrame([result_dict])
    
    # Check if file exists to write header or not
    file_exists = os.path.isfile(csv_path)
    df_new.to_csv(csv_path, mode='a', header=not file_exists, index=False)

def results_to_dataframe(results_list):
    """Converts a list of result dictionaries to a pandas DataFrame."""
    return pd.DataFrame(results_list)

def compute_crossover_point(df_results, scratch_100_r2, metric='R2_mean'):
    """
    Interpolates to find the data fraction at which transfer learning matches 
    the scratch model trained on 100% data.
    """
    df_transfer = df_results[df_results['model_type'] == 'Transfer'].sort_values('Data_Fraction')
    
    fractions = df_transfer['Data_Fraction'].values
    scores = df_transfer[metric].values
    
    # Simple linear interpolation
    crossover = np.interp(scratch_100_r2, scores, fractions)
    return crossover

def compute_data_savings(transfer_fraction, scratch_fraction):
    """
    Computes how much less data transfer needs vs. scratch.
    """
    if scratch_fraction == 0:
        return 0
    return (1 - (transfer_fraction / scratch_fraction)) * 100

def statistical_significance_test(transfer_scores, scratch_scores):
    """
    Paired t-test or Wilcoxon signed-rank test comparing transfer vs. scratch.
    """
    try:
        from scipy import stats
        # Use Wilcoxon signed-rank test for paired samples (non-parametric)
        stat, p_value = stats.wilcoxon(transfer_scores, scratch_scores)
        return {'statistic': stat, 'p_value': p_value}
    except ImportError:
        print("scipy is required for statistical significance testing.")
        return None
