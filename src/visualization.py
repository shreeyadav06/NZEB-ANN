"""
Visualization module for generating paper-quality figures.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set global seaborn styling for publication quality
sns.set_theme(style="whitegrid", font_scale=1.2)

def save_figure(fig, filename, figures_dir="figures"):
    """Saves figure in both PNG and PDF formats."""
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
        
    png_path = os.path.join(figures_dir, f"{filename}.png")
    pdf_path = os.path.join(figures_dir, f"{filename}.pdf")
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved figure to {png_path} and {pdf_path}")

def plot_training_curves(history_scratch, history_transfer, title="Training Convergence Comparison", save_name=None):
    """Plots validation loss curves for scratch vs transfer models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(history_scratch.history['val_loss'], label='Scratch (Baseline) Val Loss', color='blue', linestyle='--')
    ax.plot(history_transfer.history['val_loss'], label='Transfer Val Loss', color='red', linestyle='-')
    
    ax.set_title(title)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Validation Loss (MSE)")
    ax.legend()
    
    if save_name:
        save_figure(fig, save_name)
    return fig

def plot_data_efficiency(df_results, scenario_name, save_name=None):
    """
    Plots Target Data % vs R2 for Transfer vs Scratch for a given scenario.
    df_results should contain: 'Data_Fraction', 'Method' (Transfer/Scratch), 'R2'
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.lineplot(
        data=df_results, 
        x='Data_Fraction', y='R2', hue='Method', 
        marker='o', ax=ax, errorbar='sd'
    )
    
    ax.set_title(f"Data Efficiency: {scenario_name}")
    ax.set_xlabel("Target Data Fraction")
    ax.set_ylabel("R² Score")
    ax.set_xscale('log') # often useful if fractions are [0.05, 0.1, 0.2, 0.5, 1.0]
    ax.set_xticks(df_results['Data_Fraction'].unique())
    ax.set_xticklabels([f"{x*100:.0f}%" for x in df_results['Data_Fraction'].unique()])
    
    if save_name:
        save_figure(fig, save_name)
    return fig

def plot_data_efficiency_primary(df_results, scratch_100_baseline, scenario_name="Tempe to Cornell", save_name=None):
    """
    Enhanced version of data efficiency plot that adds a horizontal dashed line 
    for the scratch-100% baseline and clearly marks the crossover point.
    This is the key figure of the paper.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    sns.lineplot(
        data=df_results, 
        x='Data_Fraction', y='R2_mean', hue='model_type', 
        marker='o', ax=ax, errorbar='sd'
    )
    
    # Add horizontal baseline for 100% scratch
    ax.axhline(y=scratch_100_baseline, color='blue', linestyle='--', alpha=0.7, label='Scratch (100% Data)')
    
    ax.set_title(f"Data Efficiency and Crossover: {scenario_name}")
    ax.set_xlabel("Target Data Fraction")
    ax.set_ylabel("R² Score")
    ax.set_xscale('log')
    
    fractions = sorted(df_results['Data_Fraction'].unique())
    ax.set_xticks(fractions)
    ax.set_xticklabels([f"{x*100:.0f}%" for x in fractions])
    ax.legend()
    
    if save_name:
        save_figure(fig, save_name)
    return fig

def plot_convergence_comparison(history_transfer, history_scratch, fraction, save_name=None):
    """
    Shows how transfer learning converges faster at a specific data fraction.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(history_scratch.history['val_loss'], label=f'Scratch ({fraction*100:.0f}%) Val Loss', color='blue', linestyle='--')
    ax.plot(history_transfer.history['val_loss'], label=f'Transfer ({fraction*100:.0f}%) Val Loss', color='red', linestyle='-')
    
    ax.set_title(f"Convergence Comparison at {fraction*100:.0f}% Target Data")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Validation Loss (MSE)")
    ax.legend()
    
    if save_name:
        save_figure(fig, save_name)
    return fig

def plot_climate_justification(site_weather_dfs, save_name=None):
    """
    Side-by-side weather distributions (temperature) to justify domain selection.
    Positions climate difference as domain justification, not primary analysis.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Combine data for seaborn
    combined_df = pd.concat([
        df[['air_temperature']].assign(Site=name) 
        for name, df in site_weather_dfs.items()
    ])
    
    sns.kdeplot(data=combined_df, x="air_temperature", hue="Site", fill=True, ax=ax)
    
    ax.set_title("Climate Justification: Air Temperature Distributions")
    ax.set_xlabel("Air Temperature (°C)")
    ax.set_ylabel("Density")
    
    if save_name:
        save_figure(fig, save_name)
    return fig


def plot_climate_distance_vs_gain(df_distance_gain, save_name=None):
    """
    (Secondary contribution figure)
    Plots climate distance vs transfer gain (Δ R2).
    df_distance_gain should contain: 'Climate_Distance', 'Transfer_Gain', 'Scenario_Label'
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.regplot(
        data=df_distance_gain, 
        x='Climate_Distance', y='Transfer_Gain',
        ax=ax, scatter_kws={'s':100}, color='purple'
    )
    
    # Annotate points
    for i, row in df_distance_gain.iterrows():
        ax.annotate(row['Scenario_Label'], 
                    (row['Climate_Distance'], row['Transfer_Gain']),
                    xytext=(5, 5), textcoords='offset points')
        
    ax.set_title("Impact of Climate Distance on Transfer Learning Effectiveness")
    ax.set_xlabel("ASHRAE Climate Distance (Ordinal Difference)")
    ax.set_ylabel("Transfer Gain (Δ R²: Transfer - Scratch)")
    
    if save_name:
        save_figure(fig, save_name)
    return fig

def plot_transferability_heatmap(matrix, xticklabels, yticklabels, save_name=None):
    """
    (Secondary contribution figure)
    Plots the bidirectional transferability heatmap.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        matrix, annot=True, cmap="YlGnBu", fmt=".3f", 
        xticklabels=xticklabels, yticklabels=yticklabels, ax=ax
    )
    
    ax.set_title("Transferability Matrix (R² Scores)")
    ax.set_xlabel("Target Climate")
    ax.set_ylabel("Source Climate")
    
    if save_name:
        save_figure(fig, save_name)
    return fig
