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

def plot_climate_distance_vs_gain(df_distance_gain, save_name=None):
    """
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
    """Plots the bidirectional transferability heatmap."""
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
