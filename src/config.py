"""
Configuration settings for the NZEB ANN project.
Title: Transfer Learning for Cross-Climate Building Energy Prediction: Investigating Data Efficiency and Generalization Across Climate Zones
"""

import os

# --- Sites & Climate Zones ---
SITES = {
    2: "Tempe",
    4: "Berkeley",
    15: "Cornell"
}

CLIMATE_ZONES = {
    2: "2B",
    4: "3C",
    15: "6A"
}

# ASHRAE ordinal climate distance matrix
# 2B=2, 3C=3, 6A=6 in ordinal terms roughly, but we defined:
# |2B - 3C| = 1
# |2B - 6A| = 4
# |3C - 6A| = 3
CLIMATE_DISTANCE_MATRIX = {
    2: {2: 0, 4: 1, 15: 4},
    4: {2: 1, 4: 0, 15: 3},
    15: {2: 4, 4: 3, 15: 0}
}

# --- Data Efficiency is the Primary Focus ---
DATA_EFFICIENCY_IS_PRIMARY = True

# PRIMARY experiment: Maximum climate distance
PRIMARY_SCENARIO = {"source": 2, "target": 15}  # Tempe -> Cornell

# SECONDARY experiments: Varying climate distance for secondary contribution
SECONDARY_SCENARIOS = [
    {"source": 4, "target": 15},  # Berkeley -> Cornell
    {"source": 2, "target": 4},   # Tempe -> Berkeley
]

# --- Neural Network Hyperparameters ---
ANN_LAYERS = [256, 128, 64, 32]
ACTIVATION = 'relu'

# Training Hyperparameters
LEARNING_RATE = 1e-5
EPOCHS = 50
PATIENCE = 10
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 256

# Transfer Learning Hyperparameters
N_FREEZE_LAYERS = 3
TRANSFER_LEARNING_RATE = 1e-4

# --- Experiment Settings ---
N_FOLDS = 5
DATA_FRACTIONS = [0.05, 0.10, 0.20, 0.30, 0.50, 1.0]
SEEDS = [42, 123, 456]

# --- Paths (Local and Colab) ---
# When running on Colab, these paths will be adapted or prefixed with /content/drive/...
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
