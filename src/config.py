"""
Configuration settings for the NZEB ANN Climate-Aware Transfer Learning project.
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

# Bidirectional transfer scenarios (source -> target)
TRANSFER_SCENARIOS = [
    {"source": 2, "target": 4},   # Tempe -> Berkeley
    {"source": 2, "target": 15},  # Tempe -> Cornell
    {"source": 4, "target": 15},  # Berkeley -> Cornell
    {"source": 4, "target": 2},   # Berkeley -> Tempe
    {"source": 15, "target": 2},  # Cornell -> Tempe
    {"source": 15, "target": 4},  # Cornell -> Berkeley
]

# --- Neural Network Hyperparameters ---
ANN_LAYERS = [256, 128, 64, 32]
ACTIVATION = 'relu'

# Training Hyperparameters
LEARNING_RATE = 1e-5
EPOCHS = 100
PATIENCE = 10
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 32

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
