"""
Model building and training module.
"""

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
import pandas as pd
from .metrics import evaluate_original_scale

def build_ann(input_dim, layers=[256, 128, 64, 32], activation='relu'):
    """Builds the ANN architecture."""
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    
    for units in layers:
        model.add(Dense(units, activation=activation))
        
    model.add(Dense(1)) # Output layer for regression
    return model

def compile_model(model, learning_rate=1e-5):
    """Compiles the model with Adam optimizer and MSE loss."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def train_model(model, X_train, y_train, epochs=100, patience=10, validation_split=0.2):
    """Trains the model with early stopping."""
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=patience, 
        restore_best_weights=True
    )
    
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=0 # Silent training, let notebook handle progress
    )
    training_time = time.time() - start_time
    
    return model, history, training_time

def train_with_kfold(X, y, n_folds=5, layers=[256, 128, 64, 32], 
                     learning_rate=1e-5, epochs=100, patience=10):
    """Performs K-Fold cross-validation for a baseline model."""
    kf = KFold(n_splits=n_folds, shuffle=True)
    
    fold_results = []
    
    # We expect X, y to be scaled numpy arrays or pandas DataFrames.
    # KFold works best with numpy indexing.
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
        y_arr = y.values
    else:
        X_arr = X
        y_arr = y
        
    for fold, (train_index, test_index) in enumerate(kf.split(X_arr)):
        X_train_fold, X_test_fold = X_arr[train_index], X_arr[test_index]
        y_train_fold, y_test_fold = y_arr[train_index], y_arr[test_index]
        
        # Build fresh model
        model = build_ann(X_train_fold.shape[1], layers=layers)
        model = compile_model(model, learning_rate=learning_rate)
        
        # Train
        model, history, train_time = train_model(
            model, X_train_fold, y_train_fold, 
            epochs=epochs, patience=patience
        )
        
        # Evaluate
        y_pred = model.predict(X_test_fold, verbose=0).flatten()
        eval_metrics = evaluate_original_scale(y_test_fold, y_pred)
        
        eval_metrics['fold'] = fold + 1
        eval_metrics['train_time'] = train_time
        eval_metrics['epochs_run'] = len(history.history['loss'])
        
        fold_results.append(eval_metrics)
        
    return fold_results
