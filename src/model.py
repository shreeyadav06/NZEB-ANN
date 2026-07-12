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

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train_model(model, X_train, y_train, epochs=100, patience=10, validation_split=0.2, checkpoint_path=None):
    """Trains the model with early stopping and optional checkpointing."""
    callbacks = []
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=patience, 
        restore_best_weights=True
    )
    callbacks.append(early_stopping)
    
    if checkpoint_path:
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)
    
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=2 # Show progress per epoch
    )
    training_time = time.time() - start_time
    
    return model, history, training_time

def train_with_kfold(X, y, n_folds=5, layers=[256, 128, 64, 32], 
                     learning_rate=1e-5, epochs=100, patience=10, site_name="unknown"):
    """Performs K-Fold cross-validation for a baseline model."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
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
        ckpt_filename = f"{site_name}_baseline_fold_{fold+1}.keras"
        ckpt_path_working = f"/kaggle/working/{ckpt_filename}" if 'KAGGLE_KERNEL_RUN_TYPE' in __import__('os').environ else None
        
        # Determine actual checkpoint path to load
        load_path = None
        if ckpt_path_working:
            if __import__('os').path.exists(ckpt_path_working):
                load_path = ckpt_path_working
            else:
                # Search in kaggle input (in case they attached previous run output)
                for root, dirs, files in __import__('os').walk('/kaggle/input'):
                    if ckpt_filename in files:
                        load_path = __import__('os').path.join(root, ckpt_filename)
                        break
        
        if load_path:
            print(f"Loading existing checkpoint {ckpt_filename} from {load_path}...")
            model = tf.keras.models.load_model(load_path)
            train_time = 0.0
            epochs_run = 0
        else:
            model, history, train_time = train_model(
                model, X_train_fold, y_train_fold, 
                epochs=epochs, patience=patience,
                checkpoint_path=ckpt_path_working
            )
            epochs_run = len(history.history['loss'])
        
        # Evaluate
        y_pred = model.predict(X_test_fold, verbose=0).flatten()
        eval_metrics = evaluate_original_scale(y_test_fold, y_pred)
        
        eval_metrics['fold'] = fold + 1
        eval_metrics['train_time'] = train_time
        eval_metrics['epochs_run'] = epochs_run
        
        fold_results.append(eval_metrics)
        
    return fold_results
