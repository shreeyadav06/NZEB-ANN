"""
Transfer learning utilities module.
"""

import time
import tensorflow as tf
from sklearn.model_selection import KFold
import pandas as pd
from .metrics import evaluate_original_scale
from .model import train_model

def load_source_model(path):
    """Loads a saved Keras model."""
    return tf.keras.models.load_model(path)

def freeze_layers(model, n_freeze=3):
    """Freezes the first n_freeze layers of the model."""
    for i, layer in enumerate(model.layers):
        if i < n_freeze:
            layer.trainable = False
        else:
            layer.trainable = True
    return model

def recompile_for_transfer(model, learning_rate=1e-4):
    """Recompiles model for fine-tuning."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def transfer_and_train(source_model_path, X_target, y_target, n_freeze=3, 
                       lr=1e-4, epochs=100, patience=10, validation_split=0.2):
    """Full transfer learning pipeline on target data."""
    model = load_source_model(source_model_path)
    model = freeze_layers(model, n_freeze=n_freeze)
    model = recompile_for_transfer(model, learning_rate=lr)
    
    model, history, train_time = train_model(
        model, X_target, y_target,
        epochs=epochs, patience=patience, validation_split=validation_split
    )
    return model, history, train_time

def transfer_with_kfold(source_model_path, X_target, y_target, n_folds=5, 
                        n_freeze=3, lr=1e-4, epochs=100, patience=10):
    """Performs K-Fold cross-validation for a transfer learning model."""
    kf = KFold(n_splits=n_folds, shuffle=True)
    
    fold_results = []
    
    if isinstance(X_target, pd.DataFrame):
        X_arr = X_target.values
        y_arr = y_target.values
    else:
        X_arr = X_target
        y_arr = y_target
        
    for fold, (train_index, test_index) in enumerate(kf.split(X_arr)):
        X_train_fold, X_test_fold = X_arr[train_index], X_arr[test_index]
        y_train_fold, y_test_fold = y_arr[train_index], y_arr[test_index]
        
        # Load fresh source model and freeze/recompile
        model = load_source_model(source_model_path)
        model = freeze_layers(model, n_freeze=n_freeze)
        model = recompile_for_transfer(model, learning_rate=lr)
        
        # Train
        ckpt_path = f"/kaggle/working/transfer_fold_{fold+1}.keras" if 'KAGGLE_KERNEL_RUN_TYPE' in __import__('os').environ else None
        
        model, history, train_time = train_model(
            model, X_train_fold, y_train_fold, 
            epochs=epochs, patience=patience,
            checkpoint_path=ckpt_path
        )
        
        # Evaluate
        y_pred = model.predict(X_test_fold, verbose=0).flatten()
        eval_metrics = evaluate_original_scale(y_test_fold, y_pred)
        
        eval_metrics['fold'] = fold + 1
        eval_metrics['train_time'] = train_time
        eval_metrics['epochs_run'] = len(history.history['loss'])
        
        fold_results.append(eval_metrics)
        
    return fold_results

def print_transfer_summary(model):
    """Prints a summary of frozen vs. trainable layers."""
    print("Transfer Learning Model Summary:")
    for i, layer in enumerate(model.layers):
        status = "Trainable" if layer.trainable else "Frozen"
        print(f"Layer {i}: {layer.name} - {status} ({layer.count_params()} params)")
