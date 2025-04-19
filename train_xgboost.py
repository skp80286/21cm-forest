import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import f21_predict_base as base
import plot_results as pltr
import F21Stats
import Scaling
import argparse
import os

def load_training_data(override_path, samples):
    files = base.get_datafile_list('noisy', args, extn='csv', filter='train_only', override_path=override_path)
    X_train = np.zeros((samples*len(files), 24))
    y_train = np.zeros((samples*len(files), 2))
    for i, file in enumerate(files):
        curr_xHI = float(file.split('xHI')[1].split('_')[0])
        curr_logfX = float(file.split('fX')[1].split('_')[0])
        y_train[i*samples:(i+1)*samples, 0] = curr_xHI
        y_train[i*samples:(i+1)*samples, 1] = curr_logfX
        currps = np.loadtxt(file)
        X_train[i*samples:(i+1)*samples, :] = currps[:samples,:24]
    return X_train, y_train

def load_test_data(override_path):
    files = base.get_datafile_list('noisy', args, extn='csv', filter='test_only', override_path=override_path)
    X_test = np.zeros((10000*len(files), 24))
    y_test = np.zeros((10000*len(files), 2))
    for i, file in enumerate(files):
        curr_xHI = float(file.split('xHI')[1].split('_')[0])
        curr_logfX = float(file.split('fX')[1].split('_')[0])
        y_test[i*10000:(i+1)*10000, 0] = curr_xHI
        y_test[i*10000:(i+1)*10000, 1] = curr_logfX
        currps = np.loadtxt(file)
        currps_boot = F21Stats.bootstrap(ps=currps, reps=10000, size=10)
        X_test[i*10000:(i+1)*10000, :] = currps_boot[:,:24]
    return X_test, y_test

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train XGBoost model for 21cm forest parameter inference')
    parser.add_argument('--train_path', type=str, required=True, help='Path to training data files')
    parser.add_argument('--test_path', type=str, required=True, help='Path to test data files')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples per training file')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save results')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load training and test data
    print("Loading training data...")
    X_train, y_train = load_training_data(args.train_path, args.samples)
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

    print("Loading test data...")
    X_test, y_test = load_test_data(args.test_path)
    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")

    # Initialize XGBoost model
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )

    # Perform cross-validation
    print(f"\nPerforming {args.n_folds}-fold cross-validation...")
    kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    
    # Cross-validation for xHI (first target)
    xhi_scores = cross_val_score(model, X_train, y_train[:, 0], cv=kfold, scoring='r2')
    print("\nxHI Cross-validation Results:")
    print(f"R2 scores: {xhi_scores}")
    print(f"Mean R2: {xhi_scores.mean():.4f} (+/- {xhi_scores.std() * 2:.4f})")
    
    # Cross-validation for logfX (second target)
    logfX_scores = cross_val_score(model, X_train, y_train[:, 1], cv=kfold, scoring='r2')
    print("\nlogfX Cross-validation Results:")
    print(f"R2 scores: {logfX_scores}")
    print(f"Mean R2: {logfX_scores.mean():.4f} (+/- {logfX_scores.std() * 2:.4f})")

    # Train final model on full training set
    print("\nTraining final model on full training set...")
    model.fit(X_train, y_train)

    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("\nFinal Model Performance on Test Set:")
    print(f"R2 Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Plot xHI predictions
    plt.subplot(1, 2, 1)
    plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.5)
    plt.plot([y_test[:, 0].min(), y_test[:, 0].max()], 
             [y_test[:, 0].min(), y_test[:, 0].max()], 'r--')
    plt.xlabel('True xHI')
    plt.ylabel('Predicted xHI')
    plt.title('xHI Predictions')
    
    # Plot logfX predictions
    plt.subplot(1, 2, 2)
    plt.scatter(y_test[:, 1], y_pred[:, 1], alpha=0.5)
    plt.plot([y_test[:, 1].min(), y_test[:, 1].max()], 
             [y_test[:, 1].min(), y_test[:, 1].max()], 'r--')
    plt.xlabel('True logfX')
    plt.ylabel('Predicted logfX')
    plt.title('logfX Predictions')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'predictions.png'))
    print(f"\nResults saved to {args.output_dir}")

if __name__ == '__main__':
    main() 