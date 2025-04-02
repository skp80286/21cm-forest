import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import f21_predict_base as base
import plot_results as pltr
import Scaling
import optuna
from sklearn.model_selection import cross_val_score


class Args:
    def __init__(self, runmode='', telescope='uGMRT', t_int=500):
        self.runmode = runmode
        self.telescope = telescope
        self.t_int = t_int
        self.scale_y = True

args = Args() 
output_dir = base.create_output_dir(args)
logger = base.setup_logging(output_dir)

# Load training data using numpy
del_ind = [1,2,3,5,9,11,12] # 6,13
logger.info(f"Removing features with indices {del_ind}")
# Load training data using numpy
train_data = np.loadtxt('saved_output/bispectrum_data_20k/all_training_data.csv', delimiter=',')
X_train = np.delete(train_data[:,:-2], del_ind, axis=1) 
y_train = train_data[:, -2:]    # Last two columns as output
logger.info(f"Loaded training data: {X_train.shape} {y_train.shape} (skipped indices {del_ind})")

# Load test data using numpy
test_data = np.loadtxt('saved_output/bispectrum_data_20k/all_test_data.csv', delimiter=',')
X_test = np.delete(test_data[:,:-2], del_ind, axis=1)  
y_test = test_data[:, -2:]    # Last two columns as output
logger.info(f"Loaded test data: {X_test.shape} {y_test.shape} (skipped indices {del_ind})")


# Define the objective function
def objective(trial):
    # Define the hyperparameter search space
    """
    param = {
        'booster': 'gbtree', #trial.suggest_categorical(name='booster', choices=['dart', 'gbtree', 'gblinear']),
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 1),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
    }
    """
    param = {
        'verbosity': 0,
        'objective': 'reg:squarederror',  # or 'binary:logistic' for classification
        'booster': 'dart',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 10, log=True),
        'gamma': trial.suggest_float('gamma', 0, 1),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-3, 10, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10, log=True),
        'rate_drop': trial.suggest_float('rate_drop', 0.0, 0.5),
        'skip_drop': trial.suggest_float('skip_drop', 0.0, 0.5),
    }

    # Initialize the model with the parameters
    model = XGBRegressor(**param)

    # Perform cross-validation and return the negative mean squared error
    score = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5).mean()
    return score

# Create an Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000)

# Output the best hyperparameters
logger.info("Best trial:")
trial = study.best_trial
logger.info(f"  Value: {trial.value}")
logger.info("  Params: ")
for key, value in trial.params.items():
    logger.info(f"    {key}: {value}")

# Train the final model with the best parameters
best_params = study.best_params
final_model = XGBRegressor(**best_params)
final_model.fit(X_train, y_train)

# Evaluate the final model
score = final_model.score(X_test, y_test)
logger.info(f"Final model R² score: {score}")
