import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import f21_predict_base as base
import F21Stats
import Scaling
import optuna
from sklearn.model_selection import cross_val_score
import torch
from Scaling import Scaler

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
        #print(f"shape of currps={currps.shape}")
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
        print(f"shape of currps={currps.shape}")
        X_test[i*10000:(i+1)*10000, :] = currps_boot[:,:24]

    return X_test, y_test

def save_model(model):
    # Save the model architecture and weights
    logger.info(f'Saving model to: {output_dir}/psbs_xgb_model.pth')
    model_json = model.save_model(f"{output_dir}/psbs_xgb_model.pth")


# Define the objective function
def objective(trial):
    # Define the hyperparameter search space
    param = {
        'booster': 'gbtree', #trial.suggest_categorical(name='booster', choices=['dart', 'gbtree', 'gblinear']),
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 80, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        #'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        #'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
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
    """
    
    # Initialize the model with the parameters
    model = XGBRegressor(**param)

    # Perform cross-validation and return the negative mean squared error
    score = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5).mean()
    return score

# main code starts here
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.determinisitc=True
torch.backends.cudnn.benchmark=False


parser = base.setup_args_parser()
args = parser.parse_args()
output_dir = base.create_output_dir(args)
logger = base.setup_logging(output_dir)

scaler = Scaler(args)

# Load training data using numpy
del_ind = [] #[1,2,3,5,9,11,12] # 6,13
logger.info(f"Removing features with indices {del_ind}")
# Load training data using numpy
#X_train, y_train = load_training_data("output/f21_ps_dum_train_test_uGMRT_t500_20250325152125/ps/", args.limitsamplesize) #np.loadtxt('saved_output/bispectrum_data_20k/all_training_data.csv', delimiter=',')
X_train, y_train = load_training_data("output/f21_ps_dum_train_test_uGMRT_t500_20250326171749/ps/", args.limitsamplesize) 

X_train = np.delete(X_train, del_ind, axis=1) 
X_train, y_train = scaler.scaleXy(X_train, y_train)
logger.info(f"Loaded training data: {X_train.shape} {y_train.shape} (skipped indices {del_ind})")

# Load test data using numpy
X_test, y_test = load_test_data("output/f21_ps_dum_train_test_uGMRT_t500_20250326171749/test_ps/")
X_test = np.delete(X_test, del_ind, axis=1) 
X_test, y_test = scaler.scaleXy(X_test, y_test)
logger.info(f"Loaded test data: {X_test.shape} {y_test.shape} (skipped indices {del_ind})")
logger.info(f"Sample test data:\n{X_test[:10]}\n{X_test[10000:10005]}\n===\n{y_test[:10]}\n{y_test[10000:10005]}")

# Run the default regressor first
base_model = XGBRegressor(objective='reg:absoluteerror')
base_model.fit(X_train, y_train)
logger.info(f"Fitted regressor: {base_model}")
logger.info(f"Booster: {base_model.get_booster()}")
feature_importance = base_model.feature_importances_
save_model(base_model)
np.savetxt(f"{output_dir}/feature_importance.csv", feature_importance, delimiter=',')
logger.info(f"Feature importance: {feature_importance}")
for imp_type in ['weight','gain', 'cover', 'total_gain', 'total_cover']:
    logger.info(f"Importance type {imp_type}: {base_model.get_booster().get_score(importance_type=imp_type)}")

# Evaluate the final model
y_pred = base_model.predict(X_test)
r2_means = base.summarize_test_1000(y_pred, y_test, output_dir)
tse_means, rmse_means = base.calc_squared_error(y_pred, y_test)
logger.info(f"Base model: Final R2 score with means: {r2_means}, RMSE (means): {rmse_means}")
# Create an Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=args.trials)

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
logger.info(f"Fitted regressor: {final_model}")
logger.info(f"Booster: {final_model.get_booster()}")
feature_importance = final_model.feature_importances_
save_model(final_model)
np.savetxt(f"{output_dir}/feature_importance.csv", feature_importance, delimiter=',')
logger.info(f"Feature importance: {feature_importance}")
for imp_type in ['weight','gain', 'cover', 'total_gain', 'total_cover']:
    logger.info(f"Importance type {imp_type}: {final_model.get_booster().get_score(importance_type=imp_type)}")

# Evaluate the final model
y_pred = final_model.predict(X_test)
r2_means = base.summarize_test_1000(y_pred, y_test, output_dir)
mse = base.mean_squared_error(y_pred, y_test)
rmse_all = base.rmse_all(y_pred, y_test)
logger.info(f"Optimization complete. Final R2 score with means: {r2_means}, mse: {mse} rmse_all: {rmse_all}")
