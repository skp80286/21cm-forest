import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import f21_predict_base as base
import plot_results as pltr
import F21Stats
import Scaling
import optuna
from sklearn.model_selection import cross_val_score
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin

def load_training_data(override_path):
    files = base.get_datafile_list('noisy', args, extn='csv', filter='train_only', override_path=override_path)
    X_train = np.zeros((100*len(files), 24))
    y_train = np.zeros((100*len(files), 2))
    for i, file in enumerate(files):
        curr_xHI = float(file.split('xHI')[1].split('_')[0])
        curr_logfX = float(file.split('fX')[1].split('_')[0])
        y_train[i*100:(i+1)*100, 0] = curr_xHI
        y_train[i*100:(i+1)*100, 1] = curr_logfX
        currps = np.loadtxt(file)
        #print(f"shape of currps={currps.shape}")
        X_train[i*100:(i+1)*100, :] = currps[:,:24]

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
    logger.info(f'Saving model to: {output_dir}/psbs_nn_model.pth')
    torch.save(model.state_dict(), f"{output_dir}/psbs_nn_model.pth")  # Save model weights
    # Optionally, save the model architecture as well
    torch.save(model, f"{output_dir}/psbs_nn_model_full.pth")  # Save full model

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

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # First hidden layer
        self.fc2 = nn.Linear(64, 32)           # Second hidden layer
        self.fc3 = nn.Linear(32, 2)            # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))            # Activation function
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def print_architecture(self):
        print(self)

class NeuralNetworkWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model, criterion, optimizer, epochs):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = epochs

    def fit(self, X, y):
        self.model.train()
        for epoch in range(self.num_epochs):  # You can adjust the number of epochs
            self.optimizer.zero_grad()
            outputs = self.model(torch.FloatTensor(X))
            loss = self.criterion(outputs, torch.FloatTensor(y))
            loss.backward()
            self.optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.FloatTensor(X)).numpy()

# main code starts here
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.determinisitc=True
torch.backends.cudnn.benchmark=False


parser = base.setup_args_parser()
args = parser.parse_args()
output_dir = base.create_output_dir(args)
logger = base.setup_logging(output_dir)

# Load training data using numpy
del_ind = [] #[1,2,3,5,9,11,12] # 6,13
logger.info(f"Removing features with indices {del_ind}")
# Load training data using numpy
X_train, y_train = load_training_data("output/f21_ps_dum_train_test_uGMRT_t500_20250325152125/ps/") #np.loadtxt('saved_output/bispectrum_data_20k/all_training_data.csv', delimiter=',')
X_train = np.delete(X_train, del_ind, axis=1) 
logger.info(f"Loaded training data: {X_train.shape} {y_train.shape} (skipped indices {del_ind})")

# Load test data using numpy
X_test, y_test = load_test_data("output/f21_ps_dum_train_test_uGMRT_t500_20250325152125/test_ps/") #np.loadtxt('saved_output/bispectrum_data_20k/all_training_data.csv', delimiter=',')
X_test = np.delete(X_test, del_ind, axis=1) 
logger.info(f"Loaded test data: {X_test.shape} {y_test.shape} (skipped indices {del_ind})")
logger.info(f"Sample test data:\n{X_test[:10]}\n{X_test[10000:10005]}\n===\n{y_test[:10]}\n{y_test[10000:10005]}")

# Initialize the neural network
input_size = X_train.shape[1]
model = NeuralNetwork(input_size)
model.print_architecture()
criterion = nn.MSELoss()  # Loss function
optimizer = optim.Adam(model.parameters(), lr=0.000001)  # Optimizer
#nn_wrapper = NeuralNetworkWrapper(model, criterion, optimizer, args.epochs)

for epoch in range(args.epochs):
    model.train()
    optimizer.zero_grad()  # Zero the gradients
    outputs = model(torch.FloatTensor(X_train))  # Forward pass
    loss = criterion(outputs, torch.FloatTensor(y_train))  # Compute loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights

    # Cross-validation
    if (epoch+1) % 10000 == 0:
        #cv_score = cross_val_score(nn_wrapper, X_train, y_train, scoring='neg_mean_squared_error', cv=5).mean()
        model.eval()
        with torch.no_grad():
            y_pred = model(torch.FloatTensor(X_test)).numpy()  # Predictions
        test_loss = mean_squared_error(y_test, y_pred)
        logger.info(f'Epoch [{epoch+1}/{args.epochs}], Training Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}')


save_model(model)

# Evaluate the final model
model.eval()
with torch.no_grad():
    y_pred = model(torch.FloatTensor(X_test)).numpy()  # Predictions
r2_means = pltr.summarize_test_1000(y_pred, y_test, output_dir)
tse_means, rmse_means = base.calc_squared_error(y_pred, y_test)
logger.info(f"Neural Network model: Final R2 score with means: {r2_means}, RMSE (means): {rmse_means}")