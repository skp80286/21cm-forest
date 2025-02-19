'''
Predict parameters fX and xHI from the 21cm forest data using CNN.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import argparse
import glob
from datetime import datetime

import F21DataLoader as dl
import f21_predict_base as base
import Scaling
import PS1D
import F21Stats as f21stats
from UnetModel import UnetModel

import numpy as np
import sys

import matplotlib.pyplot as plt

import optuna
from xgboost import XGBRegressor

# main code starts here
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.determinisitc=True
torch.backends.cudnn.benchmark=False

parser = base.setup_args_parser()
parser.add_argument('--test_multiple', action='store_true', help='Test 1000 sets of 10 LoS for each test point and plot it')
parser.add_argument('--test_reps', type=int, default=10000, help='Test repetitions for each parameter combination')
args = parser.parse_args()
if args.input_points_to_use not in [2048, 128]: raise ValueError(f"Invalid input_points_to_use {args.input_points_to_use}")
if args.input_points_to_use == 2048: 
    step = 4
    kernel1 = 256
else: 
    step = 2
    kernel1 = 16

output_dir = base.create_output_dir(args=args)
logger = base.setup_logging(output_dir)

## Loading data
datafiles = base.get_datafile_list(type='noisy', args=args)

test_points = [[-3.00,0.11],[-2.00,0.11],[-1.00,0.11],[-3.00,0.25],[-2.00,0.25],[-1.00,0.25],[-3.00,0.52],[-2.00,0.52],[-1.00,0.52], [-3.00,0.80],[-2.00,0.80],[-1.00,0.80]]#,[0.00,0.80]]
train_files = []
test_files = []
for nof in datafiles:
    is_test_file = False
    for p in test_points:
        if nof.find(f"fX{p[0]:.2f}_xHI{p[1]:.2f}") >= 0:
            test_files.append(nof)
            is_test_file = True
            break
    if not is_test_file:
        train_files.append(nof)

# Initialize the network
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

logger.info("####")
logger.info(f"### Using \"{device}\" device ###")
logger.info("####")

## Load the trained Unet model
logger.info(f"Loading model from file {args.modelfile}")
model = UnetModel(input_size=args.input_points_to_use, input_channels=1, output_size=args.input_points_to_use+2, dropout=0.2, step=step)
model.load_model(args.modelfile)

logger.info(f"Loading training dataset {len(train_files)}")
X_train, y_train, _, keys = base.load_dataset(train_files, psbatchsize=1, limitsamplesize=args.limitsamplesize, save=False)
if args.input_points_to_use is not None:
    X_train = X_train[:, :args.input_points_to_use]
logger.info(f"Loading test dataset {len(test_files)}")
X_test, y_test, _, keys = base.load_dataset(test_files, psbatchsize=1, limitsamplesize=args.limitsamplesize, save=False)
if args.input_points_to_use is not None:
    X_test = X_test[:, :args.input_points_to_use]

# Predict on training data
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
train_enc3_flattened = model.get_latent_features(X_train_tensor)
logger.info(f"train_enc3_flattened.shape={train_enc3_flattened.shape}")
params_train, latent_train = UnetModel.aggregate_latent_data(y_train, train_enc3_flattened, 10)
# Save the enc3 output to a file
np.savetxt(f"{output_dir}/train_latent_features.csv", latent_train)
logger.info(f"Saved enc3 layer output to {output_dir}/train_latent_features.csv")

# Train XGBoostRegressor on the enc3 output
regressor = XGBRegressor()
regressor.fit(latent_train, params_train)  # Train on the flattened enc3 output

# Predict parameters for the test dataset
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
test_enc3_flattened = model.get_latent_features(X_test_tensor)
params_test, latent_test = UnetModel.aggregate_latent_data(y_test, test_enc3_flattened, 10)

# Save the enc3 output to a file
np.savetxt(f"{output_dir}/test_latent_features.csv", latent_test)
logger.info(f"Saved enc3 layer output to {output_dir}/test_latent_features.csv")
logger.info(f"test_enc3_flattened.shape={latent_test.shape}")

y_pred = regressor.predict(latent_test)  # Predict using the trained regressor
results = np.column_stack((params_test, y_pred))  # Combine y_test and y_pred
np.savetxt(f"{output_dir}/test_predictions.csv", results, delimiter=",",  fmt='%.2f', header="actual_xHI, actual_logfX, pred_xHI, pred_logfX", comments="")
logger.info(f"Saved y_test and y_pred to {output_dir}/test_predictions.csv")

# Calculate R2 score
r2 = r2_score(params_test, y_pred)
logger.info(f"R2 Score for inference: {r2}")
