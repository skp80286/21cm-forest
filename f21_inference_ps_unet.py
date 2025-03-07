'''
Use the U-Net to denoise signal and then calculate PS of denoised signal. Use this PS for inference.
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
from UnetModelNoInputSkipConn import UnetModel

import numpy as np
import sys

import matplotlib.pyplot as plt

import optuna
from xgboost import XGBRegressor

def test_multiple(datafiles, regression_model, latent_model, reps=10000, size=10, input_points_to_use=None):
    logger.info(f"Test_multiple started. {reps} reps x {size} points will be tested for {len(datafiles)} parameter combinations")
    # Create processor with desired number of worker threads
    all_y_test = np.zeros((len(datafiles)*reps, 2))
    all_y_pred = np.zeros((len(datafiles)*reps, 2))
    # Process all files and get results
    for i, f in enumerate(datafiles):
        if i==0: logger.info(f"Working on param combination #{i+1}: {f.split('/')[-1]}")
        los, params, _, _ = base.load_dataset([f], psbatchsize=1, limitsamplesize=None, save=False)
        if args.input_points_to_use is not None:
            los = los[:, :args.input_points_to_use]

        los_tensor = torch.tensor(los, dtype=torch.float32)
        denoised_los_tensor = latent_model.get_denoised_signal(los_tensor)
        logger.info(f"Computing powerspectrum")
        ks, ps = PS1D.get_P_set(denoised_los_tensor.cpu().numpy())
        ks_bin, ps_bin = f21stats.logbin_power_spectrum_by_k(ks, ps)
        #if i == 0: logger.info(f"sample test los_so:{los[:1]}")
        y_pred_for_test_point = []
        for j in range(reps):
            #pick 10 samples
            rdm = np.random.randint(len(los), size=size)
            ps_set = ps_bin[rdm]

            #print(f"latent_features_set.shape={latent_features_set.shape}")
            ps_mean = np.mean(ps_set, axis=0, keepdims=True)
            #print(f"latent_features_mean.shape={latent_features_mean.shape}")
            y_pred = regression_model.predict(ps_mean)  # Predict using the trained regressor
        
            y_pred_for_test_point.append(y_pred)
            all_y_pred[i*reps+j,:] = y_pred
            all_y_test[i*reps+j,:] = params[0]
        if i==0: 
            logger.info(f"Test_multiple: param combination min, max should be the same:{np.min(params, axis=0)}, {np.max(params, axis=0)}")
            
    logger.info(f"Test_multiple: param combination:{params[0]} predicted mean:{np.mean(y_pred_for_test_point, axis=0)}")

    logger.info(f"Test_multiple completed. actual shape {all_y_test.shape} predicted shape {all_y_pred.shape}")
    
    base.calc_squared_error(all_y_pred, all_y_test)

    r2_means = base.summarize_test_1000(all_y_pred, all_y_test, output_dir, showplots=args.interactive, saveplots=True, label="_1000")

    r2 = np.mean(r2_means)
    base.save_test_results(all_y_pred, all_y_test, output_dir)

    return r2

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
if args.maxfiles is not None: datafiles = datafiles[:args.maxfiles]

#test_points = [[-3.00,0.11],[-2.00,0.11],[-1.00,0.11],[-3.00,0.25],[-2.00,0.25],[-1.00,0.25],[-3.00,0.52],[-2.00,0.52],[-1.00,0.52], [-3.00,0.80],[-2.00,0.80],[-1.00,0.80]]#,[0.00,0.80]]
test_points = [[-3,0.11], [-3,0.80], [-1,0.11], [-1,0.80], [-2,0.52]]

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

# Predict on training data
logger.info(f"Denoising training signal")
los_tensor = torch.tensor(X_train, dtype=torch.float32)
denoised_los_tensor = model.get_denoised_signal(los_tensor)
logger.info(f"Computing powerspectrum of denoised signal")
ks, ps = PS1D.get_P_set(denoised_los_tensor.cpu().numpy())
logger.info(f"Binning powerspectrum")
ks_bin, ps_bin = f21stats.logbin_power_spectrum_by_k(ks, ps)
logger.info(f"Training set PS shape={ps_bin.shape}")

params_train, ps_train = f21stats.aggregate_f21_data(y_train, ps_bin, 10)
logger.info(f"Saving powerspectrum to {output_dir}")
np.savetxt(f"{output_dir}/train_denoised_ps.csv", ps_train)
np.savetxt(f"{output_dir}/train_denoised_ks.csv", ks_bin[0])

# Save the enc3 output to a file

# Train XGBoostRegressor on the enc3 output
regressor = XGBRegressor(random_state=42)
logger.info(f"Fitting regressor model {regressor}")
regressor.fit(ps_train, params_train)  # Train on the flattened enc3 output

# Predict parameters for the test dataset
r2 = test_multiple(test_files, regression_model=regressor, latent_model=model, input_points_to_use=args.input_points_to_use)

# Calculate R2 score
logger.info(f"R2 Score for 10k inference: {r2}")
