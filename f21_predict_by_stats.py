'''
Predict parameters fX and xHI from the 21cm forest data using Stats (skewness).
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from xgboost import plot_tree

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import argparse
import glob
from datetime import datetime

import F21DataLoader as dl
import f21_predict_base as base
import numpy as np
import os
import sys
import logging
import pickle

import optuna
from optuna.trial import TrialState
from sklearn import linear_model
from scipy import stats
from typing import Tuple

import math

def load_dataset_from_pkl():
    # Lists to store combined data
    all_los = []
    all_params = []
    pklfile = "los-21cm-forest.pkl"
    with open(pklfile, 'rb') as input_file:  # open a text file
        e = pickle.load(input_file)
        logger.info(f"Loading PS from file. keys={e.keys()}")
        all_los = e["all_los"]
        all_params = e["all_params"]
        logger.info(f"Loaded LoS from file: {pklfile}, shape={all_los.shape}")

    logger.info(f"sample ps:{all_los[0]}")
    logger.info(f"sample params:{all_params[0]}")

    return (all_los, all_params, [])

def load_dataset(datafiles, psbatchsize, limitsamplesize, save=False):
    #Input parameters
    #Read LOS data from 21cmFAST 50cMpc box
    if args.maxfiles is not None:
        datafiles = datafiles[:args.maxfiles]
    freq_axis = None
    # Lists to store combined data
    all_ks = []
    all_F21 = []
    all_params = []
    # Create processor with desired number of worker threads
    processor = dl.F21DataLoader(max_workers=8, psbatchsize=psbatchsize, limitsamplesize=limitsamplesize, skip_ps=True)

    # Process all files and get results
    results = processor.process_all_files(datafiles)
    logger.info(f"Finished data loading.")
    # Access results
    all_los = results['los']
    los_samples = results['los_samples']
    all_params = results['params']
    logger.info(f"sample los:{all_los[0]}")
    logger.info(f"sample params:{all_params[0]}")
    
    # Combine all data
    logger.info(f"\nCombined data shape: {all_los.shape}")
    logger.info(f"Combined parameters shape: {all_params.shape}")
        
    if args.runmode == 'train_test' and save:
        logger.info(f"Saving LoS data to file")
        with open('los-21cm-forest.pkl', 'w+b') as f:  # open a text file
            pickle.dump({"all_los": all_los, "all_params": all_params}, f)
            
    return (all_los, all_params, los_samples)

def load_noise():
    X_noise = None
    if args.use_noise_channel:
        noisefilepattern = str('%sF21_noiseonly_21cmFAST_200Mpc_z%.1f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f.dat' % 
               (args.path, args.redshift,args.telescope, args.spec_res, args.t_int, args.s147, args.alpha_r))
        logger.info(f"Loading noise files with pattern {noisefilepattern}")
        noisefiles = glob.glob(noisefilepattern)
        X_noise, _, _ = load_dataset(noisefiles, psbatchsize=1000, limitsamplesize=1000, save=False)

    return X_noise
    
def save_model(model):
    # Save the model architecture and weights
    torch_filename = output_dir +"/f21_predict_cnn_torch.pth"
    logger.info(f'Saving model to: {torch_filename}')
    torch.save(model, torch_filename)


def load_model():
    logger.info(f'Loading model from: {args.modelfile}')
    loaded_model = torch.load(args.modelfile)
    loaded_model.eval()  # Set to evaluation mode
    logger.info("Entire model loaded!")
    return loaded_model

def calc_odd_half(n):
    h = n//2
    return  h + 1 if h % 2 == 0 else h

def calc_odd_thirds(n):
    t1 = 2*n//3
    t1 =  t1 + 1 if t1 % 2 == 0 else t1
    t2 = n//3
    t2 =  t2 + 1 if t2 % 2 == 0 else t2
    return t1, t2

def calculate_stats_torch(X, y, kernel_size=33):
    stat_calc = []

    for i,x in enumerate(X):
        tensor_1d = torch.tensor(x)
         # Pad the tensor if length is not divisible by 3
        padding_needed = kernel_size - len(tensor_1d) % kernel_size
        if padding_needed > 0:
            tensor_1d = torch.nn.functional.pad(tensor_1d, (0, padding_needed))
    
        tensor_2d = tensor_1d.view(-1,kernel_size)
        means = torch.mean(tensor_2d, dim=1)
        std = torch.std(tensor_2d, dim=1, unbiased=False)

        centered_x = tensor_2d - means.unsqueeze(1)
        skewness = torch.mean((centered_x / (std.unsqueeze(1) + 1e-8)) ** 3, dim=1)

        mean_skew = torch.mean(skewness)
        std_skew = torch.std(skewness, unbiased=False)
        
        centered_skew = skewness - mean_skew
        skew2 = torch.mean((centered_skew / (std_skew.unsqueeze(0) + 1e-8)) ** 3)
                
        min_skew = torch.min(skewness)

        row = [mean_skew.item(), std_skew.item(), skew2.item(), min_skew.item()]
        stat_calc.append(row)

        if i < 5: logger.info(f'Mean, stdev. skew2 of skewness for xHI={y[i, 0]} logfx={y[i, 1]}, kernel_size={kernel_size} = {row}')
    return np.array(stat_calc)


def calculate_stats(X, y, kernel_size=33):
    stat_calc = []

    for i, row in enumerate(X):
        pieces = np.array_split(row, len(row)//kernel_size)
        # Calculate statistics for each piece
        skewness = [stats.skew(p) for p in pieces]
        mean_skew = np.mean(skewness)
        std_skew = np.std(skewness)
        min_skew = np.min(skewness)
        skew2 = stats.skew(skewness)
        if i < 5: 
            logger.info(f'Mean, stdev. skew2 of skewness for xHI={y[i, 0]} logfx={y[i, 1]}, kernel_size={kernel_size} = {mean_skew}, {std_skew}, {skew2}, {min_skew}')
        stat_calc.append([mean_skew, std_skew, skew2, min_skew])

    return np.array(stat_calc)

def run(X_train, train_samples, X_noise, X_test, test_samples, y_train, y_test, num_epochs, kernel_size=33, input_points_to_use=2762, showplots=False, saveplots=True):
    run_description = f"Commandline: {' '.join(sys.argv)}\nParameters: epochs: {num_epochs}, kernel_size: {kernel_size}, points: {input_points_to_use}, label={args.label}"
    logger.info(f"Starting new run: {run_description}")
    X_train, y_train = scaleXy(X_train, y_train)
    X_test, y_test = scaleXy(X_test, y_test)

    if input_points_to_use is not None:
        X_train = X_train[:, :input_points_to_use]
        if train_samples is not None: train_samples = train_samples[:,:, :input_points_to_use]
        if X_noise is not None: X_noise = X_noise[:, :input_points_to_use]
        X_test = X_test[:, :input_points_to_use]  
        if test_samples is not None: test_samples = test_samples[:,:, :input_points_to_use]
    
    X_train = calculate_stats_torch(X_train, y_train, kernel_size)
    
    logger.info(f"Starting training. {X_train.shape},{X_test.shape},{y_train.shape},{y_test.shape}")

    if args.regressor == 'linear':
        reg = linear_model.ElasticNet(alpha=0.1, max_iter=5000, fit_intercept=True)
    elif args.regressor == 'xgb':
        reg = xgb.XGBRegressor(
            #n_estimators=1000,
            #learning_rate=0.1,
            #max_depth=50,
            random_state=42
        )
                    #linear_model.QuantileRegressor(quantile=0.9, alpha=0.5) ]:
    logger.info(f"Fitting regressor: {reg}")
    if args.scale_y2:
        reg.fit(X_train, y_train[:,2])
    elif args.xhi_only:
        reg.fit(X_train, y_train[:,0])
    elif args.logfx_only:
        reg.fit(X_train, y_train[:,1])
    else:
        reg.fit(X_train, y_train)

    logger.info(f"Testing")
    X_test = calculate_stats_torch(X_test, y_test, kernel_size)
    #score = reg.score(X_test, y_test)
    #logger.info(f"Test score={score}, intercept={reg.intercept_}, coefficients=\n{reg.coef_}\n")
    y_pred = reg.predict(X_test)
    print(f"y_pred:\n{y_pred[:5]}")

    if y_pred.ndim==1:
        y_pred = y_pred.reshape(len(y_pred),1)
        if args.scale_y2:
            logger.info(f"Prediction vs Test data: \n{np.hstack((y_test[:, 2].reshape(len(y_test),1), y_pred))[:5]}")
            r2 = r2_score(y_test[:, 2], y_pred)
        elif args.xhi_only:
            logger.info(f"Prediction vs Test data: \n{np.hstack((y_test[:, 0].reshape(len(y_test),1), y_pred))[:5]}")
            r2 = r2_score(y_test[:, 0], y_pred)
        elif args.logfx_only:
            logger.info(f"Prediction vs Test data: \n{np.hstack((y_test[:, 1].reshape(len(y_test),1), y_pred))[:5]}")
            r2 = r2_score(y_test[:, 1], y_pred)
        else:
            r2 = r2_score(y_test, y_pred)
    else:
        logger.info(f"Prediction vs Test data: \n{np.hstack((y_pred, y_test))[:5]}")
        # Evaluate the model (on a test set, here we just use the training data for simplicity)
        r2 = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(len(y_pred[0]))]
    logger.info("R2 Score: " + str(r2))

    y_pred = unscale_y(y_pred)
    X_test, y_test = unscaleXy(X_test, y_test)
    logger.info(f"unscaled test result {X_test.shape} {y_test.shape} {y_pred.shape}")

    # Calculate rmse scores
    rms_scores = [mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(len(y_pred[0]))]
    rms_scores_percent = np.sqrt(rms_scores) * 100 / np.mean(y_test, axis=0)
    logger.info("RMS Error: " + str(rms_scores_percent))    
    
    base.summarize_test_1000(y_pred, y_test, output_dir=output_dir, showplots=showplots, saveplots=saveplots)
    if args.scale_y1: combined_r2 = r2[2]
    elif args.scale_y2: combined_r2 = r2
    elif args.xhi_only: combined_r2 = r2
    elif args.logfx_only: combined_r2 = r2
    else: combined_r2 = 0.5*(r2[0]+r2[1])
    
    logger.info(f"Finished run: score={combined_r2}. {run_description}")
    return combined_r2

def scaleXy(X, y):
    if args.scale_y: 
        xHI = y[:, 0].reshape(len(y), 1)
        scaledfx = (0.8 + y[:,1]/5.0).reshape(len(y), 1)
        y = np.hstack((xHI, scaledfx))
    if args.scale_y0: y[:,0] = y[:,0]*5.0
    if args.scale_y1:
        # we wish to create a single metric representing the expected
        # strength of the signal based on xHI (0 to 1) and logfx (-4 to +1)
        # We know that higher xHI and lower logfx lead to a stronger signal, 
        # First we scale logfx to range of 0 to 1.
        # Then we take a product of xHI and (1 - logfx)
        if args.trials == 1: logger.info(f"Before scaleXy: {y}")
        xHI = y[:, 0].reshape(len(y), 1)
        scaledfx = 1 - (0.8 + y[:,1]/5.0)
        product = np.sqrt(xHI**2 + scaledfx**2).reshape(len(y), 1)
        y = np.hstack((xHI, scaledfx, product))
        if args.trials == 1: logger.info(f"ScaledXy: {y}")
    if args.scale_y2:
        # we wish to create a single metric representing the expected
        # strength of the signal based on xHI (0 to 1) and logfx (-4 to +1)
        # We know that higher xHI and lower logfx lead to a stronger signal, 
        # First we scale logfx to range of 0 to 1.
        # Then we take a product of xHI and (1 - logfx)
        if args.trials == 1: logger.info(f"Before scaleXy: {y}")
        xHI = y[:, 0].reshape(len(y), 1)
        scaledfx = 1 - (0.8 + y[:,1]/5.0).reshape(len(y), 1)
        product = np.sqrt(xHI**2 + scaledfx**2).reshape(len(y), 1)
        y = np.hstack((xHI, scaledfx, product))
        if args.trials == 1: logger.info(f"ScaledXy: {y}")
    if args.logscale_X: X = np.log(X)
    return X, y

def unscaleXy(X, y):
    # Undo what we did in scaleXy function
    if args.scale_y: 
        xHI = y[:, 0].reshape(len(y), 1)
        fx = 5.0*(y[:,1] - 0.8).reshape(len(y), 1)
        y = np.hstack((xHI, fx))
    elif args.scale_y0: y[:,0] = y[:,0]/5.0
    elif args.scale_y1:
        if args.trials == 1: logger.info(f"Before unscaleXy: {y}")
        xHI = y[:, 0].reshape(len(y), 1)
        fx = 5.0*(1 - y[:,1] - 0.8)
        y = np.hstack((xHI, fx))
        if args.trials == 1: logger.info(f"UnscaledXy: {y}")
    elif args.scale_y2:
        if args.trials == 1: logger.info(f"Before unscaleXy: {y}")
        xHI = y[:, 0].reshape(len(y), 1)
        fx = 5.0*(1 - y[:,1] - 0.8).reshape(len(y), 1)
        y = np.hstack((xHI, fx))
        if args.trials == 1: logger.info(f"UnscaledXy: {y}")
                
    elif args.logscale_X: X = np.exp(X)
    return X, y

def unscale_y(y):
    # Undo what we did in the scaleXy function
    if args.scale_y: 
        xHI = y[:, 0].reshape(len(y), 1)
        fx = 5.0*(y[:,1] - 0.8).reshape(len(y), 1)
        y = np.hstack((xHI, fx))
    elif args.scale_y0: y[:,0] = y[:,0]/5.0
    elif args.scale_y1:
        # calculate fx using product and xHI 
        if args.trials == 1: logger.info(f"Before unscale_y: {y}")
        xHI = np.sqrt(y[:,2]**2 - y[:,1]**2)
        fx = 5.0*(1 - y[:,1] - 0.8)
        if args.trials == 1: logger.info(f"Unscaled_y: {y}")
        y = np.hstack((xHI, fx))
    elif args.scale_y2:
        # calculate fx using product and xHI 
        if args.trials == 1: logger.info(f"Before unscale_y: {y}")
        xHI = np.sqrt(0.5*y**2).reshape((len(y), 1))
        fx = 5.0*(1 - xHI - 0.8)
        y = np.hstack((xHI, fx))
        if args.trials == 1: logger.info(f"Unscaled_y: {y}")
    elif args.xhi_only:
        # calculate fx using xHI 
        if args.trials == 1: logger.info(f"Before unscale_y: {y}")
        xHI = y
        fx = 5.0*(1 - xHI - 0.8)
        y = np.hstack((xHI, fx))
        if args.trials == 1: logger.info(f"Unscaled_y: {y}")
    elif args.logfx_only:
        # calculate fx using xHI 
        if args.trials == 1: logger.info(f"Before unscale_y: {y}")
        fx = y
        xHI = 1 - 0.8 - fx/5.0
        y = np.hstack((xHI, fx))
        if args.trials == 1: logger.info(f"Unscaled_y: {y}")
    return y

def squared_log(predictions: np.ndarray,
                targets: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    # First component: MSE between predictions and targets
    #mse_loss_xHI = self.mse(predictions[:,0], targets[:,0])
    mse_fx = mean_squared_error(targets[:,1], predictions[:,1])
    mse_product = mean_squared_error(predictions[:,2], targets[:,2])
        
    # Combine both losses
    total_loss = args.alpha * mse_product + (1 - args.alpha) * mse_fx

def objective(trial):
    # Define hyperparameter search space
    params = {
        'num_epochs': 1, #trial.suggest_int('num_epochs', 12, 24),
        'kernel_size': trial.suggest_int('kernel_size', 17, 49),
        'input_points_to_use': trial.suggest_int('input_points_to_use', 800, 2762),
    }    
    # Run training with the suggested parameters
    try:
        r2 = run(X_train, train_samples, X_noise, X_test, test_samples, y_train, y_test, 
                   num_epochs=params['num_epochs'],
                   kernel_size=params['kernel_size'],
                   input_points_to_use=params['input_points_to_use'],
                   showplots=False,
                   saveplots=True)
            
        return r2
    
    except Exception as e:
        logger.error(f"Trial failed with error: {str(e)}")
        return float('-inf')
    

# main code start here
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.determinisitc=True
torch.backends.cudnn.benchmark=False

parser = argparse.ArgumentParser(description='Predict reionization parameters from 21cm forest')
parser.add_argument('-p', '--path', type=str, default='../data/21cmFAST_los/F21_noisy/', help='filepath')
parser.add_argument('-z', '--redshift', type=float, default=6, help='redshift')
parser.add_argument('-d', '--dvH', type=float, default=0.0, help='rebinning width in km/s')
parser.add_argument('-r', '--spec_res', type=int, default=8, help='spectral resolution of telescope (i.e. frequency channel width) in kHz')
parser.add_argument('-t', '--telescope', type=str, default='uGMRT', help='telescope')
parser.add_argument('-s', '--s147', type=float, default=64.2, help='intrinsic flux of QSO at 147Hz in mJy')
parser.add_argument('-a', '--alpha_r', type=float, default=-0.44, help='radio spectral index of QSO')
parser.add_argument('-i', '--t_int', type=float, default=500, help='integration time of obsevation in hours')
parser.add_argument('-f', '--log_fx', type=str, default='*', help='log10(f_X)')
parser.add_argument('-x', '--xHI', type=str, default='*', help='mean neutral hydrogen fraction')
parser.add_argument('-n', '--nlos', type=int, default=1000, help='num lines of sight')
parser.add_argument('-m', '--runmode', type=str, default='train_test', help='one of train_test, test_only, grid_search, plot_only')
parser.add_argument('--regressor', type=str, default='linear', help='one of linear, xgb')
parser.add_argument('-b', '--numsamplebatches', type=int, default=1, help='Number of batches of sample data to use for plotting learning curve by sample size.')
parser.add_argument('--maxfiles', type=int, default=None, help='Max num files to read')
parser.add_argument('--modelfile', type=str, default="output/cnn-torch-21cmforest-model.pth", help='model file')
parser.add_argument('--limitsamplesize', type=int, default=20, help='limit samples from one file to this number.')
parser.add_argument('--interactive', action='store_true', help='run in interactive mode. show plots as modals.')
parser.add_argument('--use_saved_los_data', action='store_true', help='load LoS data from pkl file.')
parser.add_argument('--epochs', type=int, default=15, help='Number of epoch of training.')
parser.add_argument('--trainingbatchsize', type=int, default=32, help='Size of batch for training.')
parser.add_argument('--input_points_to_use', type=int, default=2762, help='use the first n points of los. ie truncate the los to first 690 points')
parser.add_argument('--scale_y', action='store_true', help='Scale the y parameters (logfX).')
parser.add_argument('--scale_y0', action='store_true', help='Scale the y parameters (xHI).')
parser.add_argument('--scale_y1', action='store_true', help='Scale logfx and calculate product of logfx with xHI.')
parser.add_argument('--scale_y2', action='store_true', help='Scale logfx and calculate pythogorean sum of logfx with xHI.')
parser.add_argument('--logscale_X', action='store_true', help='Log scale the signal strength.')
parser.add_argument('--channels', type=int, default=1, help='Use multiple channel inputs for the CNN.')
parser.add_argument('--psbatchsize', type=int, default=None, help='batching for PS data.')
parser.add_argument('--label', type=str, default='', help='just a descriptive text for the purpose of the run.')
parser.add_argument('--trials', type=int, default=15, help='Optimization trials')
parser.add_argument('--use_noise_channel', action='store_true', help='Use noise channel as input.')
parser.add_argument('--xhi_only', action='store_true', help='calc loss for xhi only')
parser.add_argument('--logfx_only', action='store_true', help='calc loss for logfx only')

args = parser.parse_args()
output_dir = str('output/f21_pred_stats_%s_%s_t%dh_b%d_%s' % (args.runmode, args.telescope,args.t_int, 1, datetime.now().strftime("%Y%m%d%H%M%S")))
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    print("created " + output_dir)

file_handler = logging.FileHandler(filename=f"{output_dir}/f21_predict_stat.log")
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(level=logging.INFO, handlers=handlers)
logger = logging.getLogger(__name__)
logger.info(f"Commandline: {' '.join(sys.argv)}")

filepattern = str('%sF21_noisy_21cmFAST_200Mpc_z%.1f_fX%s_xHI%s_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f.dat' % 
               (args.path, args.redshift,args.log_fx, args.xHI, args.telescope, args.spec_res, args.t_int, args.s147, args.alpha_r))
logger.info(f"Loading files with pattern {filepattern}")
test_size = 16
datafiles = glob.glob(filepattern)
if args.maxfiles is not None:
    datafiles = datafiles[:args.maxfiles]
    test_size = 2
logger.info(f"Found {len(datafiles)} files matching pattern")
datafiles = sorted(datafiles)
train_files, test_files = train_test_split(datafiles, test_size=test_size, random_state=42)

if args.runmode == "train_test":
    logger.info(f"Loading train dataset {len(train_files)}")
    if args.use_saved_los_data:
        X_train, y_train, train_samples = load_dataset_from_pkl()
    else:
        X_train, y_train, train_samples = load_dataset(train_files, psbatchsize=args.psbatchsize, limitsamplesize=args.limitsamplesize, save=False)
    logger.info(f"Loaded dataset X_train:{X_train.shape} y:{y_train.shape}")
    logger.info(f"Loading test dataset {len(test_files)}")
    X_test, y_test, test_samples = load_dataset(test_files, psbatchsize=1, limitsamplesize=10, save=False)
    X_noise = load_noise()
    logger.info(f"Loaded dataset X_test:{X_test.shape} y:{y_test.shape}")
    run(X_train, train_samples, X_noise, X_test, test_samples, y_train, y_test, args.epochs, kernel_size=33, input_points_to_use=args.input_points_to_use, showplots=args.interactive)

elif args.runmode == "test_only": # test_only
    logger.info(f"Loading test dataset {len(test_files)}")
    X_test, y_test, test_samples = load_dataset(test_files, psbatchsize=1, limitsamplesize=10, save=False)
    if args.input_points_to_use is not None:
        X_test = X_test[:, :args.input_points_to_use]
    X_test, y_test = scaleXy(X_test, y_test)
    logger.info(f"Loaded dataset X_test:{X_test.shape} y:{y_test.shape}")
    model = load_model(args.modelfile)
    y_pred = model.predict(X_test)
    y_pred = y_pred.detach().cpu().numpy()
    y_pred = unscale_y(y_pred)
    X_test, y_test = unscaleXy(X_test, y_test)
    base.summarize_test(y_pred[:,:2], y_test[:,:2], output_dir=output_dir, showplots=args.interactive)


elif args.runmode == "optimize":
    logger.info("Starting hyperparameter optimization with Optuna")
    
    # Load data
    if args.use_saved_los_data:
        X_train, y_train, train_samples = load_dataset_from_pkl()
    else:
        X_train, y_train, train_samples = load_dataset(train_files, psbatchsize=args.psbatchsize, limitsamplesize=args.limitsamplesize,save=True)
        
    X_noise = load_noise()

    X_test, y_test, test_samples = load_dataset(test_files, psbatchsize=1, limitsamplesize=10, save=False)

    # Create study object
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)  # Adjust n_trials as needed

    # Print optimization results
    logger.info("Number of finished trials: {}".format(len(study.trials)))
    logger.info("Best trial:")
    trial = study.best_trial

    logger.info("  Value: {}".format(trial.value))
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info("    {}: {}".format(key, value))

    # Save optimization results
    with open(f"{output_dir}/optuna_results.txt", "w") as f:
        f.write("Best parameters:\n")
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nBest R2 score: {trial.value}")
