'''
Predict parameters fX and xHI from the 21cm forest data.
'''

import xgboost as xgb
from xgboost import plot_tree

import pandas as pd
import numpy as np
import pickle
import math

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import scipy.fftpack as fftpack
from scipy.stats import binned_statistic

import argparse
import glob
from datetime import datetime

import F21DataLoader as dl
import os
import sys

import logging
import f21_predict_base as base
#from f21_predict_by_stats import calculate_stats_torch

import optuna

def load_dataset_from_pkl():
    # Lists to store combined data
    all_ks = []
    all_ps = []
    all_params = []
    pklfile = "ps-21cm-forest.pkl"
    with open(pklfile, 'rb') as input_file:  # open a text file
        e = pickle.load(input_file)
        logger.info(f"Loading PS from file. keys={e.keys()}")
        all_ks = e["all_ks"]
        all_ps = e["all_ps"]
        all_params = e["all_params"]
        logger.info(f"Loaded PS from file: {pklfile}, shape={all_ps.shape}")

    logger.info(f"sample ks:{all_ks[0]}")
    logger.info(f"sample ps:{all_ps[0]}")
    logger.info(f"sample params:{all_params[0]}")
    
    # Combine all data
    #ps_combined = np.hstack([all_ps[:,:100], ps_std[:,:100]])
    ps_combined = all_ps
    #ps_combined = np.hstack([all_ps[:,:600], ps_std[:,:600]])
    params_combined = np.array(all_params)

    logger.info(f"\nCombined data shape: {ps_combined.shape}")
    logger.info(f"Combined parameters shape: {params_combined.shape}")
    return (ps_combined, params_combined)

def load_dataset(datafiles, psbatchsize, save=False):
    #Input parameters
    #Read LOS data from 21cmFAST 50cMpc box
    if args.maxfiles is not None:
        datafiles = datafiles[:args.maxfiles]
    freq_axis = None
    # Lists to store combined data
    all_ks = []
    all_ps = []
    all_params = []
    # Create processor with desired number of worker threads
    processor = dl.F21DataLoader(max_workers=8, psbatchsize=psbatchsize, limitsamplesize=args.limitsamplesize, ps_bins=None)
        
    # Process all files and get results
    results = processor.process_all_files(datafiles)
        
    # Access results
    all_los = results['los']
    all_ks = results['ks']
    all_ps = results['ps']
    #ps_std = results['ps_std']
    #ps_plus_std = all_ps + ps_std
    #ps_minus_std = all_ps - ps_std
    all_params = results['params']
    #plot_los(all_ps[0], freq_axis)
    logger.info(f"sample ks:{all_ks[0]}")
    logger.info(f"sample ps:{all_ps[0]}")
    #logger.info(f"sample ps_std:{ps_std[0]}")
    logger.info(f"sample params:{all_params[0]}")
    
    if args.runmode == 'train_test' and save and False:
        base.plot_power_spectra(all_ps, all_ks, all_params, output_dir=output_dir, showplots=args.interactive)
        logger.info(f"Saving PS data to file")
        with open('ps-21cm-forest.pkl', 'w+b') as f:  # open a text file
            pickle.dump({"all_ks": all_ks, "all_ps": all_ps, "all_params": all_params}, f)
    # Combine all data
    #ps_combined = np.hstack([all_ps[:,:600], ps_std[:,:600]])
    ps_combined = all_ps
    params_combined = np.array(all_params)

    logger.info(f"\nCombined data shape: {ps_combined.shape}")
    logger.info(f"Combined parameters shape: {params_combined.shape}")

    #stats = f21stats.calculate_stats_torch()

    return (ps_combined, params_combined)

def save_model(model):
    # Save the model architecture and weights
    logger.info(f'Saving model to: {output_dir}/{args.modelfile}')
    model_json = model.save_model(f"{output_dir}/{args.modelfile}")

def  bin_ps_data(X, ps_bins_to_make, ps_bins_to_use):
    if X.shape[1] <  ps_bins_to_make:
        ps_bins_to_make = X.shape[1]

    num_bins = ps_bins_to_make*ps_bins_to_use//100

    if ps_bins_to_make < X.shape[1]:
        fake_ks = range(X.shape[1])
        X_binned = []
        for x in X:
            ps, _, _ = binned_statistic(fake_ks, x, statistic='mean', bins=num_bins)
            X_binned.append(ps)
        X_binned = np.array(X_binned)
    else:
        X_binned = X
    return X_binned[:,:ps_bins_to_use]


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
    if args.logscale_X: X = np.log(np.clip(X, 1e-20, None))
    return X, y

def unscaleXy(X, y):
    # Undo what we did in scaleXy function
    if args.scale_y: 
        xHI = y[:, 0].reshape(len(y), 1)
        fx = 5.0*(y[:,1] - 0.8).reshape(len(y), 1)
        y = np.hstack((xHI, fx))
    elif args.scale_y0: y[:,0] = y[:,0]/5.0
    elif args.scale_y1:
        xHI = y[:, 0].reshape(len(y), 1)
        fx = 5.0*(1 - y[:,1] - 0.8)
        y = np.hstack((xHI, fx))
    elif args.scale_y2:
        xHI = y[:, 0].reshape(len(y), 1)
        fx = 5.0*(1 - y[:,1] - 0.8).reshape(len(y), 1)
        y = np.hstack((xHI, fx))
                
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
    return y

class ModelTester:
    def __init__(self, model, X_noise, ps_bins_to_make, ps_bins_to_use):
        self.model = model
        self.X_noise = X_noise
        self.ps_bins_to_make = ps_bins_to_make
        self.ps_bins_to_use = ps_bins_to_use
    
    def test(self, los, X_test, y_test, silent=False):
        #if y_test == [-1.00,0.25]: base.plot_single_power_spectrum(X_test, showplots=False, label="Unbinned_PS_with_noise")
        if not silent: logger.info(f"Binning PS data: original_size={X_test.shape[1]}, ps_bins_to_make={self.ps_bins_to_make}, num bins to use={self.ps_bins_to_use}")
        X_test = bin_ps_data(X_test, self.ps_bins_to_make, self.ps_bins_to_use)
        if not silent: logger.info(f"Testing dataset: X:{X_test.shape} y:{y_test.shape}")
        #if y_test == [-1.00,0.25]: base.plot_single_power_spectrum(X_test, showplots=False, label="Binned_PS_with_noise")


        if args.subtractnoise:
            #logger.info(f"Subtracting noise from test data. Shapes: {X_test.shape} {X_noise.shape}")
            X_test -= self.X_noise[:,:X_test.shape[1]]
            #if y_test == [-1.00,0.25]: base.plot_single_power_spectrum(X_test, showplots=False, label="Unbinned_PS_noise_subtracted")

        if args.filter_test:
            # Filter for xHI between 0.1 and 0.4
            mask = (y_test[:,0] >= 0.1) & (y_test[:,0] <= 0.4)
            X_test = X_test[mask]
            y_test = y_test[mask]
        if not silent: logger.info(f"Before scale y_test: {y_test[:1]}")
        X_test, y_test = scaleXy(X_test, y_test)
        if not silent: logger.info(f"After scale y_test: {y_test[:1]}")

        if not silent: logger.info("Testing prediction")
        y_pred = self.model.predict(X_test)

        if y_pred.ndim==1:
            y_pred = y_pred.reshape(len(y_pred),1)
            if args.scale_y2:
                if not silent: logger.info(f"Prediction vs Test data: \n{np.hstack((y_test[:, 2].reshape(len(y_test),1), y_pred))[:5]}")
                r2 = r2_score(y_test[:, 2], y_pred)
            elif args.xhi_only:
                if not silent: logger.info(f"Prediction vs Test data: \n{np.hstack((y_test[:, 0].reshape(len(y_test),1), y_pred))[:5]}")
                r2 = r2_score(y_test[:, 0], y_pred)
            elif args.logfx_only:
                if not silent: logger.info(f"Prediction vs Test data: \n{np.hstack((y_test[:, 1].reshape(len(y_test),1), y_pred))[:5]}")
                r2 = r2_score(y_test[:, 1], y_pred)
            else:
                r2 = r2_score(y_test, y_pred)
        else:
            if not silent: logger.info(f"Prediction vs Test data: \n{np.hstack((y_pred, y_test))[:5]}")
            # Evaluate the model (on a test set, here we just use the training data for simplicity)
            r2 = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(len(y_pred[0]))]
        if not silent: logger.info("R2 Score: " + str(r2))

        if not silent: logger.info(f"Before unscale y_pred: {y_pred[:1]}")
        y_pred = unscale_y(y_pred)
        if not silent: logger.info(f"After unscale y_pred: {y_pred[:1]}")
        if not silent: logger.info(f"Before unscale y_test: {y_test[:1]}")
        X_test, y_test = unscaleXy(X_test, y_test)
        if not silent: logger.info(f"After unscale y_test: {y_test[:1]}")
        if not silent: logger.info(f"unscaled test result {X_test.shape} {y_test.shape} {y_pred.shape}")

        # Calculate rmse scores
        #rms_scores = [mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(len(y_pred[0]))]
        #rms_scores_percent = np.sqrt(rms_scores) * 100 / np.mean(y_test, axis=0)
        #if not silent: logger.info("RMS Error: " + str(rms_scores_percent))    
        return X_test, y_test, y_pred, r2

def run(X_train, X_test, X_noise, y_train, y_test, 
                    ps_bins_to_make,
                    ps_bins_to_use,
                    model_param1,
                    model_param2,
                    showplots=False,
                    saveplots=True):
    run_description = f"output_dir={output_dir} Commandline: {' '.join(sys.argv)}. Parameters: ps_bins_to_make={ps_bins_to_make}, ps_bins_to_use={ps_bins_to_use}, model_param1={model_param1}, model_param2={model_param2}, label={args.label}"
    logger.info(f"Starting new run: {run_description}")
        
    X_train = bin_ps_data(X_train, ps_bins_to_make, ps_bins_to_use)
    logger.info(f"Before scale train: {y_train[:1]}")
    X_train, y_train = scaleXy(X_train, y_train)
    logger.info(f"After scale train: {y_train[:1]}")
    logger.info(f"Training dataset: X:{X_train.shape} y:{y_train.shape}")

    y_pred = None
    rms_scores = None

    # Train model with different sample sizes

    reg = xgb.XGBRegressor(
            n_estimators=model_param1,
            #learning_rate=0.1,
            max_depth=model_param2,
            random_state=42
        )

    logger.info(f"Fitting regressor: {reg}")
    if args.scale_y2:
        reg.fit(X_train, y_train[:,2])
    elif args.xhi_only:
        reg.fit(X_train, y_train[:,0])
    elif args.logfx_only:
        reg.fit(X_train, y_train[:,1])
    else:
        reg.fit(X_train, y_train)

    X_train, y_train = unscaleXy(X_train, y_train)

    tester = ModelTester(reg, X_noise, ps_bins_to_make, ps_bins_to_use)
    if args.test_multiple:
        all_y_pred, all_y_test = base.test_multiple(tester, test_files)
        r2 = base.summarize_test_1000(all_y_pred, all_y_test, output_dir, showplots=args.interactive, saveplots=True, label="_1000")
        base.save_test_results(all_y_pred, all_y_test, output_dir)
    else:
        X_test, y_test, y_pred, r2 = tester.test(None, X_test, y_test)
        base.summarize_test_1000(y_pred, y_test, output_dir=output_dir, showplots=showplots, saveplots=saveplots)
        base.save_test_results(y_pred, y_test, output_dir)

    
    if args.scale_y1: combined_r2 = r2[2]
    elif args.scale_y2: combined_r2 = r2
    elif args.xhi_only: combined_r2 = r2
    elif args.logfx_only: combined_r2 = r2
    else: combined_r2 = 0.5*(r2[0]+r2[1])
    
    logger.info(f"Finished run: score={combined_r2}, r2={r2}. {run_description}")

    return combined_r2, tester

def objective(trial):
    # Define hyperparameter search space
    params = {
        'input_points_to_use': 2762,#trial.suggest_int('input_points_to_use', 1800, 2762),
        'model_param1': trial.suggest_int('model_param1', 80, 150, step=5), # num XGB trees
        'model_param2': trial.suggest_int('model_param2', 3, 6), # xgb tree depth
        'ps_bins_to_make': trial.suggest_int('ps_bins_to_make', 20, 1381, log=True),
        'ps_bins_to_use': trial.suggest_int('ps_bins_to_use', 20, 100, log=True),
    }    
    # Run training with the suggested parameters
    try:
        r2, _ = run(X_train, X_test, X_noise, y_train, y_test, 
                   ps_bins_to_make=params['ps_bins_to_make'],
                   ps_bins_to_use=params['ps_bins_to_use'],
                   model_param1=params['model_param1'],
                   model_param2=params['model_param2'],
                   showplots=False,
                   saveplots=True)
            
        return r2
    
    except Exception as e:
        logger.error(f"Trial failed with error:", exc_info=True)  
        return float('-inf')
    

# main code start here
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
parser.add_argument('-b', '--numsamplebatches', type=int, default=1, help='Number of batches of sample data to use for plotting learning curve by sample size.')
parser.add_argument('--maxfiles', type=int, default=None, help='Max num files to read')
parser.add_argument('--modelfile', type=str, default="xgboost-21cmforest-model.json", help='model file')
parser.add_argument('--psbatchsize', type=int, default=None, help='batching for PS data.')
parser.add_argument('--limitsamplesize', type=int, default=None, help='limit samples from one file to this number.')
parser.add_argument('--interactive', action='store_true', help='run in interactive mode. show plots as modals.')
parser.add_argument('--use_saved_ps_data', action='store_true', help='load PS data from pkl file.')
parser.add_argument('--subtractnoise', action='store_true', help='subtract noise.')
parser.add_argument('--ps_bins_to_make', type=int, default=1381, help='bin the PS into n bins')
parser.add_argument('--ps_bins_to_use', type=int, default=100, help='use the first n bins in the model')
parser.add_argument('--scale_y', action='store_true', help='Scale the y parameters (logfX).')
parser.add_argument('--scale_y0', action='store_true', help='Scale the y parameters (xHI).')
parser.add_argument('--scale_y1', action='store_true', help='Scale logfx and calculate product of logfx with xHI.')
parser.add_argument('--scale_y2', action='store_true', help='Scale logfx and calculate pythogorean sum of logfx with xHI.')
parser.add_argument('--logscale_X', action='store_true', help='Log scale the signal strength.')
parser.add_argument('--model_param1', type=int, default=83, help='')
parser.add_argument('--model_param2', type=int, default=4, help='')
parser.add_argument('--xhi_only', action='store_true', help='calc loss for xhi only')
parser.add_argument('--logfx_only', action='store_true', help='calc loss for logfx only')
parser.add_argument('--filter_test', action='store_true', help='Filter test points in important range of xHI')
parser.add_argument('--filter_train', action='store_true', help='Filter training points in important range of xHI')
parser.add_argument('--label', type=str, default='', help='just a descriptive text for the purpose of the run.')
parser.add_argument('--trials', type=int, default=15, help='Optimization trials')
parser.add_argument('--test_multiple', action='store_true', help='Test 1000 sets of 10 LoS for each test point and plot it')

args = parser.parse_args()
print(args)
if args.ps_bins_to_use < 10 or args.ps_bins_to_use > 100: raise ValueError("--ps_bins_to_use not in acceptable range!")

output_dir = str('output/f21_ps_xgb_%s_%s_t%dh_b%d_%s' % (args.runmode, args.telescope,args.t_int, 1, datetime.now().strftime("%Y%m%d%H%M%S")))

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    print("created " + output_dir)

file_handler = logging.FileHandler(filename=f"{output_dir}/f21_predict_xgb.log")
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(level=logging.INFO, handlers=handlers)
logger = logging.getLogger(__name__)
logger.info(f"Commandline: {' '.join(sys.argv)}")

filepattern = str('%sF21_noisy_21cmFAST_200Mpc_z%.1f_fX%s_xHI%s_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f.dat' % 
               (args.path, args.redshift,args.log_fx, args.xHI, args.telescope, args.spec_res, args.t_int, args.s147, args.alpha_r))
logger.info(f"Loading files with pattern {filepattern}")
datafiles = glob.glob(filepattern)
test_size = 16
if args.maxfiles is not None:
    datafiles = datafiles[:args.maxfiles]
    test_size = 1
logger.info(f"Found {len(datafiles)} files matching pattern")
datafiles = sorted(datafiles)
#train_files, test_files = train_test_split(datafiles, test_size=test_size, random_state=42)
test_points = [[-3.00,0.25],[-2.00,0.25],[-1.00,0.25],[-3.00,0.52],[-2.00,0.52],[-1.00,0.52], [-3.00,0.80],[-2.00,0.80],[-1.00,0.80],[0.00,0.80]]
train_files = []
test_files = []
for f in datafiles:
    is_test_file = False
    for p in test_points:
        if f.find(f"fX{p[0]:.2f}_xHI{p[1]:.2f}") >= 0:
            test_files.append(f)
            is_test_file = True
            break
    if not is_test_file:
        train_files.append(f)

if args.runmode in ("train_test", "optimize") :
    logger.info(f"Loading train dataset {len(train_files)}")
    X_train, y_train = None, None
    if args.use_saved_ps_data:
        X_train, y_train = load_dataset_from_pkl()
    else:
        X_train, y_train = load_dataset(train_files, psbatchsize=args.psbatchsize, save=True)
    logger.info(f"Loaded dataset X_train:{X_train.shape} y:{y_train.shape}")
    if args.filter_train:
        # Filter for xHI between 0.1 and 0.4
        mask = (y_train[:,0] >= 0.1) & (y_train[:,0] <= 0.4)
        X_train = X_train[mask]
        y_train = y_train[mask]
        logger.info(f"Filtered train dataset to {len(X_train)} samples with 0.1 <= xHI <= 0.4")
    X_noise = None
    if args.subtractnoise:
        noisefilepattern = str('%sF21_noiseonly_21cmFAST_200Mpc_z%.1f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f.dat' % 
               (args.path, args.redshift,args.telescope, args.spec_res, args.t_int, args.s147, args.alpha_r))
        logger.info(f"Loading noise files with pattern {noisefilepattern}")
        noisefiles = glob.glob(noisefilepattern)
        X_noise, y_noise = load_dataset(noisefiles, psbatchsize=1000, save=False)
        if X_noise[:,0] == 0: X_noise[:,0] = 1 # Avoid div by zero
        logger.info(f"Loaded noise: {X_noise.shape}")
        logger.info(f"Sample PS before noise subtraction: \n{X_train[:2]}")
        X_train -= X_noise
        logger.info(f"Sample PS after noise subtraction: \n{X_train[:2]}")
    logger.info(f"Loading test dataset {len(test_files)}")
    X_test, y_test = load_dataset(test_files, psbatchsize=args.psbatchsize, save=False)
    logger.info(f"Loaded dataset X_test:{X_test.shape} y:{y_test.shape}")

    if args.runmode == "train_test":
        r2, modeltester = run(X_train, X_test, X_noise, y_train, y_test, ps_bins_to_make=args.ps_bins_to_make, ps_bins_to_use=args.ps_bins_to_use, model_param1=args.model_param1, model_param2=args.model_param2, showplots=args.interactive, saveplots=True)

    elif args.runmode == "optimize":
        # Create study object
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.trials)  # Adjust n_trials as needed

        # Print optimization results
        logger.info("Optuna: Number of finished trials: {}".format(len(study.trials)))
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

elif args.runmode == "test_only": # test_only
    logger.info(f"Loading test dataset {len(test_files)}")
    X_test, y_test = load_dataset(test_files, psbatchsize=args.psbatchsize, save=False)
    if args.psbins_to_use is not None:
        X_test = X_test[:, :args.psbins_to_use]
    X_test, y_test = scaleXy(X_test, y_test)
    logger.info(f"Loaded dataset X_test:{X_test.shape} y:{y_test.shape}")
    model = xgb.XGBRegressor()
    model.load_model(args.modelfile)
    y_pred = model.predict(X_test)
    if args.scale_y: 
        y_pred = unscale_y(y_pred)
        X_test, y_test = unscaleXy(X_test, y_test)
    base.summarize_test(y_pred, y_test, output_dir=output_dir, showplots=args.interactive)
