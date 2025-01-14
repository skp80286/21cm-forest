'''
Predict parameters fX and xHI from the 21cm forest data using bispectrum.
'''

import xgboost as xgb
from xgboost import plot_tree

import torch
import torch.nn as nn

import numpy as np
import pickle
import math

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler  

import scipy.fftpack as fftpack

import argparse
import glob
from datetime import datetime

import F21DataLoader as dl
import F21Stats as f21stats
import os
import sys

import logging
import f21_predict_base as base
import Scaling
from F21NNRegressor import NNRegressor
from F21BayesianRegressor import BayesianRegressor

#from f21_predict_by_stats import calculate_stats_torch

import optuna
from sklearn.linear_model import LinearRegression


def load_dataset(datafiles, psbatchsize, limitsamplesize, save=False, input_points_to_use=None):
    #Input parameters
    #Read LOS data from 21cmFAST 50cMpc box
    if args.maxfiles is not None:
        datafiles = datafiles[:args.maxfiles]

    # Create processor with desired number of worker threads
    processor = dl.F21DataLoader(max_workers=16, psbatchsize=psbatchsize, limitsamplesize=limitsamplesize, skip_ps=(not args.includestats), ps_bins=args.bispec_bins, skip_stats=(not args.includestats), use_bispectrum=True, input_points_to_use=input_points_to_use, perc_bins_to_use=args.perc_ps_bins_to_use)
        
    # Process all files and get results
    results = processor.process_all_files(datafiles)
        
    # Access results
    all_los = results['los']
    all_ks = results['ks']
    all_ps = results['ps']
    all_params = results['params']
    all_bispec = results['bispectrum']
    all_k_bispec = results['k_bispec']
    all_stats = results['stats']
    logger.info(f"sample ps:{all_ps[0]}")
    logger.info(f"sample ks:{all_ks[0]}")
    logger.info(f"sample stats:{all_stats[0]}")
    logger.info(f"sample bispectrum:{all_bispec[0]}")
    logger.info(f"sample k_bispec:{all_k_bispec[0]}")
    logger.info(f"sample params:{all_params[0]}")
    
    k_bispec = all_k_bispec[0]
    logger.info(f"ps shape:{all_ps.shape}")
    logger.info(f"ks shape:{all_ks.shape}")
    logger.info(f"stats shape:{all_stats.shape}")
    logger.info(f"k_bispec shape: {k_bispec.shape}")
    logger.info(f"Combined bispec shape: {all_bispec.shape}")
    logger.info(f"Combined parameters shape: {all_params.shape}")

    return (all_ks, all_ps, k_bispec, all_bispec, all_stats, all_params)

def save_model(model):
    # Save the model architecture and weights
    logger.info(f'Saving model to: {output_dir}/{args.modelfile}')
    model_json = model.save_model(f"{output_dir}/{args.modelfile}")

class ModelTester:
    def __init__(self, model, X_noise, ks, k_bispec, ps_bins_to_make, perc_ps_bins_to_use, stdscaler):
        self.model = model
        self.X_noise = X_noise
        self.ps_bins_to_make = ps_bins_to_make
        self.perc_ps_bins_to_use = perc_ps_bins_to_use
        self.k_bispec = k_bispec
        self.ks = ks
        self.stdscaler = stdscaler
    
    def test(self, los, ps_test, stats_test, bispec_test, y_test, los_so, silent=False):
        #if y_test == [-1.00,0.25]: base.plot_single_power_spectrum(X_test, showplots=False, label="Unbinned_PS_with_noise")
        if not silent: logger.info(f"Binning Bispec data: kbispec_size={self.k_bispec.shape}, original_size={bispec_test.shape}, ps_bins_to_make={self.ps_bins_to_make}, num bins to use={self.perc_ps_bins_to_use}")
        if ps_test is not None: ps_test = np.mean(ps_test, axis=0, keepdims=True)
        if stats_test is not None: stats_test = np.mean(stats_test, axis=0, keepdims=True)
        if bispec_test is not None: bispec_test = np.mean(bispec_test, axis=0, keepdims=True)
        y_test = np.mean(y_test, axis=0, keepdims=True)

        if args.use_log_bins:
            _, ps_test = f21stats.logbin_power_spectrum_by_k(self.ks, ps_test)
        elif args.use_linear_bins:
            ps_test = f21stats.bin_ps_data(ps_test, self.ps_bins_to_make, self.perc_ps_bins_to_use)
            #if y_test == [-1.00,0.25]: base.plot_single_power_spectrum(X_test, showplots=False, label="Binned_PS_with_noise")

        if not silent: logger.info(f"Before scale y_test: {y_test[:1]}")
        bispec_test, y_test = scaler.scaleXy(bispec_test, y_test)
        if not silent: logger.info(f"After scale y_test: {y_test[:1]}")

        if args.subtractnoise:
            if not silent: logger.info(f"Subtracting noise from test data. Shapes: {X_test.shape} {self.X_noise.shape}")
            bispec_test -= self.X_noise
            #if y_test == [-1.00,0.25]: base.plot_single_power_spectrum(X_test, showplots=False, label="Unbinned_PS_noise_subtracted")

        if args.filter_test:
            # Filter for xHI between 0.1 and 0.4
            mask = (y_test[:,0] >= 0.1) & (y_test[:,0] <= 0.4)
            bispec_test = bispec_test[mask]
            stats_test = stats_test[mask]
            y_test = y_test[mask]
        #print(f"stats_test.shape={stats_test.shape}")
        if not silent: logger.info(f"Testing dataset: ps:{ps_test.shape} stats:{stats_test.shape} bispec:{bispec_test.shape} y:{y_test.shape}")
        if args.includestats: bispec_test_with_stats = np.hstack((ps_test, stats_test, bispec_test))
        else: bispec_test_with_stats = bispec_test

        bispec_test_with_stats = np.log(np.clip(bispec_test_with_stats, 1e-20, None))
        bispec_test_with_stats = self.stdscaler.transform(bispec_test_with_stats)

        if not silent: logger.info("Testing prediction")
        if not silent: logger.info(f"Sample data before testing y:{y_test[0]}\nX:{bispec_test_with_stats[0]}")
        if args.dump_all_training_data: 
            with open(f"{output_dir}/all_test_data.csv", 'a') as f:
                data_to_save = np.hstack((bispec_test_with_stats, y_test))
                np.savetxt(f, data_to_save, delimiter=",")
                if not silent: logger.info(f"Saved {len(data_to_save)} lines.")
        y_pred = self.model.predict(bispec_test_with_stats)
        if args.model_type == 'bayesian':
            (y_pred, y_pred_var) = y_pred
            logger.info(f'prediction variance={y_pred_var}')

        """
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
        """
        
        r2 = 1.0 # TBD
        if y_pred.ndim > 1 and y_pred.shape[0] > 1:
            if not silent: logger.info(f"Prediction vs Test data: \n{np.hstack((y_pred, y_test))[:5]}")
            # Evaluate the model (on a test set, here we just use the training data for simplicity)
            r2 = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(len(y_pred[0]))]
        
        if not silent: logger.info("R2 Score: " + str(r2))
        if not silent: logger.info(f"Before unscale y_pred: {y_pred[:1]}")
        y_pred = scaler.unscale_y(y_pred)
        if not silent: logger.info(f"After unscale y_pred: {y_pred[:1]}")
        if not silent: logger.info(f"Before unscale y_test: {y_test[:1]}")
        bispec_test, y_test = scaler.unscaleXy(bispec_test, y_test)
        if not silent: logger.info(f"After unscale y_test: {y_test[:1]}")
        if not silent: logger.info(f"unscaled test result {bispec_test.shape} {y_test.shape} {y_pred.shape}")

        # Calculate rmse scores
        #rms_scores = [mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(len(y_pred[0]))]
        #rms_scores_percent = np.sqrt(rms_scores) * 100 / np.mean(y_test, axis=0)
        #if not silent: logger.info("RMS Error: " + str(rms_scores_percent))    
        return bispec_test, y_test, y_pred, r2

def run(ks, ps_train, kbispec, bispec_train, stats_train, ps_test, bispec_test, stats_test, ps_noise, bispec_noise, stats_noise, y_train, y_test, 
                    ps_bins_to_make,
                    perc_ps_bins_to_use,
                    model_param1,
                    model_param2,
                    showplots=False,
                    saveplots=True):
    if not args.use_saved_data:
        run_description = f"output_dir={output_dir} Commandline: {' '.join(sys.argv)}. Parameters: ps_bins_to_make={ps_bins_to_make}, perc_ps_bins_to_use={perc_ps_bins_to_use}, model_param1={model_param1}, model_param2={model_param2}, label={args.label}"
        logger.info(f"Starting new run: {run_description}")
        if args.use_log_bins:
            _, ps_train = f21stats.logbin_power_spectrum_by_k(ks=ks, ps=ps_train, silent=False)
        elif args.use_linear_bins:
            ps_train = f21stats.bin_ps_data(ps_train, ps_bins_to_make, perc_ps_bins_to_use)
        logger.info(f"Before scale train: {y_train[:1]}")
        ps_train, y_train = scaler.scaleXy(ps_train, y_train)
        logger.info(f"After scale train: {y_train[:1]}")
        if ps_noise is not None: 
            if args.use_log_bins:
                _, ps_noise = f21stats.logbin_power_spectrum_by_k(ks=ks, ps=ps_noise, silent=False)
            elif args.use_linear_bins:
                ps_noise = f21stats.bin_ps_data(ps_noise, ps_bins_to_make, perc_ps_bins_to_use)
            ps_noise, _ = scaler.scaleXy(ps_noise, np.array([[0.0, 0.0]]))
            if not args.signalonly_training:
                logger.info(f"Sample PS before noise subtraction: \n{ps_train[:2]}")
                ps_train -= ps_noise
                logger.info(f"Sample PS after noise subtraction: \n{ps_train[:2]}")

        #_, X_train = logbin_power_spectrum_by_k(ks, X_train)
        logger.info(f"Training dataset: X:{ps_train.shape} y:{y_train.shape}")

        y_pred = None
        rms_scores = None

        """
        reg = xgb.XGBRegressor(
                n_estimators=model_param1,
                #learning_rate=0.1,
                max_depth=model_param2,
                random_state=42
            )
        """
    if args.includestats:
        X_train_with_stats = np.hstack((ps_train, stats_train, bispec_train))
        logger.info(f"data for training. ps:{ps_train.shape}, stats:{stats_train.shape}, bispec:{bispec_train.shape} ")
    else:
        X_train_with_stats = bispec_train
        logger.info(f"data for training. bispec:{bispec_train.shape} ")

    X_train_with_stats = np.log(np.clip(X_train_with_stats, 1e-20, None))
    # Normalize the training data
    stdscaler = StandardScaler()  # Create a scaler instance
    X_train_with_stats = stdscaler.fit_transform(X_train_with_stats)  # Fit and transform the training data

    logger.info(f"Sample data before fitting y:{y_train[0]}\nX:{X_train_with_stats[0]}")
    # Train model with different sample sizes
    if args.model_type == 'linear':
        reg = LinearRegression()
    elif args.model_type == 'xgb':
        reg = xgb.XGBRegressor(random_state=42)
    elif args.model_type == 'nn':
        reg = NNRegressor(input_size=X_train_with_stats.shape[1])
    elif args.model_type == 'bayesian':
        reg = BayesianRegressor()

    logger.info(f"Fitting regressor: {reg}")
    if args.dump_all_training_data: 
        np.savetxt(f"{output_dir}/all_training_data.csv", np.hstack((X_train_with_stats, y_train)), delimiter=",")
        if kbispec is not None:
            if kbispec.ndim > 1:
                kbispec_to_save = kbispec[0]
            else:
                kbispec_to_save = kbispec
                np.savetxt(f"{output_dir}/kbispec.csv", kbispec_to_save, delimiter=',')
            
        if ks is not None:
            if ks.ndim > 1:
                ks_to_save = ks[0]
            else:
                ks_to_save = ks
                np.savetxt(f"{output_dir}/ks.csv", ks_to_save, delimiter=',')

    if args.scale_y2:
        reg.fit(X_train_with_stats, y_train[:,2])
    elif args.xhi_only:
        reg.fit(X_train_with_stats, y_train[:,0])
    elif args.logfx_only:
        reg.fit(X_train_with_stats, y_train[:,1])
    else:
        reg.fit(X_train_with_stats, y_train)

    logger.info(f"Fitted regressor: {reg}")
    if args.model_type == 'xgb':
        logger.info(f"Booster: {reg.get_booster()}")
        feature_importance = reg.feature_importances_
        save_model(reg)
        np.savetxt(f"{output_dir}/feature_importance.csv", feature_importance, delimiter=',')
        logger.info(f"Feature importance: {feature_importance}")
        for imp_type in ['weight','gain', 'cover', 'total_gain', 'total_cover']:
            logger.info(f"Importance type {imp_type}: {reg.get_booster().get_score(importance_type=imp_type)}")

    if not args.use_saved_data:
        ps_train, y_train = scaler.unscaleXy(ps_train, y_train)

    tester = ModelTester(reg, ps_noise, ks, kbispec, ps_bins_to_make, perc_ps_bins_to_use, stdscaler)
    if args.test_multiple:
        all_y_pred, all_y_test = base.test_multiple(tester, test_files, reps=args.test_reps, skip_stats=(not args.includestats), skip_ps=(not args.includestats), use_bispectrum=args.use_bispectrum, input_points_to_use=args.input_points_to_use, ps_bins=args.bispec_bins, perc_bins_to_use=args.perc_ps_bins_to_use)
        r2 = base.summarize_test_1000(all_y_pred, all_y_test, output_dir, showplots=args.interactive, saveplots=True, label="_1000")
        base.save_test_results(all_y_pred, all_y_test, output_dir)
    else:
        X_test, y_test, y_pred, r2 = tester.test(None, None, stats_test, X_test, y_test, None)
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
        'model_param1': 83, #trial.suggest_int('model_param1', 80, 150, step=5), # num XGB trees
        'model_param2': 4, #trial.suggest_int('model_param2', 3, 6), # xgb tree depth
        'ps_bins_to_make': trial.suggest_int('ps_bins_to_make', 8, 160, step=4),
        'perc_ps_bins_to_use': trial.suggest_int('perc_ps_bins_to_use', 5, 60, step=5),
    }    
    # Run training with the suggested parameters
    try:
        r2, _ = run(ks, ps_train, kbispec, bispec_train, stats_train, ps_test, bispec_test, stats_test, ps_noise, bispec_noise, stats_noise, y_train, y_test, 
                   ps_bins_to_make=params['ps_bins_to_make'],
                   perc_ps_bins_to_use=params['perc_ps_bins_to_use'],
                   model_param1=params['model_param1'],
                   model_param2=params['model_param2'],
                   showplots=False,
                   saveplots=True)
            
        return r2
    
    except Exception as e:
        logger.error(f"Trial failed with error:", exc_info=True)  
        return float('-inf')
    

# main code start here
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.determinisitc=True
torch.backends.cudnn.benchmark=False

print(f"All arguments: {sys.argv}")

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
parser.add_argument('--modelfile', type=str, default="xgb-21cmf-bispec-model.json", help='model file')
parser.add_argument('--psbatchsize', type=int, default=None, help='batching for PS data.')
parser.add_argument('--limitsamplesize', type=int, default=None, help='limit samples from one file to this number.')
parser.add_argument('--interactive', action='store_true', help='run in interactive mode. show plots as modals.')
parser.add_argument('--use_saved_ps_data', action='store_true', help='load PS data from pkl file.')
parser.add_argument('--subtractnoise', action='store_true', help='subtract noise.')
parser.add_argument('--ps_bins_to_make', type=int, default=20, help='bin the PS into n bins')
parser.add_argument('--perc_ps_bins_to_use', type=int, default=10, help='use the first n bins in the model')
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
parser.add_argument('--test_reps', type=int, default=10000, help='Test repetitions for each parameter combination')
parser.add_argument('--includestats', action='store_true', help='Include statistics in the model')
parser.add_argument('--dump_all_training_data', action='store_true', help='Dump all training data for analysis')
parser.add_argument('--use_bispectrum', action='store_true', help='Use bispectrum as stats')
parser.add_argument('--signalonly_training', action='store_true', help='Use signalonly LoS for traning')
parser.add_argument('--use_log_bins', action='store_true', help='Use logarithmically binned Powerspectrum')
parser.add_argument('--use_linear_bins', action='store_true', help='Use linearly binned Powerspectrum')
parser.add_argument('--model_type', type=str, default='xgb', help='xgb, linear, nn or bayesian')
parser.add_argument('--use_saved_data', action='store_true', help='Skip data loading and processing. Use data saved as CSV')
parser.add_argument('--input_points_to_use', type=int, default=2048, help='use the first n points of los. ie truncate the los to first 690 points')
parser.add_argument('--bispec_bins', type=int, default=20, help='use bispec binning')

args = parser.parse_args()
print(args)

if args.perc_ps_bins_to_use < 5 or args.perc_ps_bins_to_use > 100: raise ValueError("--perc_ps_bins_to_use not in acceptable range!")
if args.use_log_bins and args.use_linear_bins: raise ValueError("--use_log_bins and --use_linear_bins are mutually exclusive!")

output_dir = base.create_output_dir(args)
logger = base.setup_logging(output_dir)

#logfx_part = '_fX%s_'
#if args.highlogfx: logfx_part = '_fX%s_'

sofilepattern = str('%sF21_signalonly_21cmFAST_200Mpc_z%.1f_fX%s_xHI%s_%dkHz.dat' % 
               (args.path, args.redshift,args.log_fx, args.xHI, args.spec_res))
sodatafiles = glob.glob(sofilepattern)

filepattern = str('%sF21_noisy_21cmFAST_200Mpc_z%.1f_fX%s_xHI%s_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f.dat' % 
               (args.path, args.redshift,args.log_fx, args.xHI, args.telescope, args.spec_res, args.t_int, args.s147, args.alpha_r))
logger.info(f"Loading files with pattern {filepattern}")
datafiles = glob.glob(filepattern)
test_size = 16
if args.maxfiles is not None:
    sodatafiles = sodatafiles[:args.maxfiles]
    datafiles = datafiles[:args.maxfiles]
    test_size = 1
logger.info(f"Found {len(datafiles)} noisy files and {len(sodatafiles)} signalonly files matching pattern")
if len(datafiles) != len(sodatafiles): raise ValueError("noisy files and signalonly files should be the same!")
sodatafiles = sorted(sodatafiles)
datafiles = sorted(datafiles)

"""
print(datafiles)
sys.exit(0)
"""

#train_files, test_files = train_test_split(datafiles, test_size=test_size, random_state=42)
test_points = [[-3.00,0.11],[-2.00,0.11],[-1.00,0.11],[-3.00,0.25],[-2.00,0.25],[-1.00,0.25],[-3.00,0.52],[-2.00,0.52],[-1.00,0.52], [-3.00,0.80],[-2.00,0.80],[-1.00,0.80]]#,[0.00,0.80]]
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

if args.signalonly_training:
    train_files = []
    for f in sodatafiles:
        is_test_file = False
        for p in test_points:
            if f.find(f"fX{p[0]:.2f}_xHI{p[1]:.2f}") >= 0:
                is_test_file = True
                break
        if not is_test_file:
            train_files.append(f)
scaler = Scaling.Scaler(args)
if args.runmode in ("train_test", "optimize") :
    logger.info(f"Loading train dataset {len(train_files)}")
    ks, ps_train, stats_train, y_train, ps_test, stats_test, y_test = None, None, None, None, None, None, None
    kbispec, bispec_train, bispec_test = None, None, None 
    ps_noise, bispec_noise, stats_noise = None, None, None
    if args.use_saved_data:
        # saved data for 200 samples per file
        # data = pd.read_csv('./saved_output/ps_stats_data_200/all_training_data.csv')
        # data = pd.read_csv('./saved_output/ps_stats_data_200/all_test_data.csv')

        # saved data for 400 samples per file
        training_data = np.loadtxt('./saved_output/bispectrum_data/log_normalized_training_data.csv', delimiter=',')
        #testing_data = np.loadtxt('./saved_output/bispectrum_data/log_normalized_test_data.csv', delimiter=',')
        kbispec = np.loadtxt('./saved_output/bispectrum_data/kbispec.csv', delimiter=',')
        ks = np.loadtxt('./saved_output/bispectrum_data/ks.csv', delimiter=',')

        logger.info(f"Loaded dataset X_train:{training_data.shape} ks:{ks.shape}")
        ps_train = training_data[:,:2]
        stats_train = training_data[:,2:8]
        bispec_train = training_data[:,8:12]
        y_train = training_data[:,-2:]
        """
        ps_test = testing_data[:,:2]
        stats_test = testing_data[:,2:8]
        bispec_test = testing_data[:,8:12]
        y_test = testing_data[:,-2:]
        """
    else:
        ks, ps_train, kbispec, bispec_train, stats_train, y_train = load_dataset(train_files, psbatchsize=args.psbatchsize, limitsamplesize=args.limitsamplesize, save=True, input_points_to_use=args.input_points_to_use)
        if args.filter_train:
            # Filter for xHI between 0.1 and 0.4
            mask = (y_train[:,0] >= 0.1) & (y_train[:,0] <= 0.4)
            ps_train = ps_train[mask]
            y_train = y_train[mask]
            bispec_train = bispec_train[mask]
            stats_train = stats_train[mask]
            logger.info(f"Filtered train dataset to {len(bispec_train)} samples with 0.1 <= xHI <= 0.4")
        if args.subtractnoise:
            noisefilepattern = str('%sF21_noiseonly_21cmFAST_200Mpc_z%.1f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f.dat' % 
                (args.path, args.redshift,args.telescope, args.spec_res, args.t_int, args.s147, args.alpha_r))
            logger.info(f"Loading noise files with pattern {noisefilepattern}")
            noisefiles = glob.glob(noisefilepattern)
            ks, ps_noise, kbispec, bispec_noise, stats_noise, y_noise = load_dataset(noisefiles, psbatchsize=1000, limitsamplesize=1000, save=False)
            #if X_noise[:,0] == 0: X_noise[:,0] = 1 # Avoid div by zero
            logger.info(f"Loaded noise: {ps_noise.shape}")
        logger.info(f"Loading test dataset {len(test_files)}")
        ks, ps_test, kbispec, bispec_test, stats_test, y_test = load_dataset(test_files, psbatchsize=args.psbatchsize, limitsamplesize=args.limitsamplesize, save=False, input_points_to_use=args.input_points_to_use)

    logger.info(f"Loaded train dataset X_train:{ps_train.shape}, {stats_train.shape}, {bispec_train.shape},  y:{y_train.shape}")
    #logger.info(f"Loaded test dataset X_test:{ps_test.shape}, {stats_test.shape}, {bispec_test.shape} y:{y_test.shape}")

    if args.runmode == "train_test":
        r2, modeltester = run(ks, ps_train, kbispec, bispec_train, stats_train, ps_test, bispec_test, stats_test, ps_noise, bispec_noise, stats_noise, y_train, y_test, ps_bins_to_make=args.ps_bins_to_make, perc_ps_bins_to_use=args.perc_ps_bins_to_use, model_param1=args.model_param1, model_param2=args.model_param2, showplots=args.interactive, saveplots=True)

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
    ks, X_test, stats_test, y_test, stats_test = load_dataset(test_files, psbatchsize=args.psbatchsize, limitsamplesize=args.limitsamplesize, save=False, input_points_to_use=args.input_points_to_use)
    if args.psbins_to_use is not None:
        X_test = X_test[:, :args.psbins_to_use]
    X_test, y_test = scaler.scaleXy(X_test, y_test)
    logger.info(f"Loaded dataset X_test:{X_test.shape} y:{y_test.shape}")
    model = xgb.XGBRegressor()
    model.load_model(args.modelfile)
    y_pred = model.predict(np.hstack((X_test, stats_test)))
    if args.scale_y: 
        y_pred = scaler.unscale_y(y_pred)
        X_test, y_test = scaler.unscaleXy(X_test, y_test)
    base.summarize_test(y_pred, y_test, output_dir=output_dir, showplots=args.interactive)
