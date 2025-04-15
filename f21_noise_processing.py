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
import plot_results as pltr

def load_dataset(datafiles):
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
    processor = dl.F21DataLoader(max_workers=8, psbatchsize=args.psbatchsize, limitsamplesize=args.limitsamplesize)
        
    # Process all files and get results
    results = processor.process_all_files(datafiles)
        
    # Access results
    all_ks = results['ks']
    all_ps = results['ps']
    ps_std = results['ps_std']
    ps_plus_std = all_ps + ps_std
    ps_minus_std = all_ps - ps_std
    all_params = results['params']
    #plot_los(all_ps[0], freq_axis)
    logger.info(f"sample ks:{all_ks[0]}")
    logger.info(f"sample ps:{all_ps[0]}")
    logger.info(f"sample ps_std:{ps_std[0]}")
    logger.info(f"sample ps_std:{ps_plus_std[0]}")
    logger.info(f"sample ps_std:{ps_minus_std[0]}")
    logger.info(f"sample params:{all_params[0]}")
    
    """
    if args.runmode == 'train_test':
        base.plot_power_spectra(all_ps, all_ks, all_params, output_dir=output_dir, showplots=args.interactive)
        with open('ps-21cm-forest.pkl', 'w+b') as f:  # open a text file
            pickle.dump({"all_ks": all_ks, "all_ps": all_ps, "all_params": all_params}, f)
    """       
    # Combine all data
    ps_combined = np.hstack([all_ps, ps_std])
    params_combined = np.array(all_params)

    logger.info(f"\nCombined data shape: {ps_combined.shape}")
    logger.info(f"Combined parameters shape: {params_combined.shape}")
    return (ps_combined, params_combined)

def save_model(model):
    # Save the model architecture and weights
    logger.info(f'Saving model to: {output_dir}/{args.modelfile}')
    model_json = model.save_model(f"{output_dir}/{args.modelfile}")

def run(X_train, X_test, y_train, y_test):
    logger.info("Starting training")

    # Split the data into training and testing sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Define a list to store training loss and validation loss
    training_loss = []
    validation_loss = []
    test_loss = []
    r2_scores = []
    
    """
    We can specify sample sizes in ascending order here if we want to see 
    the training and test loss curve trend as sample size increases. By default, 
    we will train for only the full length of the available samples.
    """
    num_samples = len(X_train)
    min_sample_size = num_samples//args.numsamplebatches
    sample_sizes = []
    for i in range(args.numsamplebatches - 1):
        sample_sizes.append((i+1)*min_sample_size)
    sample_sizes.append(num_samples)  
    
    y_pred = None
    history = None
    rms_scores = None
    y_train_subset = None

    # Train model with different sample sizes
    for size in sample_sizes:
        print (f'## Sample size: {size}')
        X_train_subset = X_train[:size]
        y_train_subset = y_train[:size]
        logger.info(f"Training dataset: X:{X_train_subset.shape} y:{y_train_subset.shape}")

        model = xgb.XGBRegressor(
            #n_estimators=1000,
            #learning_rate=0.1,
            #max_depth=50,
            random_state=42
        )

        logger.info("Fitting model")
        history = model.fit(X_train_subset, y_train_subset)

        #logger.info(f"History: {history}")
            
        #training_loss.append(history.best_score)  # Store last training loss for each iteration
        #validation_loss.append(history.history['val_loss'][-1])  
        # Test the model
        logger.info("Testing prediction")
        y_pred = model.predict(X_test)

        # Calculate R2 scores
        r2 = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(2)]
        logger.info("R2 Score: " + str(r2))
        r2_scores.append(50*(r2[0]+r2[1]))
        # Calculate rmse scores
        rms_scores = [mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(2)]
        rms_scores_percent = np.sqrt(rms_scores) * 100 / np.mean(y_test, axis=0)
        logger.info("RMS Error: " + str(rms_scores_percent))
        test_loss.append(0.5*(rms_scores_percent[0]+rms_scores_percent[1]))


    base.summarize_test(y_pred, y_test, output_dir=output_dir, showplots=args.interactive)
    logger.info('Plotting Decision Tree')
    plot_tree(model)
    plt.savefig(f"{output_dir}/xgboost_tree.pdf", format = "pdf", dpi=600) 
    save_model(model)

# main code start here
output_dir = str('output/xgb_%s' % (datetime.now().strftime("%Y%m%d%H%M%S")))
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    print("created " + output_dir)

file_handler = logging.FileHandler(filename=f"{output_dir}/f21_predict_xgb.log")
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]

logging.basicConfig(level=logging.INFO, handlers=handlers)
logger = logging.getLogger(__name__)
logger.info(f"Commandline: {' '.join(sys.argv)}")

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

args = parser.parse_args()

filepattern = str('%sF21_noisy_21cmFAST_200Mpc_z%.1f_fX%s_xHI%s_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f.dat' % 
               (args.path, args.redshift,args.log_fx, args.xHI, args.telescope, args.spec_res, args.t_int, args.s147, args.alpha_r))
logger.info(f"Loading files with pattern {filepattern}")
datafiles = glob.glob(filepattern)
logger.info(f"Found {len(datafiles)} files matching pattern")
datafiles = sorted(datafiles)
train_files, test_files = train_test_split(datafiles, test_size=16, random_state=42)

if args.runmode == "train_test":
    logger.info(f"Loading train dataset {len(train_files)}")
    X_train, y_train = load_dataset(train_files)
    logger.info(f"Loaded dataset X_train:{X_train.shape} y:{y_train.shape}")
    logger.info(f"Loading test dataset {len(test_files)}")
    X_test, y_test = load_dataset(test_files)
    logger.info(f"Loaded dataset X_test:{X_test.shape} y:{y_test.shape}")
    run(X_train, X_test, y_train, y_test)
elif args.runmode == "test_only": # test_only
    logger.info(f"Loading test dataset {len(test_files)}")
    X_test, y_test = load_dataset(test_files)
    logger.info(f"Loaded dataset X_test:{X_test.shape} y:{y_test.shape}")
    model = xgb.XGBRegressor()
    model.load_model(args.modelfile)
    y_pred = model.predict(X_test)
    base.summarize_test(y_pred, y_test, output_dir=output_dir, showplots=args.interactive)
