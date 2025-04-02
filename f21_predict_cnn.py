'''
Predict parameters fX and xHI from the 21cm forest data using CNN.
'''

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import huber

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import argparse
import glob
from datetime import datetime

import F21DataLoader as dl
import f21_predict_base as base
import plot_results as pltr
import numpy as np
import os
import sys
import logging
import pickle

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

    return (all_los, all_params)

def load_dataset(datafiles, save=False):
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
    processor = dl.F21DataLoader(max_workers=8, psbatchsize=1, limitsamplesize=args.limitsamplesize, skip_ps=True)
        
    # Process all files and get results
    results = processor.process_all_files(datafiles)
    logger.info(f"Finished data loading.")
    # Access results
    all_los = results['los']
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
    return (all_los, all_params)

def save_model(model):
    # Save the model architecture and weights
    keras_filename = output_dir +"/f21_predict_cnn.keras"
    logger.info(f'Saving model to: {keras_filename}')
    model_json = model.save(keras_filename)

def load_model():
    logger.info(f'Loading model from: {args.modelfile}')
    return models.load_model(args.modelfile)

def create_cnn_model(input_shape):
    """Create a CNN model for regression"""
    model = models.Sequential([
        layers.Reshape((*input_shape, 1), input_shape=input_shape),
        layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        layers.Conv1D(32, kernel_size=7, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.1),
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        layers.Conv1D(64, kernel_size=7, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),
        layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        layers.Conv1D(128, kernel_size=7, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(128, activation='tanh'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(2)  # Output layer with 2 neurons for xHI and logfX
    ])

    # Compile the model  
    model.compile(loss=huber, optimizer='adam', metrics=['accuracy'])

    # Summary of the model
    model.summary()
    logger.info("######## completed model setup #########")

    return model

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

        # Create and compile the model
        input_shape = (X_train_subset.shape[1],)
        model = create_cnn_model(input_shape)
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    
        logger.info(f"{X_train_subset.shape}")
        X_train_subset = X_train_subset.reshape(len(X_train_subset), X_train_subset.shape[1], 1)
        logger.info(f"Fitting data: {X_train_subset.shape}")
        history = model.fit(
            X_train_subset, y_train_subset,
            epochs=args.epochs,
            batch_size=args.trainingbatchsize,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )

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
    save_model(model)

# main code start here
tf.config.list_physical_devices('GPU')
#print("### GPU Enabled!!!")

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
parser.add_argument('--modelfile', type=str, default="output/cnn-21cmforest-model.json", help='model file')
parser.add_argument('--limitsamplesize', type=int, default=20, help='limit samples from one file to this number.')
parser.add_argument('--interactive', action='store_true', help='run in interactive mode. show plots as modals.')
parser.add_argument('--use_saved_los_data', action='store_true', help='load LoS data from pkl file.')
parser.add_argument('--epochs', type=int, default=10, help='Number of epoch of training.')
parser.add_argument('--trainingbatchsize', type=int, default=64, help='Size of batch for training.')

args = parser.parse_args()
output_dir = str('output/cnn_%s_%s_t%dh_b%d_%s' % (args.runmode, args.telescope,args.t_int, 1, datetime.now().strftime("%Y%m%d%H%M%S")))
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    print("created " + output_dir)

file_handler = logging.FileHandler(filename=f"{output_dir}/f21_predict_cnn.log")
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
    test_size = 1
logger.info(f"Found {len(datafiles)} files matching pattern")
datafiles = sorted(datafiles)
train_files, test_files = train_test_split(datafiles, test_size=test_size, random_state=42)

if args.runmode == "train_test":
    logger.info(f"Loading train dataset {len(train_files)}")
    if args.use_saved_los_data:
        X_train, y_train = load_dataset_from_pkl()
    else:
        X_train, y_train = load_dataset(train_files, save=True)
    logger.info(f"Loaded dataset X_train:{X_train.shape} y:{y_train.shape}")
    logger.info(f"Loading test dataset {len(test_files)}")
    X_test, y_test = load_dataset(test_files, save=False)
    logger.info(f"Loaded dataset X_test:{X_test.shape} y:{y_test.shape}")
    run(X_train, X_test, y_train, y_test)
elif args.runmode == "test_only": # test_only
    logger.info(f"Loading test dataset {len(test_files)}")
    X_test, y_test = load_dataset(test_files, save=False)
    logger.info(f"Loaded dataset X_test:{X_test.shape} y:{y_test.shape}")
    model = load_model(args.modelfile)
    y_pred = model.predict(X_test)
    base.summarize_test(y_pred, y_test, output_dir=output_dir, showplots=args.interactive)
