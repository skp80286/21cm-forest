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
import numpy as np
import os
import sys
import logging
import pickle

import optuna
from optuna.trial import TrialState

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

    return (all_los, all_params)

def load_dataset(datafiles, psbatchsize, save=False):
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
    processor = dl.F21DataLoader(max_workers=8, psbatchsize=psbatchsize, limitsamplesize=args.limitsamplesize, skip_ps=True)

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
    torch_filename = output_dir +"/f21_predict_cnn_torch.pth"
    logger.info(f'Saving model to: {torch_filename}')
    torch.save(model, torch_filename)


def load_model():
    logger.info(f'Loading model from: {args.modelfile}')
    loaded_model = torch.load(args.modelfile)
    loaded_model.eval()  # Set to evaluation mode
    logger.info("Entire model loaded!")
    return loaded_model


class CNNModel(nn.Module):
    def __init__(self, input_size, output_size, channels=1, kernel1=11, kernel2=3, dropout=0.2):
        super(CNNModel, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(channels, 32, kernel_size=kernel1, padding='same'),
            nn.Tanh(),
            nn.Conv1d(32, 32, kernel_size=kernel2, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
            
            # Second conv block
            nn.Conv1d(32, 64, kernel_size=kernel1, padding='same'),
            nn.Tanh(),
            nn.Conv1d(64, 64, kernel_size=kernel2, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),

            # Third conv block
            nn.Conv1d(64, 128, kernel_size=kernel1, padding='same'),
            nn.Tanh(),
            nn.Conv1d(128, 128, kernel_size=kernel2, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
        )
        
        # Calculate the size of flattened features
        self.flatten_size = (((input_size//4)//4)//4)*128
        # Dense layers
        self.dense_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(1024, 128),
            nn.LeakyReLU(),
            nn.Linear(128, output_size)  # Output layer for xHI and logfX
        )
        
    
    def forward(self, x):
        # If input is single channel, add channel dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        # If input already has channels, it will remain unchanged

        # Apply conv layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Apply dense layers
        x = self.dense_layers(x)
        return x
    
#def convert_to_pytorch_tensors(X, y):
class ProductLoss(nn.Module):
    def __init__(self):
        super(ProductLoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets):
        pred_product = predictions[:, 0] * predictions[:, 1]
        target_product = targets[:, 0] * targets[:, 1]
        product_loss = self.mse(pred_product, target_product)
        
        return product_loss
    
class CustomLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.25):
        super(CustomLoss, self).__init__()
        self.alpha = alpha  # Weight for balancing the two loss components
        self.beta = beta  # Weight for balancing the two loss components
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets):
        # First component: MSE between predictions and targets
        mse_loss1 = self.mse(predictions[:,:2], targets[:,:2])
        mse_loss2 = self.mse(predictions[:,2], targets[:,2])
        
        # Second component: MSE between product of predictions and value 3
        pred_product = predictions[:, 0] * predictions[:, 1]
        product_loss = 100*self.mse(pred_product, predictions[:,2])
        
        # Combine both losses
        total_loss = self.alpha * (self.beta * mse_loss1 + (1 - self.beta)*mse_loss2) + (1 - self.alpha) * product_loss

        #logger.info(f"Loss calculation:\n##predictions\n{predictions}\n##targets:\n{targets}\n##loss1\n{mse_loss1}\n##loss2\n{mse_loss2}\n##productloss\n{product_loss}\n##total_loss\n{total_loss}")
        return total_loss

def convert_to_pytorch_tensors(X, y):
    # Create different channel representations based on args.channels
    channels = []
    
    # Channel 1: Original signal (always included)
    channels.append(X)
    
    if args.channels > 1:
        # Channel 2: Log of signal
        channels.append(np.log(np.clip(X, 1e-10, None)))
        
    if args.channels > 2:
        # Channel 3: Gradient of signal
        channels.append(np.gradient(X, axis=1))
        
    if args.channels > 3:
        # Channel 4: Moving average
        window_size = 5
        channels.append(np.array([np.convolve(row, np.ones(window_size)/window_size, mode='same') for row in X]))
    
    # Stack channels along a new axis
    X = np.stack(channels[:args.channels], axis=1)
    
    # Convert to PyTorch tensors with float32 dtype
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    return X_tensor, y_tensor

def run(X_train, X_test, y_train, y_test, num_epochs, batch_size, lr, kernel1, kernel2, dropout, input_points_to_use, showplots=False, saveplots=True):
    if input_points_to_use is not None:
        X_train = X_train[:, :input_points_to_use]
        X_test = X_test[:, :input_points_to_use]  
    logger.info("Starting training")

    # Convert data to PyTorch tensors
    inputs, outputs = convert_to_pytorch_tensors(X_train, y_train)

    logger.info(f"Shape of inouts, outputs: {inputs.shape}, {outputs.shape}")
    # Create DataLoader for batching
    train_dataset = TensorDataset(inputs, outputs)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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

    channels = args.channels
    
    model = CNNModel(input_size=len(X_train[0]), output_size=len(y_train[0]), channels=channels, kernel1=kernel1, kernel2=kernel2, dropout=dropout)
    logger.info(f"Created model: {model}")
    # Loss function and optimizer
    if args.scale_y1:
        criterion = CustomLoss(alpha=0.5)  # You can adjust alpha as needed
    else:
        criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop    
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for i, (input_batch, output_batch) in enumerate(dataloader):
            # Zero the gradients
            optimizer.zero_grad()
            
            if epoch==0 and i==0: logger.info(f"Shape of inout_batch, output_batch: {input_batch.shape}, {output_batch.shape}")

            # Forward pass
            predictions = model(input_batch)
            
            # Compute the loss

            loss = criterion(predictions, output_batch)

            # Backpropagation
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print loss for every epoch
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')

    # Evaluate the model (on a test set, here we just use the training data for simplicity)
    model.eval()  # Set the model to evaluation mode
    save_model(model)
    with torch.no_grad():
        # Test the model
        logger.info("Testing prediction")
        #test_input, test_output = convert_to_pytorch_tensors(X_test, y_test)

        test_input, test_output = convert_to_pytorch_tensors(X_test, y_test)
        logger.info(f"Shape of test_input, test_output: {test_input.shape}, {test_output.shape}")
        y_pred = model(test_input)
        test_loss = criterion(y_pred, test_output)
        logger.info(f'Test Loss: {test_loss.item():.4f}')


        # Calculate R2 scores
        y_pred_np = y_pred.detach().cpu().numpy()

        r2 = [r2_score(y_test[:, i], y_pred_np[:, i]) for i in range(len(y_pred_np[0]))]
        logger.info("R2 Score: " + str(r2))
        # Calculate rmse scores
        rms_scores = [mean_squared_error(y_test[:, i], y_pred_np[:, i]) for i in range(len(y_pred_np[0]))]
        rms_scores_percent = np.sqrt(rms_scores) * 100 / np.mean(y_test, axis=0)
        logger.info("RMS Error: " + str(rms_scores_percent))
 
        y_pred = unscale_y(y_pred)
        X_test, y_test = unscaleXy(X_test, y_test)
        logger.info(f"unscaled test result {X_test.shape} {y_test.shape} {y_pred.shape}")
    
    mean_r2_score = base.summarize_test(y_pred_np[:,:2], y_test[:,:2], output_dir=output_dir, showplots=showplots, saveplots=saveplots)
    return mean_r2_score

def scaleXy(X, y):
    if args.scale_y: y[:,1] = 0.8 + y[:,1]/5.0
    if args.scale_y0: y[:,0] = y[:,0]*5.0
    if args.scale_y1:
        y[:,1] = 0.8 + y[:,1]/5.0
        product = (y[:,0] * y[:,1]).reshape(len(y), 1)
        y = np.hstack((y, product))

    if args.logscale_X: X = np.log(X)
    return X, y

def unscaleXy(X, y):
    if args.scale_y: y[:,1] = 5.0*(y[:,1] - 0.8)
    if args.scale_y0: y[:,0] = y[:,0]/5.0
    if args.scale_y1:
        y[:,1] = 5.0*(y[:,1] - 0.8)
        y = y[:,:2]
    if args.logscale_X: X = np.exp(X)
    return X, y

def unscale_y(y):
    if args.scale_y: 
        y[:,1] = 5.0*(y[:,1] - 0.8)
        return y
    
    if args.scale_y1:
        sqrt = np.sqrt(y[:,2]).reshape((len(y), 1))
        logfx = 5.0*(sqrt - 0.8)
        xHI = sqrt
        return np.hstack((xHI, logfx))
    
    return y

def objective(trial):
    # Define hyperparameter search space
    params = {
        'num_epochs': trial.suggest_int('num_epochs', 10, 50),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256]),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'kernel1': trial.suggest_int('kernel1', 3, 15, step=2),  # odd numbers only
        'kernel2': trial.suggest_int('kernel2', 3, 7, step=2),   # odd numbers only
        'dropout': trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4, 0.5]),
        'input_points_to_use': trial.suggest_int('input_points_to_use', 400, 2762),
    }    
    # Run training with the suggested parameters
    try:
        r2 = run(X_train, X_test, y_train, y_test, 
                   num_epochs=params['num_epochs'],
                   batch_size=params['batch_size'],
                   lr=params['learning_rate'],
                   kernel1=params['kernel1'],
                   kernel2=params['kernel2'],
                   dropout=params['dropout'],
                   input_points_to_use=params['input_points_to_use'],
                   showplots=False,
                   saveplots=False)
            
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
parser.add_argument('-b', '--numsamplebatches', type=int, default=1, help='Number of batches of sample data to use for plotting learning curve by sample size.')
parser.add_argument('--maxfiles', type=int, default=None, help='Max num files to read')
parser.add_argument('--modelfile', type=str, default="output/cnn-torch-21cmforest-model.pth", help='model file')
parser.add_argument('--limitsamplesize', type=int, default=20, help='limit samples from one file to this number.')
parser.add_argument('--interactive', action='store_true', help='run in interactive mode. show plots as modals.')
parser.add_argument('--use_saved_los_data', action='store_true', help='load LoS data from pkl file.')
parser.add_argument('--epochs', type=int, default=10, help='Number of epoch of training.')
parser.add_argument('--trainingbatchsize', type=int, default=64, help='Size of batch for training.')
parser.add_argument('--input_points_to_use', type=int, default=None, help='use the first n points of los. ie truncate the los to first 690 points')
parser.add_argument('--scale_y', action='store_true', help='Scale the y parameters (logfX).')
parser.add_argument('--scale_y0', action='store_true', help='Scale the y parameters (xHI).')
parser.add_argument('--scale_y1', action='store_true', help='Scale logfx and calculate product of logfx with xHI.')
parser.add_argument('--logscale_X', action='store_true', help='Log scale the signal strength.')
parser.add_argument('--channels', type=int, default=1, help='Use multiple channel inputs for the CNN.')
parser.add_argument('--psbatchsize', type=int, default=None, help='batching for PS data.')

args = parser.parse_args()
output_dir = str('output/cnn_torch_%s_%s_t%dh_b%d_%s' % (args.runmode, args.telescope,args.t_int, 1, datetime.now().strftime("%Y%m%d%H%M%S")))
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    print("created " + output_dir)

file_handler = logging.FileHandler(filename=f"{output_dir}/f21_predict_cnn_torch.log")
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
        X_train, y_train = load_dataset(train_files, psbatchsize=args.psbatchsize, save=True)
    X_train, y_train = scaleXy(X_train, y_train)
    logger.info(f"Loaded dataset X_train:{X_train.shape} y:{y_train.shape}")
    logger.info(f"Loading test dataset {len(test_files)}")
    X_test, y_test = load_dataset(test_files, psbatchsize=args.psbatchsize, save=False)
    if args.input_points_to_use is not None:
        X_test = X_test[:, :args.input_points_to_use]    
    X_test, y_test = scaleXy(X_test, y_test)
    logger.info(f"Loaded dataset X_test:{X_test.shape} y:{y_test.shape}")
    run(X_train, X_test, y_train, y_test, args.epochs, args.trainingbatchsize, lr=0.003207368350697034, kernel1=9, kernel2=5, dropout=0.5, input_points_to_use=args.input_points_to_use, showplots=args.interactive)

elif args.runmode == "test_only": # test_only
    logger.info(f"Loading test dataset {len(test_files)}")
    X_test, y_test = load_dataset(test_files, psbatchsize=args.psbatchsize, save=False)
    if args.input_points_to_use is not None:
        X_test = X_test[:, :args.input_points_to_use]
    X_test, y_test = scaleXy(X_test, y_test)
    logger.info(f"Loaded dataset X_test:{X_test.shape} y:{y_test.shape}")
    model = load_model(args.modelfile)
    y_pred = model.predict(X_test)

    y_pred = unscale_y(y_pred)
    X_test, y_test = unscaleXy(X_test, y_test)
    base.summarize_test(y_pred[:,:2], y_test[:,:2], output_dir=output_dir, showplots=args.interactive)


elif args.runmode == "optimize":
    logger.info("Starting hyperparameter optimization with Optuna")
    
    # Load data
    if args.use_saved_los_data:
        X_train, y_train = load_dataset_from_pkl()
    else:
        X_train, y_train = load_dataset(train_files, psbatchsize=args.psbatchsize, save=True)
    
    X_train, y_train = scaleXy(X_train, y_train)
    
    X_test, y_test = load_dataset(test_files, psbatchsize=args.psbatchsize, save=False)
  
    X_test, y_test = scaleXy(X_test, y_test)
    
    # Create study object
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)  # Adjust n_trials as needed

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