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
    torch_filename = output_dir +"/f21_predict_cnn_torch.pth"
    logger.info(f'Saving model to: {torch_filename}')
    torch.save(model, torch_filename)


def load_model():
    logger.info(f'Loading model from: {args.modelfile}')
    loaded_model = torch.load(args.modelfile)
    loaded_model.eval()  # Set to evaluation mode
    logger.info("Entire model loaded!")
    return loaded_model

class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=2):
        super(RNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=1,  # Single feature at each time step
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Dense layers for prediction
        self.dense_layers = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, output_size)  # Output layer for xHI and logfX
        )
    
    def forward(self, x):
        # Reshape input to (batch_size, sequence_length, features)
        x = x.unsqueeze(-1)  # Add feature dimension
        
        # Initialize hidden state and cell state
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Use the output from the last time step
        last_output = lstm_out[:, -1, :]
        
        # Apply dense layers
        output = self.dense_layers(last_output)
        return output
    
def run(X_train, X_test, y_train, y_test):
    logger.info("Starting training")

    
    # Convert data to PyTorch tensors
    inputs = torch.tensor(X_train)
    outputs = torch.tensor(y_train)
    logger.info(f"Shape of inouts, outputs: {inputs.shape}, {outputs.shape}")
    # Create DataLoader for batching
    train_dataset = TensorDataset(inputs, outputs)
    dataloader = DataLoader(train_dataset, batch_size=args.trainingbatchsize, shuffle=True)

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
    model = RNNModel(input_size=2762, output_size=2)
    logger.info(f"Created model: {model}")
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #optimizer = optim.Adagrad(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = args.epochs
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
        test_input = torch.tensor(X_test)  # Replace with actual test data
        test_output = torch.tensor(y_test)
        logger.info(f"Shape of test_input, test_output: {test_input.shape}, {test_output.shape}")
        y_pred = model(test_input)
        test_loss = criterion(y_pred, test_output)
        logger.info(f'Test Loss: {test_loss.item():.4f}')


        # Calculate R2 scores
        y_pred_np = y_pred.detach().cpu().numpy()
        r2 = [r2_score(y_test[:, i], y_pred_np[:, i]) for i in range(2)]
        logger.info("R2 Score: " + str(r2))
        # Calculate rmse scores
        rms_scores = [mean_squared_error(y_test[:, i], y_pred_np[:, i]) for i in range(2)]
        rms_scores_percent = np.sqrt(rms_scores) * 100 / np.mean(y_test, axis=0)
        logger.info("RMS Error: " + str(rms_scores_percent))

        base.summarize_test(y_pred_np, y_test, output_dir=output_dir, showplots=args.interactive)


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
parser.add_argument('--modelfile', type=str, default="output/cnn-torch-21cmforest-model.pth", help='model file')
parser.add_argument('--limitsamplesize', type=int, default=20, help='limit samples from one file to this number.')
parser.add_argument('--interactive', action='store_true', help='run in interactive mode. show plots as modals.')
parser.add_argument('--use_saved_los_data', action='store_true', help='load LoS data from pkl file.')
parser.add_argument('--epochs', type=int, default=10, help='Number of epoch of training.')
parser.add_argument('--trainingbatchsize', type=int, default=64, help='Size of batch for training.')

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
