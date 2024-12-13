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
import sys


import matplotlib.pyplot as plt

import optuna

class UnetModel(nn.Module):
    def __init__(self, input_size, output_size, kernel1, kernel2, dropout):
        super(UnetModel, self).__init__()
        
        kernel_size = kernel1
        # Encoder (unchanged)
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2)
        )
        
        kernel_size = kernel_size//2
        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2)
        )
        
        kernel_size = kernel_size//2
        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2)
        )
        
        kernel_size = kernel_size//2
        self.enc4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2)
        )
        
        # Decoder (modified with output_padding)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(512, 256, 2, stride=2, output_padding=0),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(512, 128, 2, stride=2, output_padding=0),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(256, 64, 2, stride=2, output_padding=1),  # Added output_padding=1
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(128, 32, 2, stride=2, output_padding=0),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final layer
        # Modify the final layer to output the correct shape
        self.final = nn.Sequential(
            nn.Conv1d(32, 1, 1),  # Change output channels to 1
            nn.Flatten()  # Add flatten layer to match target shape
        )

    def forward(self, x):
        # Print shapes for debugging
        print(f"Input shape: {x.shape}")
        x = x.unsqueeze(1)
        print(f"After unsqueeze: {x.shape}")        
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        
        # Decoder with skip connections
        dec1 = self.dec1(enc4)
        dec1 = torch.cat([dec1, enc3], dim=1)
        
        dec2 = self.dec2(dec1)
        dec2 = torch.cat([dec2, enc2], dim=1)
        
        dec3 = self.dec3(dec2)
        dec3 = torch.cat([dec3, enc1], dim=1)
        
        dec4 = self.dec4(dec3)
        
        print(f"Before final: {dec4.shape}")
        out = self.final(dec4)
        print(f"Output shape: {out.shape}")
        return out
        
def validate_filelist(train_files, so_train_files, test_files, so_test_files):
    if len(train_files) != len(so_train_files):
        raise ValueError(f'Mismatch in length of noisy and signalonly training files! {len(train_files)} != {len(so_train_files)}')
    if len(test_files) != len(so_test_files):
        raise ValueError(f'Mismatch in length of noisy and signalonly training files! {len(test_files)} != {len(so_test_files)}')
    indices1 = [0,2,3,4,5,6,8]
    indices2 = [0,2,3,4,5,6,7]
        
    for train_file, so_train_file in zip(train_files, so_train_files):
        train_file_parts = [train_file.split('/')[-1].split('_')[i] for i in indices1]
        so_train_file_parts = [so_train_file.rstrip('.dat').split('/')[-1].split('_')[i] for i in indices2]
        if train_file_parts != so_train_file_parts:
            raise ValueError(f'Mismatch in file name. {train_file_parts} does not match {so_train_file_parts}')
    for test_file, so_test_file in zip(test_files, so_test_files):
        test_file_parts = [test_file.split('/')[-1].split('_')[i] for i in indices1]
        so_test_file_parts = [so_test_file.rstrip('.dat').split('/')[-1].split('_')[i] for i in indices2]
        if test_file_parts != so_test_file_parts:
            raise ValueError(f'Mismatch in file name. {test_file_parts} does not match {so_test_file_parts}')

def convert_to_pytorch_tensors(X, y, y_so, window_size):
    #y_combined = np.hstack(y, y_so)
    # Convert to PyTorch tensors with float32 dtype
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y_so, dtype=torch.float32)
    
    return X_tensor, y_tensor

class CustomLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super(CustomLoss, self).__init__()
        self.alpha = alpha  # Weight for balancing the two loss components
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets):
        # First component: MSE between predictions and targets
        #mse_loss_xHI = self.mse(predictions[:,0], targets[:,0])
        mse_params = self.mse(predictions[:,:2], targets[:,:2])
        mse_los = self.mse(predictions[:,2:], targets[:,2:])
        
        # Combine both losses
        total_loss = self.alpha * mse_params + (1 - self.alpha) * mse_los

        return total_loss
    
def plot_predictions(y_test_so, y_pred, samples=1, showplots=True, saveplots=True):
    plt.rcParams['figure.figsize'] = [15, 9]
    plt.title(f'Reconstructed LoS vs Actual Noiseless LoS')
    for i, test, pred in enumerate(zip(y_test_so[:samples], y_pred[:samples])):
        plt.plot(test, label='Actual')
        plt.plot(pred, label='Reconstructed')
        if i> 10: break
    plt.xlabel('frequency'), 
    plt.ylabel('flux/S147')
    if showplots: plt.show()
    if saveplots: plt.savefig(f"{output_dir}/reconstructed_los.png")
    plt.clf()

def run(X_train, X_test, y_train, y_train_so, y_test, y_test_so, num_epochs, batch_size, lr, kernel1, kernel2, dropout, input_points_to_use, showplots=False, saveplots=True):
    run_description = f"Commandline: {' '.join(sys.argv)}. Parameters: epochs: {num_epochs}, kernel_size: {kernel1}, points: {input_points_to_use}, label={args.label}"
    logger.info(f"Starting new run: {run_description}")
    X_train, y_train = base.scaleXy(X_train, y_train, args)
    X_test, y_test = base.scaleXy(X_test, y_test, args)

    if input_points_to_use is not None:
        X_train = X_train[:, :input_points_to_use]
        y_train_so = y_train_so[:, :input_points_to_use]
        X_test = X_test[:, :input_points_to_use]  
        y_test_so = y_test_so[:, :input_points_to_use]  
    logger.info(f"Starting training. {X_train.shape},{X_test.shape},{y_train.shape},{y_test.shape},{y_train_so.shape},{y_test_so.shape}")

    #kernel2 = calc_odd_half(kernel1)
    # Convert data to PyTorch tensors
    inputs, outputs = convert_to_pytorch_tensors(X_train, y_train, y_train_so, window_size=kernel1)

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

    model = UnetModel(input_size=len(X_train[0]), output_size=len(y_train[0]), kernel1=kernel1, kernel2=kernel2, dropout=dropout)
    logger.info(f"Created model: {model}")
    # Loss function and optimizer
    #criterion = CustomLoss()  # You can adjust alpha as needed
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop    
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for i, (input_batch, output_batch) in enumerate(dataloader):
            # Zero the gradients
            optimizer.zero_grad()
            
            if epoch==0 and i==0: logger.info(f"Shape of input_batch, output_batch: {input_batch.shape}, {output_batch.shape}")

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
    #save_model(model)

    r2 = None
    with torch.no_grad():
        # Test the model
        logger.info("Testing prediction")

        test_input, test_output = convert_to_pytorch_tensors(X_test, y_test, y_test_so, window_size=kernel2)
        logger.info(f"Shape of test_input, test_output: {test_input.shape}, {test_output.shape}")
        y_pred = model(test_input)
        test_loss = criterion(y_pred, test_output)
        logger.info(f'Test Loss: {test_loss.item():.4f}')

        # Calculate R2 scores
        y_pred = y_pred.detach().cpu().numpy()

        r2 = r2_score(y_test_so, y_pred)
        logger.info("R2 Score: " + str(r2))
        # Calculate rmse scores
        rms_scores = mean_squared_error(y_test, y_pred)
        rms_scores_percent = np.sqrt(rms_scores) * 100 / np.mean(y_test, axis=0)
        logger.info("RMS Error: " + str(rms_scores_percent))

        plot_predictions(y_test_so, y_pred)
 
        #y_pred_params = base.unscale_y(y_pred[:2], args)
        #X_test, y_test = base.unscaleXy(X_test, y_test, args)
        #logger.info(f"unscaled test result {X_test.shape} {y_test.shape} {y_pred.shape}")
    
    """
    base.summarize_test_1000(y_pred[:,:2], y_test[:,:2], output_dir=output_dir, showplots=showplots, saveplots=saveplots)
    if args.scale_y1: combined_r2 = r2[2]
    elif args.scale_y2: combined_r2 = r2
    elif args.xhi_only: combined_r2 = r2
    elif args.logfx_only: combined_r2 = r2
    else: combined_r2 = 0.5*(r2[0]+r2[1])
    """
    logger.info(f"Finished run: r2={r2}. {run_description}")
    return r2


def objective(trial):
    # Define hyperparameter search space
    params = {
        'num_epochs': trial.suggest_int('num_epochs', 12, 24),
        'batch_size': 32, #trial.suggest_categorical('batch_size', [16, 32, 48]),
        'learning_rate': 0.002, # trial.suggest_float('learning_rate', 7e-4, 7e-3, log=True), # 0.0019437504084241922, 
        'kernel1': trial.suggest_int('kernel1', 33, 33, step=10),
        'kernel2': trial.suggest_int('kernel2', 33, 33, step=10),
        'dropout': 0.5, #trial.suggest_categorical('dropout', [0.2, 0.3, 0.4, 0.5]),
        'input_points_to_use': 915,#trial.suggest_int('input_points_to_use', 900, 1400),
    }    
    # Run training with the suggested parameters
    try:
        r2 = run(X_train, train_samples, X_noise, X_test, test_samples, y_train, y_test, 
                   num_epochs=params['num_epochs'],
                   batch_size=params['batch_size'],
                   lr=params['learning_rate'],
                   kernel1=params['kernel1'],
                   kernel2=params['kernel2'],
                   dropout=params['dropout'],
                   input_points_to_use=params['input_points_to_use'],
                   showplots=False,
                   saveplots=True)
            
        return r2
    
    except Exception as e:
        logger.error(f"Trial failed with error: {str(e)}")
        return float('-inf')

# main code starts here
parser = base.setup_args_parser()
args = parser.parse_args()
output_dir = base.create_output_dir(args=args)
logger = base.setup_logging(output_dir)

datafiles = base.get_datafile_list(type='noisy', args=args)
so_datafiles = base.get_datafile_list(type='signalonly', args=args)

test_size = 16
if args.maxfiles is not None:
    datafiles = datafiles[:args.maxfiles]
    test_size = 1

# Important to keep the random_state fixed, so that we can reproduce the results.
train_files, test_files = train_test_split(datafiles, test_size=test_size, random_state=42)
so_train_files, so_test_files = train_test_split(so_datafiles, test_size=test_size, random_state=42)
validate_filelist(train_files, so_train_files, test_files, so_test_files)

if args.runmode in ("train_test", "optimize") :
    logger.info(f"Loading train dataset {len(train_files)}")
    X_train, y_train, _ = base.load_dataset(train_files, psbatchsize=1, limitsamplesize=args.limitsamplesize, save=False)
    y_train_so, _, _ = base.load_dataset(so_train_files, psbatchsize=1, limitsamplesize=args.limitsamplesize, save=False)
    logger.info(f"Loaded datasets X_train:{X_train.shape} y_train:{y_train.shape} y_train_so:{y_train_so.shape}")
    if args.filter_train:
        # Filter for xHI between 0.1 and 0.4
        mask = (y_train[:,0] >= 0.1) & (y_train[:,0] <= 0.4)
        X_train = X_train[mask]
        y_train = y_train[mask]
        y_train_so = y_train_so[mask]
        logger.info(f"Filtered train dataset to {len(X_train)} samples with 0.1 <= xHI <= 0.4")
    logger.info(f"Loading test dataset {len(test_files)}")
    X_test, y_test, _ = base.load_dataset(test_files, psbatchsize=args.psbatchsize, limitsamplesize=args.limitsamplesize, save=False)
    y_test_so, _, _ = base.load_dataset(so_train_files, psbatchsize=1, limitsamplesize=args.limitsamplesize, save=False)
    if args.filter_test:
        # Filter for xHI between 0.1 and 0.4
        mask = (y_test[:,0] >= 0.1) & (y_test[:,0] <= 0.4)
        X_test = X_test[mask]
        y_test = y_test[mask]
        y_test_so = y_test_so[mask]
    logger.info(f"Loaded dataset X_test:{X_test.shape} y_test:{y_test.shape} y_test_so:{y_test_so.shape}")

    if args.runmode == "train_test":
        run(X_train, X_test, y_train, y_train_so, y_test, y_test_so, args.epochs, args.trainingbatchsize, lr=0.0019437504084241922, kernel1=269, kernel2=135, dropout=0.5, input_points_to_use=args.input_points_to_use, showplots=args.interactive)

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

