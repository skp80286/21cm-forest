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
            nn.MaxPool1d(4)
        )
        
        kernel_size = kernel_size//4
        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(4)
        )
        
        kernel_size = kernel_size//4
        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(4)
        )
        
        kernel_size = kernel_size//4
        self.enc4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(4)
        )
        
        # Decoder 
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(512, 256, 4, stride=4, output_padding=0),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(512, 128, 4, stride=4, output_padding=0),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(256, 64, 4, stride=4, output_padding=0), 
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(128, 32, 4, stride=4, output_padding=0),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Dense layers after enc4 for parameters extraction
        self.dense1 = nn.Linear(5120, 256)  # First dense layer
        self.dense2 = nn.Linear(256, 64)  # Second dense layer
        self.dense3 = nn.Linear(64, 2)    # Third dense layer (output 2 values)

        # Final layer
        # Modify the final layer to output the correct shape
        self.final = nn.Sequential(
            nn.Conv1d(32, 1, 1),  # Change output channels to 1
            nn.Flatten()  # Add flatten layer to match target shape
        )

    def forward(self, x):
        # Print shapes for debugging
        #print(f"Input shape: {x.shape}")
        x = x.unsqueeze(1)
        #print(f"After unsqueeze: {x.shape}")        
        # Encoder
        enc1 = self.enc1(x)
        #print(f"After enc1: {enc1.shape}")        
        enc2 = self.enc2(enc1)
        #print(f"After enc2: {enc2.shape}")        
        enc3 = self.enc3(enc2)
        #print(f"After enc3: {enc3.shape}")        
        enc4 = self.enc4(enc3)
        #print(f"After enc4: {enc4.shape}")        
        
        # Pass through the dense layers for parameters extraction
        dense_out = self.dense1(enc4.view(enc4.size(0), -1))  # Flatten for dense layer
        #print(f"After dense1: {dense_out.shape}")        
        dense_out = nn.ReLU()(dense_out)
        dense_out = self.dense2(dense_out)
        #print(f"After dense2: {dense_out.shape}")        
        dense_out = nn.ReLU()(dense_out)
        dense_out = self.dense3(dense_out)
        #print(f"After dense3: {dense_out.shape}")        

        #print(f"Output of parameters extraction network: {dense_out.shape}")

        # Decoder with skip connections
        dec1 = self.dec1(enc4)
        #print(f"After dec1: {dec1.shape}")        
        dec1 = torch.cat([dec1, enc3], dim=1)
        
        dec2 = self.dec2(dec1)
        #print(f"After dec2: {dec2.shape}")        
        dec2 = torch.cat([dec2, enc2], dim=1)
        
        dec3 = self.dec3(dec2)
        #print(f"After dec3: {dec3.shape}")        
        dec3 = torch.cat([dec3, enc1], dim=1)
        
        dec4 = self.dec4(dec3)
 
        #print(f"Before final: {dec4.shape}")
        out = self.final(dec4)

        # Concatenate parameter extraction output with decoder output
        out = torch.cat((dense_out, out), dim=1)  # Concatenate along the feature dimension
        #print(f"Output shape after concatenation: {out.shape}")

        #print(f"Output shape: {out.shape}")
        return out

class ModelTester:
    def __init__(self, model, criterion, input_points_to_use):
        self.model = model
        self.criterion = criterion
        self.input_points_to_use = input_points_to_use
    
    def test(self, los_test, p_test, stats_test, y_test, los_so, silent=False):
        if self.input_points_to_use is not None: 
            los_test = los_test[:, :self.input_points_to_use]
            los_so = los_so[:, :self.input_points_to_use]

        if not silent: logger.info(f"Testing dataset: X:{los_test.shape} y:{y_test.shape}")
        if not silent: logger.info(f"Before scale y_test: {y_test[:1]}")
        los_test, y_test = scaler.scaleXy(los_test, y_test)
        if not silent: logger.info(f"After scale y_test: {y_test[:1]}")

        if not silent: logger.info("Testing prediction")
        if not silent: logger.info(f"Sample data before testing y:{y_test[0]}\nX:{los_test[0]}")

        r2 = None
        with torch.no_grad():
            # Test the model
            if not silent: logger.info("Testing prediction")

            test_input, test_output = convert_to_pytorch_tensors(los_test, y_test, los_so)
            if not silent: logger.info(f"Shape of test_input, test_output: {test_input.shape}, {test_output.shape}")
            y_pred_tensor = self.model(test_input)
            test_loss = self.criterion(y_pred_tensor, test_output)
            if not silent: logger.info(f'Test Loss: {test_loss.item():.4f}')

            # Calculate R2 scores
            y_pred_np = y_pred_tensor.detach().cpu().numpy()
            y_pred = y_pred_np[:,:2]
            y_pred_so = y_pred_np[:,2:]
            r2 = r2_score(y_test, y_pred)
            if not silent: logger.info("R2 Score: " + str(r2))
            # Calculate rmse scores
            rms_scores = mean_squared_error(y_test, y_pred)
            rms_scores_percent = np.sqrt(rms_scores) * 100 / np.mean(y_test, axis=0)
            if not silent: logger.info("RMS Error: " + str(rms_scores_percent))

            if not silent: plot_predictions(los_so, y_pred_so)
    
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
            y_pred = scaler.unscale_y(y_pred)
            if not silent: logger.info(f"After unscale y_pred: {y_pred[:1]}")
            if not silent: logger.info(f"Before unscale y_test: {y_test[:1]}")
            los_test, y_test = scaler.unscaleXy(los_test, y_test)
            if not silent: logger.info(f"After unscale y_test: {y_test[:1]}")
            if not silent: logger.info(f"unscaled test result {los_test.shape} {y_test.shape} {y_pred.shape}")

            # Calculate rmse scores
            #rms_scores = [mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(len(y_pred[0]))]
            #rms_scores_percent = np.sqrt(rms_scores) * 100 / np.mean(y_test, axis=0)
            #if not silent: logger.info("RMS Error: " + str(rms_scores_percent))    
        return los_test, y_test, y_pred, r2

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

def convert_to_pytorch_tensors(X, y, y_so):
    #y_combined = np.hstack(y, y_so)
    # Convert to PyTorch tensors with float32 dtype
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(np.hstack((y, y_so)), dtype=torch.float32)
    
    return X_tensor, y_tensor

class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5):
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
    
def plot_predictions(y_test_so, y_pred_so, samples=1, showplots=True, saveplots=True):
    plt.rcParams['figure.figsize'] = [15, 9]
    plt.title(f'Reconstructed LoS vs Actual Noiseless LoS')
    for i, (test, pred) in enumerate(zip(y_test_so[:samples], y_pred_so[:samples])):
        plt.plot(test, label='Actual')
        plt.plot(pred, label='Reconstructed')
        if i> 2: break
    plt.xlabel('frequency'), 
    plt.ylabel('flux/S147')
    plt.legend()
    if showplots: plt.show()
    if saveplots: plt.savefig(f"{output_dir}/reconstructed_los.png")
    plt.clf()

def run(X_train, X_test, y_train, y_train_so, y_test, y_test_so, num_epochs, batch_size, lr, kernel1, kernel2, dropout, input_points_to_use, showplots=False, saveplots=True):
    run_description = f"Commandline: {' '.join(sys.argv)}. Parameters: epochs: {num_epochs}, batch_size: {batch_size}, lr: {lr}, kernel_sizes: [{kernel1}, {kernel2}], dropout: {dropout}, points: {input_points_to_use}, label={args.label}"
    logger.info(f"Starting new run: {run_description}")
    logger.info(f"Before scale train: {y_train[:1]}")
    X_train, y_train = scaler.scaleXy(X_train, y_train)
    logger.info(f"After scale train: {y_train[:1]}")

    if input_points_to_use is not None:
        X_train = X_train[:, :input_points_to_use]
        y_train_so = y_train_so[:, :input_points_to_use]
        X_test = X_test[:, :input_points_to_use]  
        y_test_so = y_test_so[:, :input_points_to_use]  
    logger.info(f"Starting training. {X_train.shape},{X_test.shape},{y_train.shape},{y_test.shape},{y_train_so.shape},{y_test_so.shape}")

    #kernel2 = calc_odd_half(kernel1)
    # Convert data to PyTorch tensors
    inputs, outputs = convert_to_pytorch_tensors(X_train, y_train, y_train_so)

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
    tester = ModelTester(model, criterion, input_points_to_use)
    if args.test_multiple:
        all_y_pred, all_y_test = base.test_multiple(tester, test_files, reps=args.test_reps, skip_stats=True, use_bispectrum=False, skip_ps=True, so_datafiles=sotest_files)
        r2 = base.summarize_test_1000(all_y_pred, all_y_test, output_dir, showplots=args.interactive, saveplots=True, label="_1000")
        base.save_test_results(all_y_pred, all_y_test, output_dir)
    else:
        X_test, stats_test, y_test, y_pred, r2 = tester.test(None, X_test, stats_test, y_test)
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
        'num_epochs': trial.suggest_int('num_epochs', 12, 24),
        'batch_size': 32, #trial.suggest_categorical('batch_size', [16, 32, 48]),
        'learning_rate': 0.002, # trial.suggest_float('learning_rate', 7e-4, 7e-3, log=True), # 0.0019437504084241922, 
        'kernel1': trial.suggest_int('kernel1', 33, 33, step=10),
        'kernel2': trial.suggest_int('kernel2', 33, 33, step=10),
        'dropout': 0.2, #trial.suggest_categorical('dropout', [0.2, 0.3, 0.4, 0.5]),
        'input_points_to_use': 2560,#trial.suggest_int('input_points_to_use', 900, 1400),
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
parser.add_argument('--test_multiple', action='store_true', help='Test 1000 sets of 10 LoS for each test point and plot it')
parser.add_argument('--test_reps', type=int, default=10000, help='Test repetitions for each parameter combination')
args = parser.parse_args()

output_dir = base.create_output_dir(args=args)
logger = base.setup_logging(output_dir)

datafiles = base.get_datafile_list(type='noisy', args=args)
so_datafiles = base.get_datafile_list(type='signalonly', args=args)

test_points = [[-3.00,0.11],[-2.00,0.11],[-1.00,0.11],[-3.00,0.25],[-2.00,0.25],[-1.00,0.25],[-3.00,0.52],[-2.00,0.52],[-1.00,0.52], [-3.00,0.80],[-2.00,0.80],[-1.00,0.80]]#,[0.00,0.80]]
train_files = []
test_files = []
sotrain_files = []
sotest_files = []
for sof, nof in zip(so_datafiles, datafiles):
    is_test_file = False
    for p in test_points:
        if nof.find(f"fX{p[0]:.2f}_xHI{p[1]:.2f}") >= 0:
            test_files.append(nof)
            sotest_files.append(sof)
            is_test_file = True
            break
    if not is_test_file:
        train_files.append(nof)
        sotrain_files.append(sof)

validate_filelist(train_files, sotrain_files, test_files, sotest_files)
scaler = Scaling.Scaler(args)

if args.runmode in ("train_test", "optimize") :
    logger.info(f"Loading train dataset {len(train_files)}")
    X_train, y_train, _ = base.load_dataset(train_files, psbatchsize=1, limitsamplesize=args.limitsamplesize, save=False)
    y_train_so, _, _ = base.load_dataset(sotrain_files, psbatchsize=1, limitsamplesize=args.limitsamplesize, save=False)
    logger.info(f"Loaded datasets X_train:{X_train.shape} y_train:{y_train.shape} y_train_so:{y_train_so.shape}")
    logger.info(f"Loading test dataset {len(test_files)}")
    X_test, y_test, _ = base.load_dataset(test_files, psbatchsize=1, limitsamplesize=args.limitsamplesize, save=False)
    y_test_so, _, _ = base.load_dataset(sotest_files, psbatchsize=1, limitsamplesize=args.limitsamplesize, save=False)
    logger.info(f"Loaded dataset X_test:{X_test.shape} y_test:{y_test.shape} y_test_so:{y_test_so.shape}")

    if args.runmode == "train_test":
        run(X_train, X_test, y_train, y_train_so, y_test, y_test_so, args.epochs, args.trainingbatchsize, lr=0.0019437504084241922, kernel1=269, kernel2=269, dropout=0.2, input_points_to_use=args.input_points_to_use, showplots=args.interactive)

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

