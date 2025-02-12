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
import PS1D

import numpy as np
import sys

import matplotlib.pyplot as plt

import optuna

class UnetModel(nn.Module):
    def __init__(self, input_size, input_channels, output_size, dropout=0.2, step=4):
        super(UnetModel, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 41, padding=20),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 64, 21, padding=10),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 64, 11, padding=5),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 64, 5, padding=2),  # New layer added
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(step)
        )

        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, 41, padding=20),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 128, 21, padding=10),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 128, 11, padding=5),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 128, 5, padding=2),  # New layer added
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(step)
        )

        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, 41, padding=20),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(256, 256, 21, padding=10),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(256, 256, 11, padding=5),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(256, 256, 5, padding=2),  # New layer added
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(step)
        )
        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, step, stride=step, output_padding=0),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(256, 64, step, stride=step, output_padding=0),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(128, 32, step, stride=step, output_padding=0),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Final layer
        channels = input_channels + 32
        self.final = nn.Sequential(
            nn.Conv1d(channels, 1, 1),  # Change output channels to 1
            nn.Flatten()  # Add flatten layer to match target shape
        )

    def forward(self, x):
        # Print shapes for debugging
        #print(f"Input shape: {x.shape}")
        # If input is single channel, add channel dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        # If input already has channels, it will remain unchanged
        #print(f"After unsqueeze: {x.shape}")        
        # Encoder
        enc1 = self.enc1(x)
        #print(f"After enc1: {enc1.shape}")        
        enc2 = self.enc2(enc1)
        #print(f"After enc2: {enc2.shape}")        
        enc3 = self.enc3(enc2)
        #print(f"After enc3: {enc3.shape}")        
        
        """
        # Pass through the dense layers for parameters extraction
        dense_out = self.dense1(enc4.view(enc4.size(0), -1))  # Flatten for dense layer
        #print(f"After dense1: {dense_out.shape}")        
        dense_out = nn.Tanh()(dense_out)
        dense_out = self.dense2(dense_out)
        #print(f"After dense2: {dense_out.shape}")        
        dense_out = nn.ReLU()(dense_out)
        dense_out = self.dense3(dense_out)
        #print(f"After dense3: {dense_out.shape}")        
        """
        #print(f"Output of parameters extraction network: {dense_out.shape}")

        # Decoder with skip connections
        dec1 = self.dec1(enc3)
        #print(f"After dec1: {dec1.shape}")        
        dec1 = torch.cat([dec1, enc2], dim=1)
        #print(f"After dec1-cat: {dec1.shape}")        

        # Decoder with skip connections
        dec2 = self.dec2(dec1)
        #print(f"After dec1: {dec1.shape}")        
        dec2 = torch.cat([dec2, enc1], dim=1)
        #print(f"After dec1-cat: {dec1.shape}")        
        
        dec3 = self.dec3(dec2)
        #print(f"After dec2: {dec2.shape}")        
        dec3 = torch.cat([dec3, x], dim=1)
        
        #print(f"Before final: {dec4.shape}")
        out = self.final(dec3)

        # Concatenate parameter extraction output with decoder output
        # out = torch.cat((dense_out, out), dim=1)  # Concatenate along the feature dimension
        #print(f"Output shape after concatenation: {out.shape}")

        #print(f"Output shape: {out.shape}")
        return out

    def save_model(self, file_path):
        """Save the model to a file."""
        torch.save(self.state_dict(), file_path)  # Save the model's state_dict

    def load_model(self, file_path):
        """Load the model from a file."""
        self.load_state_dict(torch.load(file_path))  # Load the model's state_dict
        self.eval()  # Set the model to evaluation mode

class ModelTester:
    def __init__(self, model, criterion, input_points_to_use, noise):
        self.model = model
        self.criterion = criterion
        self.input_points_to_use = input_points_to_use
        if noise is not None: self.noise = noise[:, :self.input_points_to_use]
        else: self.noise = None
    
    def test(self, los_test, p_test, stats_test, bispectrum_set, y_test, los_so, silent=True):
        if self.input_points_to_use is not None: 
            los_test = los_test[:, :self.input_points_to_use]
            los_so = los_so[:, :self.input_points_to_use]

        if not silent: logger.info(f"Testing dataset: X:{los_test.shape} y:{y_test.shape}")
        #if not silent: logger.info(f"Before scale y_test: {y_test[:1]}")
        #los_test, y_test = scaler.scaleXy(los_test, y_test)
        #if not silent: logger.info(f"After scale y_test: {y_test[:1]}")

        #if not silent: logger.info("Testing prediction")
        #if not silent: logger.info(f"Sample data before testing y:{y_test[0]}\nX:{los_test[0]}")

        r2 = None
        with torch.no_grad():
            # Test the model
            if not silent: logger.info("Testing prediction")

            test_input, test_output = convert_to_pytorch_tensors(los_test, y_test, los_so, self.noise, silent=silent)
            if not silent: logger.info(f"Shape of test_input, test_output: {test_input.shape}, {test_output.shape}")
            y_pred_tensor = self.model(test_input)
            test_loss = self.criterion(y_pred_tensor, test_output)
            if not silent: logger.info(f'Test Loss: {test_loss.item():.8f}')

            # Calculate R2 scores
            y_pred_np = y_pred_tensor.detach().cpu().numpy()
            #y_pred = y_pred_np[:,:2]
            y_pred_los = y_pred_np#[:,2:]
            r2 = r2_score(los_test, y_pred_los)
            #if not silent: logger.info("R2 Score: " + str(r2))
            # Calculate rmse scores
            rms_scores = mean_squared_error(los_test, y_pred_los)
            rms_scores_percent = np.sqrt(rms_scores) * 100 / np.mean(los_test, axis=0)
            if not silent: logger.info("RMS Error: " + str(rms_scores_percent))

            if not silent: plot_predictions(los_test, los_so, y_pred_los, label=f"xHI{y_test[0][0]:.2f}_logfx{5.0*(y_test[0][1] - 0.8):.2f}")
            if not silent: analyse_predictions(los_test, los_so, y_pred_los, label=f"xHI{y_test[0][0]:.2f}_logfx{5.0*(y_test[0][1] - 0.8):.2f}")
    
            
            #if not silent: logger.info("R2 Score: " + str(r2))

            #if not silent: logger.info(f"Before unscale y_pred: {y_pred[:1]}")
            #y_pred = scaler.unscale_y(y_pred)
            #if not silent: logger.info(f"After unscale y_pred: {y_pred[:1]}")
            #if not silent: logger.info(f"Before unscale y_test: {y_test[:1]}")
            #los_test, y_test = scaler.unscaleXy(los_test, y_test)
            #if not silent: logger.info(f"After unscale y_test: {y_test[:1]}")
            #if not silent: logger.info(f"unscaled test result {los_test.shape} {y_test.shape} {y_pred.shape}")

            # Calculate rmse scores
            #rms_scores = [mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(len(y_pred[0]))]
            #rms_scores_percent = np.sqrt(rms_scores) * 100 / np.mean(y_test, axis=0)
            #if not silent: logger.info("RMS Error: " + str(rms_scores_percent))    
        return los_test, y_test, dummy_y_pred(y_test), r2

def dummy_y_pred(y_test):
    dummy = np.zeros(y_test.shape)
    return dummy

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

def convert_to_pytorch_tensors(X, y, y_so, X_noise, silent=True):
# Create different channel representations based on args.channels
    channels = []
    if not silent: logger.info(f"Appending channel with shape: {X.shape}")
    channels.append(X)

    # Channel 1: Noise
    if args.use_noise_channel:
        noisechannel = np.repeat(X_noise, repeats=len(X), axis=0)
        if not silent: logger.info(f"Appending channel with shape: {noisechannel.shape}")
        channels.append(noisechannel)

    # Stack channels along a new axis
    combined_input = np.stack(channels, axis=1)
    
    # Convert to PyTorch tensors with float32 dtype
    X_tensor = torch.tensor(combined_input, dtype=torch.float32)
    y_tensor = torch.tensor(X, dtype=torch.float32)

    if not silent: logger.info(f"convert_to_pytorch_tensors: shape of tensors: X:{X_tensor.shape}, Y: {y_tensor.shape}")

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
    
def plot_power_spectra(ps_set, ks, title, labels, xscale='log', yscale='log', showplots=False, saveplots=True):
    print(f"shapes: {ps_set.shape},{ks.shape}")

    base.initplt()
    plt.title(f'{title}')
    if len(ps_set.shape) > 1:
        for i, ps in enumerate(ps_set):
            if labels is not None: label = labels[i]
            row_ks = None
            if ks is not None:
                if len(ks.shape) > 1: row_ks = ks[i]
                else: row_ks = ks
            plt.plot(row_ks*1e6, ps, label=label, marker='o')
    else:
        row_ks = None
        if ks is not None:
                if len(ks.shape) > 1: row_ks = ks[0]
                else: row_ks = ks
        plt.plot(ks*1e6, ps, label=label, marker='o')
        #plt.scatter(ks[1:]*1e6, ps[1:], label=label)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlabel(r'k (Hz$^{-1}$)')
    plt.ylabel(r'$kP_{21}$')
    plt.legend()
    if showplots: plt.show()
    if saveplots: plt.savefig(f"{output_dir}/reconstructed_ps_{title}.png")
    plt.clf()

def analyse_predictions(los_test, y_test_so, y_pred_so, samples=1, showplots=False, saveplots=True, label='', signal_bandwidth=22089344.0):
    ks_noisy, ps_noisy = PS1D.get_P_set(los_test, signal_bandwidth, scaled=True)
    logger.info(f'get_P_set: {ks_noisy.shape}, {ps_noisy.shape},')
    ps_noisy_mean = np.mean(ps_noisy, axis=0)
    ks_so, ps_so = PS1D.get_P_set(y_test_so, signal_bandwidth, scaled=True)
    ps_so_mean = np.mean(ps_so, axis=0)
    ks_pred, ps_pred = PS1D.get_P_set(y_pred_so, signal_bandwidth, scaled=True)
    ps_pred_mean = np.mean(ps_pred, axis=0)

    plot_power_spectra(np.vstack((ps_so_mean,ps_noisy_mean,ps_pred_mean)), ks_noisy[0,:], title=label, labels=["signal-only", "noisy-signal", "reconstructed"])

def plot_predictions(los_test, y_test_so, y_pred_so, samples=1, showplots=False, saveplots=True, label=''):
    plt.rcParams['figure.figsize'] = [15, 9]
    plt.title(f'Reconstructed LoS vs Actual Noiseless LoS {label}')
    for i, (noisy, test, pred) in enumerate(zip(los_test[:samples], y_test_so[:samples], y_pred_so[:samples])):
        plt.plot(noisy-0.01, label='Signal with Noise')
        plt.plot(test, label='Actual signal')
        plt.plot(pred+0.01, label='Reconstructed')
        plt.plot(test-pred+0.98, label='Reconstructed - Signal')
        if i> 2: break
    plt.xlabel('frequency'), 
    plt.ylabel('flux/S147')
    plt.legend()
    if showplots: plt.show()
    if saveplots: plt.savefig(f"{output_dir}/reconstructed_los_{label}.png")
    plt.clf()

def load_noise():
    X_noise = None
    if args.use_noise_channel:
        noisefiles = base.get_datafile_list(type='noiseonly', args=args)
        X_noise, _, _ = base.load_dataset(noisefiles, psbatchsize=1000, limitsamplesize=1000, save=False)
        logger.info(f"Loaded noise with shape: {X_noise.shape}")
    return X_noise
    
def run(X_train, X_test, y_train, y_train_so, y_test, y_test_so, X_noise, num_epochs, batch_size, lr, kernel1, kernel2, dropout, step, input_points_to_use, showplots=False, saveplots=True):
    run_description = f"Commandline: {' '.join(sys.argv)}. Parameters: epochs: {num_epochs}, batch_size: {batch_size}, lr: {lr}, kernel_sizes: [{kernel1}, {kernel2}], dropout: {dropout}, points: {input_points_to_use}, label={args.label}"
    logger.info(f"Starting new run: {run_description}")
    logger.info(f"Before scale train: {y_train[:1]}")
    #X_train, y_train = scaler.scaleXy(X_train, y_train)
    #if X_noise is not None: X_noise, _ = scaler.scaleXy(X_noise, None)
    #logger.info(f"After scale train: {y_train[:1]}")

    if input_points_to_use is not None:
        X_train = X_train[:, :input_points_to_use]
        y_train_so = y_train_so[:, :input_points_to_use]
        if X_noise is not None: X_noise = X_noise[:, :input_points_to_use]
    logger.info(f"Starting training. {X_train.shape},{y_train.shape},{y_train_so.shape}")

    #kernel2 = calc_odd_half(kernel1)
    # Convert data to PyTorch tensors
    inputs, outputs = convert_to_pytorch_tensors(X_train, y_train, y_train_so, X_noise)

    logger.info(f"Shape of inputs, outputs: {inputs.shape}, {outputs.shape}")
    # Create DataLoader for batching
    train_dataset = TensorDataset(inputs, outputs)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    num_channels = 1
    if args.use_noise_channel: num_channels = 2
    model = UnetModel(input_size=len(X_train[0]), input_channels=num_channels, output_size=len(y_train[0]), dropout=dropout, step=step)
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
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.8f}')

    # Evaluate the model (on a test set, here we just use the training data for simplicity)
    model.eval()  # Set the model to evaluation mode
    model.save_model(f"{output_dir}/unet_model.pth")  # Save the model to a specified path

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
    return test(X_test, y_test, y_test_so, X_noise, model, criterion, input_points_to_use, run_description)

def test(X_test, y_test, y_test_so, X_noise, model, criterion, input_points_to_use, run_description):
    if input_points_to_use is not None:
        X_test = X_test[:, :input_points_to_use]  
        y_test_so = y_test_so[:, :input_points_to_use]  
    logger.info(f"Starting testing. {X_test.shape},{y_test.shape},{y_test_so.shape}")

    tester = ModelTester(model, criterion, input_points_to_use, X_noise)
    if args.test_multiple:
        all_y_pred, all_y_test = base.test_multiple(tester, test_files, reps=args.test_reps, skip_stats=True, use_bispectrum=False, skip_ps=True, so_datafiles=sotest_files)
        r2 = base.summarize_test_1000(all_y_pred, all_y_test, output_dir, showplots=args.interactive, saveplots=True, label="_1000")
        base.save_test_results(all_y_pred, all_y_test, output_dir)
    else:
        X_test, y_test, y_pred, r2 = tester.test(X_test, None, None, None, y_test, los_so=y_test_so, silent=False)
        r2 = base.summarize_test_1000(y_pred, y_test, output_dir=output_dir, showplots=args.interactive, saveplots=True)
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
        'learning_rate': 0.001, # trial.suggest_float('learning_rate', 7e-4, 7e-3, log=True), # 0.0019437504084241922, 
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
if args.input_points_to_use not in [2048, 128]: raise ValueError(f"Invalid input_points_to_use {args.input_points_to_use}")
if args.input_points_to_use == 2048: 
    step = 4
    kernel1 = 256
else: 
    step = 2
    kernel1 = 16

output_dir = base.create_output_dir(args=args)
logger = base.setup_logging(output_dir)

logger.info(f"input_points={args.input_points_to_use}, kernel1={kernel1}, step={step}")

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

if args.runmode in ("train_test", "test_only", "optimize") :
    if args.runmode in ("train_test", "optimize") :
        logger.info(f"Loading train dataset {len(train_files)}")
        X_train, y_train, _ = base.load_dataset(train_files, psbatchsize=1, limitsamplesize=args.limitsamplesize, save=False)
        y_train_so, _, _ = base.load_dataset(sotrain_files, psbatchsize=1, limitsamplesize=args.limitsamplesize, save=False)
        logger.info(f"Loaded datasets X_train:{X_train.shape} y_train:{y_train.shape} y_train_so:{y_train_so.shape}")
    logger.info(f"Loading test dataset {len(test_files)}")
    X_test, y_test, _ = base.load_dataset(test_files, psbatchsize=1, limitsamplesize=args.limitsamplesize, save=False)
    y_test_so, _, _ = base.load_dataset(sotest_files, psbatchsize=1, limitsamplesize=args.limitsamplesize, save=False)
    logger.info(f"Loaded dataset X_test:{X_test.shape} y_test:{y_test.shape} y_test_so:{y_test_so.shape}")
    X_noise = load_noise()
    if args.runmode == "train_test":
        run(X_train, X_test, y_train, y_train_so, y_test, y_test_so, X_noise, args.epochs, args.trainingbatchsize, lr=0.001, kernel1=kernel1, kernel2=kernel1, dropout=0.2, step=step, input_points_to_use=args.input_points_to_use, showplots=args.interactive)
    elif args.runmode == "test_only":
        logger.info(f"Loading model from file {args.modelfile}")
        model = UnetModel(input_size=args.input_points_to_use, input_channels=1, output_size=args.input_points_to_use+2, dropout=0.2, step=step)
        model.load_model(args.modelfile)
        logger.info(f"testing with {len(X_test)} test cases")
        test(X_test, y_test, y_test_so, None, model, nn.MSELoss(), args.input_points_to_use, "test_only")
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



