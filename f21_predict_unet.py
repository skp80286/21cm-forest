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
import F21Stats as f21stats
from UnetModelNoInputSkipConn import UnetModel

import numpy as np
import sys

import matplotlib.pyplot as plt

import optuna

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
            y_pred_so = y_pred_np#[:,2:]
            r2 = r2_score(los_so, y_pred_so)
            if not silent: logger.info("R2 Score: " + str(r2))
            # Calculate rmse scores
            rms_score = np.mean(mean_squared_error(los_so, y_pred_so))
            #rms_scores_percent = np.sqrt(rms_scores) * 100 / np.mean(y_test_so, axis=0)
            if not silent: logger.info("RMS Error: " + str(rms_score))
            if not silent:
                    # Write each array to separate CSV files
                    np.savetxt(f"{output_dir}/los_test.csv", X_test, delimiter=",")
                    np.savetxt(f"{output_dir}/los_so.csv", y_test_so, delimiter=",")
                    np.savetxt(f"{output_dir}/y_pred_so.csv", y_pred_so, delimiter=",")
                    np.savetxt(f"{output_dir}/y_test.csv", y_test, delimiter=",")

            if not silent:
                # for i in range(0, len(los_test)):
                #     logger.info(f"los_test[{i}/{len(los_test)}]:xHI{y_test[i][0]:.2f}_logfx{y_test[i][1]:.2f}")
                processed_labels = set() 
                for i in range(len(los_test)):
                    label=f"xHI{y_test[i][0]:.2f}_logfx{y_test[i][1]:.2f}"
                    if label in processed_labels:  # Skip if label has already been processed
                        continue
                    processed_labels.add(label)  # Add label to the set
                    plot_predictions(los_test[i:i+1], los_so[i:i+1], y_pred_so[i:i+1], label=label)
                    analyse_predictions(los_test[i:i+1], los_so[i:i+1], y_pred_so[i:i+1], label=label)
    
            
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
    y_tensor = torch.tensor(y_so, dtype=torch.float32)

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
    
class ChiSquareLoss(nn.Module):
    def __init__(self):
        super(ChiSquareLoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets):
        # First component: MSE between predictions and targets
        #mse_loss_xHI = self.mse(predictions[:,0], targets[:,0])
        mse = self.mse(predictions, targets)
        var = targets.var()
        if var < 1e-8: var = 1.0

        return mse/var

markers=['o', 'x', '*']
def plot_power_spectra(ps_set, ks, title, labels, xscale='log', yscale='log', showplots=False, saveplots=True):
    #print(f"plot_power_spectra: shapes: {ps_set.shape},{ks.shape}")

    base.initplt()
    plt.title(f'{title}')
    if len(ps_set.shape) > 1:
        for i, ps in enumerate(ps_set):

            if labels is not None: label = labels[i]
            row_ks = None
            if ks is not None:
                if len(ks.shape) > 1: row_ks = ks[i]
                else: row_ks = ks
            plt.plot(row_ks*1e6, ps, label=label, marker=markers[i% len(markers)], alpha=0.5)
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
    plt.close()

def analyse_predictions(los_test, y_test_so, y_pred_so, samples=1, showplots=False, saveplots=True, label='', signal_bandwidth=22089344.0):
    ks_noisy, ps_noisy = PS1D.get_P_set(los_test, signal_bandwidth, scaled=True)
    #logger.info(f'get_P_set: {ks_noisy.shape}, {ps_noisy.shape},')
    ks_noisy, ps_noisy = f21stats.logbin_power_spectrum_by_k(ks_noisy, ps_noisy)
    #logger.info(f'get_P_set: {ks_noisy.shape}, {ps_noisy.shape},')
    ps_noisy_mean = np.mean(ps_noisy, axis=0)
    ks_so, ps_so = PS1D.get_P_set(y_test_so, signal_bandwidth, scaled=True)
    ks_so, ps_so = f21stats.logbin_power_spectrum_by_k(ks_so, ps_so)
    ps_so_mean = np.mean(ps_so, axis=0)
    ks_pred, ps_pred = PS1D.get_P_set(y_pred_so, signal_bandwidth, scaled=True)
    ks_pred, ps_pred = f21stats.logbin_power_spectrum_by_k(ks_pred, ps_pred)
    ps_pred_mean = np.mean(ps_pred, axis=0)

    plot_power_spectra(np.vstack((ps_so_mean,ps_noisy_mean,ps_pred_mean)), ks_noisy[0,:], title=label, labels=["signal-only", "noisy-signal", "reconstructed"])

def plot_predictions(los_test, y_test_so, y_pred_so, samples=1, showplots=False, saveplots=True, label=''):
    for i, (noisy, test, pred) in enumerate(zip(los_test[:samples], y_test_so[:samples], y_pred_so[:samples])):
        plt.figure()
        plt.rcParams['figure.figsize'] = [15, 9]
        plt.title(f'Reconstructed LoS vs Actual Noiseless LoS {label}')
        plt.plot(noisy-0.01, label='Signal with Noise')
        plt.plot(test, label='Actual signal')
        plt.plot(pred+0.01, label='Reconstructed')
        plt.plot(test-pred+0.98, label='Reconstructed - Signal')
        plt.xlabel('frequency'), 
        plt.ylabel('flux/S147')
        plt.legend()
        if showplots: plt.show()
        if saveplots: plt.savefig(f"{output_dir}/reconstructed_los_{label}.png")
        if i> 5: break
        plt.close()

def load_noise():
    X_noise = None
    if args.use_noise_channel:
        noisefiles = base.get_datafile_list(type='noiseonly', args=args)
        X_noise, _, _ = base.load_dataset(noisefiles, psbatchsize=1000, limitsamplesize=1000, save=False)
        logger.info(f"Loaded noise with shape: {X_noise.shape}")
    return X_noise
    
def run(X_train, X_test, y_train, y_train_so, y_test, y_test_so, X_noise, num_epochs, batch_size, lr, kernel1, kernel2, dropout, step, input_points_to_use, showplots=False, saveplots=True, criterion=nn.MSELoss()):
    run_description = f"Commandline: {' '.join(sys.argv)}. Parameters: epochs: {num_epochs}, batch_size: {batch_size}, lr: {lr}, kernel_sizes: [{kernel1}, {kernel2}], dropout: {dropout}, points: {input_points_to_use}, label={args.label}"
    logger.info(f"Starting new run: {run_description}")
    logger.info(f"Before scale train: {y_train[:1]}")
    X_train, y_train = scaler.scaleXy(X_train, y_train)
    if X_noise is not None: X_noise, _ = scaler.scaleXy(X_noise, None)
    logger.info(f"After scale train: {y_train[:1]}")

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
            #if i == 50: logger.info(f'predictions:{predictions}\input_batch:{input_batch}\noutput_batch:{output_batch}\nloss:{loss}')

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

def reorder_so(y_so, keys_so, keys):
    logger.info(f"Reordering.. {len(y_so)}, {len(keys_so)}, {len(keys)}")
    key_to_index = {key: index for index, key in enumerate(keys_so)}
    y_so_reordered = np.zeros_like(y_so)

    for index, key in enumerate(keys):
        if key in key_to_index:
            y_so_reordered[index] = y_so[key_to_index[key]]
        else:
            raise ValueError(f"key not found! {key}")

    keys_so = [keys_so[key_to_index[key]] for key in keys if key in key_to_index]
    return y_so_reordered

# main code starts here
parser = base.setup_args_parser()
parser.add_argument('--test_multiple', action='store_true', help='Test 1000 sets of 10 LoS for each test point and plot it')
parser.add_argument('--test_reps', type=int, default=10000, help='Test repetitions for each parameter combination')
args = parser.parse_args()
#if args.input_points_to_use not in [2048, 128]: raise ValueError(f"Invalid input_points_to_use {args.input_points_to_use}")
if args.input_points_to_use >= 2048: 
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

if args.runmode in ("train_test", "test_only", "optimize"):
    # Loss function and optimizer
    #criterion = CustomLoss()  # You can adjust alpha as needed
    #criterion = nn.MSELoss()
    criterion = ChiSquareLoss()  
    
    if args.runmode in ("train_test", "optimize") :
        logger.info(f"Loading train dataset {len(train_files)}")
        X_train, y_train, _, keys = base.load_dataset(train_files, psbatchsize=1, limitsamplesize=args.limitsamplesize, save=False)
        y_train_so, _, _, keys_so = base.load_dataset(sotrain_files, psbatchsize=1, limitsamplesize=args.limitsamplesize, save=False)
        y_train_so = reorder_so(y_train_so, keys_so, keys)
        logger.info(f"Loaded datasets X_train:{X_train.shape} y_train:{y_train.shape} y_train_so:{y_train_so.shape}")
    logger.info(f"Loading test dataset {len(test_files)}")
    X_test, y_test, _, keys = base.load_dataset(test_files, psbatchsize=1, limitsamplesize=args.limitsamplesize, save=False)
    y_test_so, _, _, keys_so = base.load_dataset(sotest_files, psbatchsize=1, limitsamplesize=args.limitsamplesize, save=False)
    y_test_so = reorder_so(y_test_so, keys_so, keys)

    logger.info(f"Loaded dataset X_test:{X_test.shape} y_test:{y_test.shape} y_test_so:{y_test_so.shape}")
    X_noise = load_noise()
    if args.runmode == "train_test":
        run(X_train, X_test, y_train, y_train_so, y_test, y_test_so, X_noise, args.epochs, args.trainingbatchsize, lr=0.001, kernel1=kernel1, kernel2=kernel1, dropout=0.2, step=step, input_points_to_use=args.input_points_to_use, showplots=args.interactive, criterion=criterion)
    elif args.runmode == "test_only":
        logger.info(f"Loading model from file {args.modelfile}")
        model = UnetModel(input_size=args.input_points_to_use, input_channels=1, output_size=args.input_points_to_use+2, dropout=0.2, step=step)
        model.load_model(args.modelfile)
        logger.info(f"testing with {len(X_test)} test cases")
        test(X_test, y_test, y_test_so, None, model, criterion, args.input_points_to_use, "test_only")
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

