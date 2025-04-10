'''
Predict parameters fX and xHI from the 21cm forest data using CNN and FFNN.
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
import plot_results as pltr
import Scaling
import PS1D
import F21Stats as f21stats
from F21CnnInferenceModel import CnnInferenceModel

import numpy as np
import sys

import matplotlib.pyplot as plt

import optuna
from xgboost import XGBRegressor

def test_multiple(datafiles, model, reps=10000, size=10, input_points_to_use=None):
    logger.info(f"Test_multiple started. {reps} reps x {size} points will be tested for {len(datafiles)} parameter combinations")
    # Create processor with desired number of worker threads
    all_y_test = np.zeros((len(datafiles)*reps, 2))
    all_y_pred = np.zeros((len(datafiles)*reps, 2))
    # Process all files and get results
    for i, f in enumerate(datafiles):
        if i==0: logger.info(f"Working on param combination #{i+1}: {f.split('/')[-1]}")
        los, params, _, _, _ = base.load_dataset([f], psbatchsize=1, limitsamplesize=None, save=False)
        if input_points_to_use is not None:
            los = los[:, :input_points_to_use]

        los_tensor = torch.tensor(los, dtype=torch.float32)
        params_pred = model.predict(los_tensor)

        #if i == 0: logger.info(f"sample test los_so:{los[:1]}")
        y_pred_for_test_point = []
        for j in range(reps):
            #pick 10 samples
            rdm = np.random.randint(len(los), size=size)
            params_pred_set = params_pred[rdm]

            #print(f"latent_features_set.shape={latent_features_set.shape}")
            params_pred_mean = np.mean(params_pred_set, axis=0, keepdims=True)
            #print(f"latent_features_mean.shape={latent_features_mean.shape}")
        
            y_pred_for_test_point.append(params_pred_mean)
            all_y_pred[i*reps+j,:] = params_pred_mean
            all_y_test[i*reps+j,:] = params[0]
            
        logger.info(f"Test_multiple: param combination:{params[0]} predicted mean:{np.mean(y_pred_for_test_point, axis=0)}")

    logger.info(f"Test_multiple completed. actual shape {all_y_test.shape} predicted shape {all_y_pred.shape}")
    
    pltr.calc_squared_error(all_y_pred, all_y_test)

    r2_means = pltr.summarize_test_1000(all_y_pred, all_y_test, output_dir, showplots=args.interactive, saveplots=True, label="_1000")

    r2 = np.mean(r2_means)
    base.save_test_results(all_y_pred, all_y_test, output_dir)

    return r2

def run(X_train, y_train, test_files, num_epochs, batch_size, lr, kernel1, kernel2, dropout, step, input_points_to_use, showplots=False, saveplots=True, criterion=nn.MSELoss()):
    run_description = f"Commandline: {' '.join(sys.argv)}. Parameters: epochs: {num_epochs}, batch_size: {batch_size}, lr: {lr}, kernel_sizes: [{kernel1}, {kernel2}], dropout: {dropout}, points: {input_points_to_use}, label={args.label}"
    logger.info(f"Starting new run: {run_description}")
    logger.info(f"Before scale train: {y_train[:1]}")
    #X_train, y_train = scaler.scaleXy(X_train, y_train)
    logger.info(f"After scale train: {y_train[:1]}")

    if input_points_to_use is not None:
        X_train = X_train[:, :input_points_to_use]

    logger.info(f"Starting training. {X_train.shape},{y_train.shape}")

    # Convert data to PyTorch tensors
    inputs = torch.tensor(X_train)
    outputs = torch.tensor(y_train)

    logger.info(f"Shape of inputs, outputs: {inputs.shape}, {outputs.shape}")
    # Create DataLoader for batching
    train_dataset = TensorDataset(inputs, outputs)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    num_channels = 1
    model = CnnInferenceModel(input_size=len(X_train[0]), input_channels=num_channels, output_size=len(y_train[0]), dropout=dropout, step=step)
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
    model.save_model(f"{output_dir}/cnn_inf_model.pth")  # Save the model to a specified path

    return test_multiple(test_files, model, input_points_to_use=input_points_to_use)


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
        r2 = run(X_train, y_train, test_files, 
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
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.determinisitc=True
torch.backends.cudnn.benchmark=False

parser = base.setup_args_parser()
parser.add_argument('--test_multiple', action='store_true', help='Test 1000 sets of 10 LoS for each test point and plot it')
parser.add_argument('--test_reps', type=int, default=10000, help='Test repetitions for each parameter combination')
parser.add_argument('--vae', action='store_true', help='Use VAE technique')
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

## Loading data
datafiles = base.get_datafile_list(type='noisy', args=args)
if args.maxfiles is not None: datafiles = datafiles[:args.maxfiles]

#test_points = [[-3.00,0.11],[-2.00,0.11],[-1.00,0.11],[-3.00,0.25],[-2.00,0.25],[-1.00,0.25],[-3.00,0.52],[-2.00,0.52],[-1.00,0.52], [-3.00,0.80],[-2.00,0.80],[-1.00,0.80]]#,[0.00,0.80]]
test_points = [[-3,0.11], [-3,0.80], [-1,0.11], [-1,0.80], [-2,0.52]]

train_files = []
test_files = []
for nof in datafiles:
    is_test_file = False
    for p in test_points:
        if nof.find(f"fX{p[0]:.2f}_xHI{p[1]:.2f}") >= 0:
            test_files.append(nof)
            is_test_file = True
            break
    if not is_test_file:
        train_files.append(nof)

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


scaler = Scaling.Scaler(args)


if args.runmode in ("train_test", "test_only", "optimize"):
    # Loss function and optimizer
    #criterion = CustomLoss()  # You can adjust alpha as needed
    criterion = nn.MSELoss()
    #criterion = ChiSquareLoss()  
    
    if args.runmode in ("train_test", "optimize") :
        logger.info(f"Loading training dataset {len(train_files)}")
        X_train, y_train, _, keys, _ = base.load_dataset(train_files, psbatchsize=1, limitsamplesize=args.limitsamplesize, save=False, skip_ps=True)
        if args.input_points_to_use is not None:
            X_train = X_train[:, :args.input_points_to_use]
        logger.info(f"Loaded datasets X_train:{X_train.shape} y_train:{y_train.shape}")

    if args.runmode == "train_test":
        run(X_train, y_train, test_files, args.epochs, args.trainingbatchsize, lr=0.0001, kernel1=kernel1, kernel2=kernel1, dropout=0.2, step=step, input_points_to_use=args.input_points_to_use, showplots=args.interactive, criterion=criterion)
    elif args.runmode == "test_only":
        logger.info(f"Loading model from file {args.modelfile}")
        model = CnnInferenceModel(input_size=args.input_points_to_use, input_channels=1, output_size=args.input_points_to_use+2, dropout=0.2, step=step)
        model.load_model(args.modelfile)
        logger.info(f"testing with {len(test_files)} test cases")
        test_multiple(test_files, model=model, input_points_to_use=args.input_points_to_use)
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




