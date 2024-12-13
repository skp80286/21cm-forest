'''
Predict parameters fX and xHI from the 21cm forest data using CNN.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


import f21_predict_base as base
import numpy as np
import sys
import pickle

import optuna

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

    return (all_los, all_params, [])

def load_noise():
    X_noise = None
    if args.use_noise_channel:
        noisefiles = base.get_datafile_list(type='noiseonly', args=args)
        X_noise, _, _ = base.load_dataset(noisefiles, psbatchsize=1000, limitsamplesize=1000, save=False)
    return X_noise
    
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

def calc_odd_half(n):
    h = n//2
    return  h + 1 if h % 2 == 0 else h

def calc_odd_thirds(n):
    t1 = 2*n//3
    t1 =  t1 + 1 if t1 % 2 == 0 else t1
    t2 = n//3
    t2 =  t2 + 1 if t2 % 2 == 0 else t2
    return t1, t2

class CNNModel(nn.Module):
    def __init__(self, input_size, output_size, channels=1, kernel1=269, kernel2=135, dropout=0.5):
        super(CNNModel, self).__init__()
        #kernel2 = calc_odd_half(kernel1)
        #kernel3 = calc_odd_half(kernel2)

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(channels, 32, kernel_size=kernel1, padding='same'),
            nn.LeakyReLU(),
            nn.Conv1d(32, 32, kernel_size=kernel1, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),

            # Second conv block
            nn.Conv1d(32, 64, kernel_size=kernel1//4, padding='same'),
            nn.LeakyReLU(),
            nn.Conv1d(64, 64, kernel_size=kernel1//4, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=kernel1//16, padding='same'),
            nn.LeakyReLU(),
            nn.Conv1d(128, 128, kernel_size=kernel1//16, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),
        )
        
        # Calculate the size of flattened features
        self.flatten_size = ((((input_size)//4)//4)//4)*128
        # Dense layers
        self.dense_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 1024),
            nn.LeakyReLU(),
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
    
class ProductLoss(nn.Module):
    def __init__(self):
        super(ProductLoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets):
        pred_product = predictions[:, 0] * predictions[:, 1]
        target_product = targets[:, 0] * targets[:, 1]
        product_loss = self.mse(pred_product, target_product)
        
        return product_loss

class XHILoss(nn.Module):
    def __init__(self):
        super(XHILoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets):

        xHI_loss = self.mse(predictions[:,0], targets[:,0])
        
        return xHI_loss


class LogfxLoss(nn.Module):
    def __init__(self):
        super(LogfxLoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets):

        logfx_loss = self.mse(predictions[:,1], targets[:,1])
        
        return logfx_loss

class CustomLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.25):
        super(CustomLoss, self).__init__()
        self.alpha = alpha  # Weight for balancing the two loss components
        self.beta = beta  # Weight for balancing the two loss components
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets):
        # First component: MSE between predictions and targets
        #mse_loss_xHI = self.mse(predictions[:,0], targets[:,0])
        mse_fx = self.mse(predictions[:,1], targets[:,1])
        mse_product = self.mse(predictions[:,2], targets[:,2])
        
        # Second component: MSE between product of predictions and value 3
        #pred_product = predictions[:, 0] * predictions[:, 1]
        #product_loss = 100*self.mse(pred_product, predictions[:,2])
        
        # Combine both losses
        total_loss = self.alpha * mse_product + (1 - self.alpha) * mse_fx

        #if args.trials == 1: logger.info(f"Loss calculation:\n##predictions\n{predictions}\n##targets:\n{targets}\n##loss1\n{mse_loss_fx}\n##loss2\n{mse_loss_product}\n##total_loss\n{total_loss}")
        return total_loss

def convert_to_pytorch_tensors(X, y, samples, X_noise, window_size):
    # Create different channel representations based on args.channels
    channels = []
    
    # Channel 1: One of either Noise or Aggregated LoS 
    if args.use_noise_channel:
        noisechannel = np.repeat(X_noise, repeats=len(X), axis=0)
        logger.info(f"Appending channel with shape: {noisechannel.shape}")
        channels.append(noisechannel)
    else: 
        logger.info(f"Appending channel with shape: {X.shape}")
        channels.append(X)
    
    if args.channels > 1:
        # Channel 2: Log of signal
        logchannel = np.log(np.clip(samples[:, 0, :], 1e-10, None))
        logger.info(f"Appending channel with shape: {logchannel.shape}")
        channels.append(logchannel)
        
    if args.channels > 2:
        # Channel 3: Gradient of signal
        gradchannel = np.gradient(samples[:, 0, :], axis=1)
        logger.info(f"Appending channel with shape: {gradchannel.shape}")
        channels.append(gradchannel)
        
    if args.channels > 3:
        # Channel 4: Moving average
        movavgchannel = np.array([np.convolve(row, np.ones(window_size)/window_size, mode='same') for row in samples[:,0,:]])
        logger.info(f"Appending channel with shape: {movavgchannel.shape}")
        channels.append(movavgchannel)
    if args.channels > 4:
        for i in range(1):#samples.shape[1]):
            sampleschannel = np.array(samples)[:,i,:]
            logger.info(f"Appending channel with shape: {sampleschannel.shape}")
            channels.append(sampleschannel)

    # Stack channels along a new axis
    combined_input = np.stack(channels[:args.channels], axis=1)
    
    # Convert to PyTorch tensors with float32 dtype
    X_tensor = torch.tensor(combined_input, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    return X_tensor, y_tensor

def run(X_train, train_samples, X_noise, X_test, test_samples,y_train, y_test, num_epochs, batch_size, lr, kernel1, kernel2, dropout, input_points_to_use, showplots=False, saveplots=True):
    run_description = f"output_dir={output_dir} Commandline: {' '.join(sys.argv)}. Parameters: epochs: {num_epochs}, kernel_size: {kernel1}, points: {input_points_to_use}, label={args.label}"
    logger.info(f"Starting new run: {run_description}")
    X_train, y_train = base.scaleXy(X_train, y_train, args)
    X_test, y_test = base.scaleXy(X_test, y_test, args)

    if input_points_to_use is not None:
        X_train = X_train[:, :input_points_to_use]
        train_samples = train_samples[:,:, :input_points_to_use]
        X_noise = X_noise[:, :input_points_to_use]
        X_test = X_test[:, :input_points_to_use]  
        test_samples = test_samples[:,:, :input_points_to_use]
    logger.info(f"Starting training. {X_train.shape},{X_test.shape},{y_train.shape},{y_test.shape}")

    #kernel2 = calc_odd_half(kernel1)
    # Convert data to PyTorch tensors
    inputs, outputs = convert_to_pytorch_tensors(X_train, y_train, train_samples, X_noise, window_size=kernel1)

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
    if args.xhi_only:
        criterion = XHILoss()  
    elif args.logfx_only:
        criterion = LogfxLoss()
    elif args.scale_y1:
        criterion = CustomLoss()  # You can adjust alpha as needed
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
    save_model(model)

    r2 = None
    with torch.no_grad():
        # Test the model
        logger.info("Testing prediction")

        test_input, test_output = convert_to_pytorch_tensors(X_test, y_test, test_samples, X_noise, window_size=kernel2)
        logger.info(f"Shape of test_input, test_output: {test_input.shape}, {test_output.shape}")
        y_pred = model(test_input)
        test_loss = criterion(y_pred, test_output)
        logger.info(f'Test Loss: {test_loss.item():.4f}')


        # Calculate R2 scores
        y_pred = y_pred.detach().cpu().numpy()

        r2 = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(len(y_pred[0]))]
        logger.info("R2 Score: " + str(r2))
        # Calculate rmse scores
        rms_scores = [mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(len(y_pred[0]))]
        rms_scores_percent = np.sqrt(rms_scores) * 100 / np.mean(y_test, axis=0)
        logger.info("RMS Error: " + str(rms_scores_percent))
 
        y_pred = unscale_y(y_pred)
        X_test, y_test = unscaleXy(X_test, y_test)
        logger.info(f"unscaled test result {X_test.shape} {y_test.shape} {y_pred.shape}")
    
    base.summarize_test_1000(y_pred, y_test, output_dir=output_dir, showplots=showplots, saveplots=saveplots)
    if args.scale_y1: combined_r2 = r2[2]
    elif args.scale_y2: combined_r2 = r2
    elif args.xhi_only: combined_r2 = r2
    elif args.logfx_only: combined_r2 = r2
    else: combined_r2 = 0.5*(r2[0]+r2[1])
    
    logger.info(f"Finished run: score={combined_r2}, r2={r2}. {run_description}")
    return combined_r2


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

# main code start here
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.determinisitc=True
torch.backends.cudnn.benchmark=False

parser = base.setup_args_parser()
args = parser.parse_args()
output_dir = base.create_output_dir(args)
logger = base.setup_logging()

datafiles = base.get_datafile_list(type='noisy', args=args)
test_size = 16
if args.maxfiles is not None:
    datafiles = datafiles[:args.maxfiles]
    test_size = 1

train_files, test_files = train_test_split(datafiles, test_size=test_size, random_state=42)

if args.runmode == "train_test":
    logger.info(f"Loading train dataset {len(train_files)}")
    if args.use_saved_los_data:
        X_train, y_train, train_samples = base.load_dataset_from_pkl()
    else:
        X_train, y_train, train_samples = base.load_dataset(train_files, psbatchsize=args.psbatchsize, limitsamplesize=args.limitsamplesize, save=True)
    logger.info(f"Loaded dataset X_train:{X_train.shape} y:{y_train.shape}")
    logger.info(f"Loading test dataset {len(test_files)}")
    X_test, y_test, test_samples = base.load_dataset(test_files, psbatchsize=1, limitsamplesize=10, save=False)
    logger.info(f"Loaded dataset X_test:{X_test.shape} y:{y_test.shape}")
    X_noise = load_noise()
    logger.info(f"Loaded dataset X_noise:{X_noise.shape}")
    run(X_train, train_samples, X_noise, X_test, test_samples, y_train, y_test, args.epochs, args.trainingbatchsize, lr=0.0019437504084241922, kernel1=269, kernel2=135, dropout=0.5, input_points_to_use=args.input_points_to_use, showplots=args.interactive)

elif args.runmode == "optimize":
    logger.info("Starting hyperparameter optimization with Optuna")
    
    # Load data
    if args.use_saved_los_data:
        X_train, y_train, train_samples = load_dataset_from_pkl()
    else:
        X_train, y_train, train_samples = base.load_dataset(train_files, psbatchsize=args.psbatchsize, limitsamplesize=args.limitsamplesize,save=False)
        
    X_noise = load_noise()

    X_test, y_test, test_samples = base.load_dataset(test_files, psbatchsize=1, limitsamplesize=10, save=False)

    # Create study object
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)  # Adjust n_trials as needed

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
