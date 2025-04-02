## PS Based noise subtraction with XGB Regression model for parameter inference

import F21DataLoader as dl
import f21_predict_base as base
import plot_results as pltr
import Scaling
import PS1D
import F21Stats as f21stats

import numpy as np
import sys
import torch

def load_dataset(datafiles, psbatchsize, limitsamplesize, max_workers=8, skip_ps=False):
    # Lists to store combined data
    all_params = []
    # Create processor with desired number of worker threads
    processor = dl.F21DataLoader(max_workers=max_workers, psbatchsize=psbatchsize, limitsamplesize=limitsamplesize, skip_ps=skip_ps)

    # Process all files and get results
    results = processor.process_all_files(datafiles)
    logger.info(f"Finished data loading.")
    # Access results
    keys = results['key']
    all_los = results['los']
    #los_samples = results['los_samples']
    params = results['params']
    all_ps = results['ps']
    all_ks = results['ks']


    logger.info(f"sample los:{all_los[0]}")
    logger.info(f"sample ps:{all_ps[0]}")
    logger.info(f"sample ks:{all_ks[0]}") 
    logger.info(f"sample params:{params[0]}")
    
    ks, ps = f21stats.logbin_power_spectrum_by_k(all_ks, all_ps, silent=False)

    # Combine all data
    logger.info(f"Los shape: {all_los.shape}")
    logger.info(f"PS shape: {ps.shape}")
    logger.info(f"KS shape: {ks.shape}")
    logger.info(f"Parameters shape: {params.shape}")
        
    return (ps, ks, params, keys)

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

def load_noise():
    X_noise = None
    noisefiles = base.get_datafile_list(type='noiseonly', args=args)
    ps_noise, ks_noise, _, _ = load_dataset(noisefiles, psbatchsize=1000, limitsamplesize=1000)
    logger.info(f"Loaded noise with shape: {ps_noise.shape}")
    return ps_noise


# main code starts here
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.determinisitc=True
torch.backends.cudnn.benchmark=False

parser = base.setup_args_parser()
args = parser.parse_args()

output_dir = base.create_output_dir(args=args)
logger = base.setup_logging(output_dir)

datafiles = base.get_datafile_list(type='noisy', args=args)
so_datafiles = base.get_datafile_list(type='signalonly', args=args)

test_points = [[-3,0.11], [-3,0.80], [-1,0.11], [-1,0.80], [-2,0.52]]
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

logger.info(f"Loading train dataset {len(train_files)}")
ps_train, ks_train, y_train, keys_train = load_dataset(train_files, psbatchsize=args.psbatchsize, limitsamplesize=args.limitsamplesize)
logger.info(f"Loaded datasets ps:{ps_train.shape} y_train:{y_train.shape}")
logger.info(f"Loading test dataset {len(test_files)}")
ps_test, ks_test, y_test, keys_test = load_dataset(test_files, psbatchsize=1, limitsamplesize=1000)
logger.info(f"Loaded datasets ps:{ps_test.shape} y_test:{y_test.shape}")

noise = load_noise()

# Noise subtraction
X_train = ps_train - noise
X_test = ps_test - noise

# Save X_train and X_test to CSV files
np.savetxt(f'{output_dir}/ps_train.csv', ps_train, delimiter=',')
np.savetxt(f'{output_dir}/X_train.csv', X_train, delimiter=',')
np.savetxt(f'{output_dir}/y_train.csv', y_train, delimiter=',')
np.savetxt(f'{output_dir}/ps_test.csv', ps_test, delimiter=',')
np.savetxt(f'{output_dir}/X_test.csv', X_test, delimiter=',')
np.savetxt(f'{output_dir}/y_test.csv', y_test, delimiter=',')
#np.savetxt(f'{output_dir}/keys_test.csv', keys_test, delimiter=',')
