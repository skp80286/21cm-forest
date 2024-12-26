import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import numpy as np
import logging
import F21DataLoader as dl

class F21Stats:
    logger = logging.getLogger(__name__)

    @staticmethod 
    def load_dataset_for_stats(datafiles, limitsamplesize):
        F21Stats.logger.info(f"Started data loading for stats.")
        processor = dl.F21DataLoader(max_workers=8, psbatchsize=1, limitsamplesize=limitsamplesize, skip_ps=True)

        # Process all files and get results
        results = processor.process_all_files(datafiles)
        F21Stats.logger.info(f"Finished data loading.")
        # Access results
        all_los = results['los']
        return all_los


    @staticmethod
    def calculate_stats_torch(X, y=None, kernel_sizes=[268]):
            # Validate X dimensions
        if not isinstance(X, (np.ndarray, torch.Tensor)) or len(X.shape) != 2:
            raise ValueError(f"X must be a 2-dimensional array or tensor, got {type(X).__name__} with shape {X.shape if hasattr(X, 'shape') else 'N/A'}")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"X dimensions must be non-zero, got shape {X.shape if hasattr(X, 'shape') else 'N/A'}")
        
        # Validate kernel_sizes
        if not isinstance(kernel_sizes, (list, tuple, np.ndarray)):
            raise ValueError("kernel_sizes must be a list, tuple, or array")
        if not all(isinstance(k, (int, np.integer)) and k > 0 for k in kernel_sizes):
            raise ValueError("kernel_sizes must contain positive integers")

        #print(y)
        stat_calc = []

        for i,x in enumerate(X):
            row = []
            tensor_1d = torch.tensor(x)
            # Pad the tensor if length is not divisible by 3
            total_mean = torch.mean(tensor_1d)
            total_std = torch.std(tensor_1d, unbiased=False)
            #total_centered_x = tensor_1d - total_mean
            #total_skewness = torch.mean((total_centered_x / (total_std)) ** 3)

            row += [total_mean, total_std] # total_skewness

            for kernel_size in kernel_sizes:
                padding_needed = kernel_size - len(tensor_1d) % kernel_size
                if padding_needed > 0:
                    tensor_1d = torch.nn.functional.pad(tensor_1d, (0, padding_needed))
                
                tensor_2d = tensor_1d.view(-1,kernel_size)

                means = torch.mean(tensor_2d, dim=1)
                std = torch.std(tensor_2d, dim=1, unbiased=False)

                centered_x = tensor_2d - means.unsqueeze(1)
                skewness = torch.mean((centered_x / (std.unsqueeze(1) + 1e-8)) ** 3, dim=1)

                mean_skew = torch.mean(skewness)
                std_skew = torch.std(skewness, unbiased=False)
                
                centered_skew = skewness - mean_skew
                skew2 = torch.mean((centered_skew / (std_skew.unsqueeze(0) + 1e-8)) ** 3)
                        
                min_skew = torch.min(skewness)

                #skew5 = torch.mean((centered_x / (std.unsqueeze(1) + 1e-8)) ** 5)
                #skew7 = torch.mean((centered_x / (std.unsqueeze(1) + 1e-8)) ** 7)

                row += [mean_skew.item(), std_skew.item(), skew2.item(), min_skew.item()]
            
            stat_calc.append(row)
            label = "Stats "
            if y is not None and len(y) > 0:
                if len(y.shape) == 1: label = f"Stats for xHI={y[0]} logfx={y[1]}"
                else: label = f"Stats for xHI={y[i, 0]} logfx={y[i, 1]}"

            if False: F21Stats.logger.info(f'{label}, kernel_size={kernel_size} Stats={row}')
        
        return np.array(stat_calc)
    
    @staticmethod
    def calculate_bispectrum(data, nfft=None):
        """
        Calculate the bispectrum of 2-dimensional data using PyTorch.
        
        Parameters:
        - data: 2D array-like input data.
        - nfft: Number of points for FFT. If None, defaults to the length of each row of data.
        
        Returns:
        - bispectrum_mean: 2D tensor representing the mean bispectrum across rows.
        """
        if nfft is None:
            nfft = data.shape[1]  # Use the length of each row
        
        # Convert data to a PyTorch tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Initialize a tensor to accumulate bispectrum results
        bispectrum_sum = torch.zeros(nfft * nfft, dtype=torch.complex64)

        # Calculate bispectrum for each row
        for row in data_tensor:
            fft_data = torch.fft.fft(row, n=nfft)
            for i in range(nfft):
                for j in range(nfft):
                    if (i + j) < nfft:
                        bispectrum_sum[i*nfft + j] += fft_data[i] * torch.conj(fft_data[j]) * fft_data[i + j]

        # Calculate the mean bispectrum across rows
        bispectrum_mean = bispectrum_sum / data_tensor.shape[0]
        bispectrum_mean = torch.abs(bispectrum_mean)
        bispectrum_mean_np = bispectrum_mean.numpy()
        return bispectrum_mean_np
    
def logbin_power_spectrum_by_k(ks, ps, silent=True):
    if not silent: F21Stats.logger.info(f"logbin_power_spectrum_by_k: Shapes: {ks.shape} {ps.shape}")
    if not silent: F21Stats.logger.info(f"logbin_power_spectrum_by_k: original ks: {ks[0,:5]} .. {ks[0,-5:]}")
    if not silent: F21Stats.logger.info(f"original ps: {ps[0,:5]}..{ps[0,-5:]}")

    d_log_k_bins = 0.25
    log_k_bins = np.arange(-7.0-d_log_k_bins/2.,-3.+d_log_k_bins/2.,d_log_k_bins)

    k_bins = np.power(10.,log_k_bins)
    k_bins_cent = np.power(10.,log_k_bins+d_log_k_bins/2.)[:-1]
    if not silent: F21Stats.logger.info(k_bins_cent)

    binlist=np.zeros((ps.shape[0], len(k_bins_cent)))
    pslist=np.zeros((ps.shape[0], len(k_bins_cent)))
    for i, (row_ks, row_ps) in enumerate(zip(ks, ps)):
      for l in range(len(k_bins_cent)):
        mask = (row_ks >= k_bins[l]) & (row_ks < k_bins[l+1])
        # If any values fall in this bin, take their mean
        if np.any(mask):
            pslist[i,l] = np.mean(row_ps[mask])
        else:
            pslist[i,l] = 0.
        binlist[i,l] = k_bins_cent[l]

    if not silent: F21Stats.logger.info(f"logbin_power_spectrum_by_k: final ks: {binlist[0,:5]}..{binlist[0,-5:]}")
    if not silent: F21Stats.logger.info(f"final ps: {pslist[0,:5]}..{pslist[0,-5:]}")
    return binlist, pslist

def logbin_power_spectrum_by_k_flex(ks, ps, ps_bins_to_make, perc_ps_bins_to_use):
    num_bins = ps_bins_to_make*perc_ps_bins_to_use//100
    
    min_log_k = None
    if (ks[0] > 0): min_log_k = np.log10(ks[0])
    else: min_log_k = np.log10(ks[1]/np.sqrt(10))
    max_log_k = np.log10(ks[-1])

    log_bins = np.linspace(min_log_k, max_log_k, ps_bins_to_make+1)
    #print(f"log_bins: {log_bins}")
    bins = np.power(10, log_bins)
    #print(f"bins: {bins}")
    # widths = (bins[1:] - bins[:-1])
    #print(f"widths: {widths}")
    log_centers = 0.5*(log_bins[:-1]+log_bins[1:])
    bin_centers = np.power(10, log_centers)
    #print(f"bin_centers: {bin_centers}")
    pslist=np.zeros((ps.shape[0], ps_bins_to_make))
    # Calculate histogram
    for i, (p) in enumerate(ps):
        hist = np.histogram(ks, bins=bins, weights=p)
        #print(f"hist: {hist}")
        # normalize by bin width
        #hist_norm = hist[0]/widths
        #print(f"hist_norm: {hist_norm}")
        pslist[i,:] = hist[0]

    return bin_centers, pslist[:,:num_bins]
