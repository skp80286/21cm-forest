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

                skew5 = torch.mean((centered_x / (std.unsqueeze(1) + 1e-8)) ** 5)
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
        Calculate the bispectrum of 1-dimensional data.
        
        Parameters:
        - data: 1D array-like input data.
        - nfft: Number of points for FFT. If None, defaults to the length of data.
        
        Returns:
        - bispectrum: 2D array representing the bispectrum.
        """
        if nfft is None:
            nfft = len(data)
        
        # Perform FFT
        fft_data = np.fft.fft(data, n=nfft)
        bispectrum = np.zeros((nfft, nfft), dtype=complex)

        # Calculate bispectrum
        for i in range(nfft):
            for j in range(nfft):
                bispectrum[i, j] = fft_data[i] * np.conj(fft_data[j]) * fft_data[i + j] if (i + j) < nfft else 0

        return np.abs(bispectrum)

