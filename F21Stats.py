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
    def calculate_stats_torch(X, y, kernel_sizes):
            # Validate X dimensions
        if not isinstance(X, (np.ndarray, torch.Tensor)) or len(X.shape) != 2:
            raise ValueError("X must be a 2-dimensional array or tensor")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X dimensions must be non-zero")

        # Validate y dimensions
        if not isinstance(y, (np.ndarray, torch.Tensor)) or len(y.shape) != 2:
            raise ValueError("y must be a 2-dimensional array or tensor")
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"First dimension of y ({y.shape[0]}) must match first dimension of X ({X.shape[0]})")
        if y.shape[1] != 2:
            raise ValueError(f"Second dimension of y must be 2, got {y.shape[1]}")

        # Validate kernel_sizes
        if not isinstance(kernel_sizes, (list, tuple, np.ndarray)):
            raise ValueError("kernel_sizes must be a list, tuple, or array")
        if not all(isinstance(k, (int, np.integer)) and k > 0 for k in kernel_sizes):
            raise ValueError("kernel_sizes must contain positive integers")


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

            if i < 5: F21Stats.logger.info(f'Stats for xHI={y[i, 0]} logfx={y[i, 1]}, kernel_size={kernel_size} = {row}')
        
        return np.array(stat_calc)

