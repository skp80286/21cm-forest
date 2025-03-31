import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class CnnInferenceModel(nn.Module):
    def __init__(self, input_size, input_channels, output_size, dropout=0.2, step=4):
        super(CnnInferenceModel, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 5, padding=2),
            nn.BatchNorm1d(64),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(step)
        )

        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(step)
        )

        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, 5, padding=2),
            nn.BatchNorm1d(256),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.BatchNorm1d(256),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(step)
        )

        self.enc4 = nn.Sequential(
            nn.Conv1d(256, 512, 5, padding=2),
            nn.BatchNorm1d(512),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(512, 512, 3, padding=1),
            nn.BatchNorm1d(512),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(step)
        )

        # Dense layers for parameter inference
        self.dense1 = nn.Linear(512 * (input_size // (step ** 4)), 512)  # Adjust input size based on pooling
        self.dense2 = nn.Linear(512, 128)  # Adjust input size based on pooling
        self.dense3 = nn.Linear(128, 2) # Output 2 parameters

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
        enc4 = self.enc4(enc3)
        #print(f"After enc4: {enc4.shape}")        
        
        # Pass through the dense layers for parameters extraction
        dense_out = self.dense1(enc4.view(enc4.size(0), -1))  # Flatten for dense layer
        dense_out = nn.Tanh()(dense_out)
        dense_out = self.dense2(dense_out)
        dense_out = nn.ReLU()(dense_out)
        dense_out = self.dense3(dense_out)
        dense_out = nn.ReLU()(dense_out)
        #print(f"Output of parameters extraction network: {dense_out.shape}")
        return dense_out

    def predict(self, X_test):
        with torch.no_grad():
            # Test the model
            print(f"Testing prediction")
            y_pred = self(X_test)
            return y_pred.detach().cpu().numpy()


    def save_model(self, file_path):
        """Save the model to a file."""
        torch.save(self.state_dict(), file_path)  # Save the model's state_dict

    def load_model(self, file_path):
        """Load the model from a file."""
        self.load_state_dict(torch.load(file_path))  # Load the model's state_dict
        self.eval()  # Set the model to evaluation mode
