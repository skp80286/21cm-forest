import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class UnetModel(nn.Module):
    def __init__(self, input_size, input_channels, output_size, dropout=0.2, step=4):
        super(UnetModel, self).__init__()

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

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, step, stride=step, output_padding=0),
            nn.BatchNorm1d(128),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(256, 64, step, stride=step, output_padding=0),
            nn.BatchNorm1d(64),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(128, 32, step, stride=step, output_padding=0),
            nn.BatchNorm1d(32),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Final layer
        channels = 32 #input_channels + 
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
        #dec3 = torch.cat([dec3, x], dim=1)
                 
        #print(f"Before final: {dec4.shape}")
        out = self.final(dec3)

        # Concatenate parameter extraction output with decoder output
        # out = torch.cat((dense_out, out), dim=1)  # Concatenate along the feature dimension
        #print(f"Output shape after concatenation: {out.shape}")

        #print(f"Output shape: {out.shape}")
        return out

    def get_denoised_signal(self, x):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            return self.forward(x)

    def get_latent_features(self, x):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            # Pass input to the first layer and get output from enc3
            #input_tensor = self.convert_to_pytorch_tensors(X_train, y_train, None, None)[0]
            if len(x.shape) == 2:
                x = x.unsqueeze(1)  # Add channel dimension
            # If input already has channels, it will remain unchanged
            #print(f"After unsqueeze: {x.shape}")        
            # Encoder
            enc1 = self.enc1(x)
            #print(f"After enc1: {enc1.shape}")        
            enc2 = self.enc2(enc1)
            #print(f"After enc2: {enc2.shape}")        
            enc3_output = self.enc3(enc2)
            #print(f"After enc3: {enc3.shape}")        
            enc3_flattened = enc3_output.view(enc3_output.size(0), -1).cpu().numpy()  # Flatten and convert to numpy
            return enc3_flattened

    def save_model(self, file_path):
        """Save the model to a file."""
        torch.save(self.state_dict(), file_path)  # Save the model's state_dict

    def load_model(self, file_path):
        """Load the model from a file."""
        self.load_state_dict(torch.load(file_path))  # Load the model's state_dict
        self.eval()  # Set the model to evaluation mode

    @staticmethod
    def aggregate_latent_data(params, latent_features, num_rows=10):
        # Create a dictionary to hold the aggregated results
        aggregated_data = {}
        
        for i in range(len(params)):
            # Create a key by combining both values in the row
            key = f"{params[i][0]:.2f}_{params[i][1]:.2f}"
            
            # If the key is not in the dictionary, initialize it
            if key not in aggregated_data:
                aggregated_data[key] = []
            
            # Append the corresponding latent feature to the key
            aggregated_data[key].append(latent_features[i])
        
        # Prepare results
        result_keys = []
        result_means = []
        
        for key, features in aggregated_data.items():
            # Aggregate in chunks of num_rows
            for start in range(0, len(features), num_rows):
                chunk = features[start:start + num_rows]
                result_keys.append(key)
                result_means.append(np.mean(chunk, axis=0))  # Mean across the chunk
        
        parsed_keys = np.array([[float(x) for x in key.split('_')] for key in result_keys])  # Parse keys into floats
        return parsed_keys, np.array(result_means)