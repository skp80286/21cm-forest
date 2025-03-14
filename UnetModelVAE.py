import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F 
import numpy as np

class UnetModelVAE(nn.Module):
    def __init__(self, input_size, input_channels, output_size, dropout=0.2, step=4, latent_dim=512):
        super(UnetModelVAE, self).__init__()
        self.input_size = input_size
        self.step = step

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

        # Latent space
        self.flatten = nn.Flatten()

        # Calculate the correct input size for the linear layer
        # Assuming input dimensions (depth, height, width) are 64x64x64
        # After 4 pooling operations with stride 2, the size will be reduced by a factor of 2^4 = 16
        self.flat_size = 256 * (input_size // (step ** 3))
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flat_size)

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
        
        # Latent space
        latent_input = self.flatten(enc3)
        mu = self.fc_mu(latent_input)
        logvar = self.fc_logvar(latent_input)
        z = self.reparameterize(mu, logvar)

        # Decode from latent space
        # Decode from latent space
        dec1 = self.fc_dec(z).view(-1, 256, int(self.input_size // (self.step ** 3)), 1) 
        # Decoder with skip connections
        dec1 = self.dec1(dec1)  # Changed from enc3 to dec1
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
        return out, mu, logvar

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

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss