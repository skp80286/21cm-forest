import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F 
import numpy as np

class UnetModel(nn.Module):
    def __init__(self, input_size, input_channels, output_size, dropout=0.2, step=4, latent_dim=256):
        super(UnetModel, self).__init__()
        self.input_size = input_size
        self.step = step

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(step)
        )

        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(step)
        )

        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(step)
        )

        self.enc4 = nn.Sequential(
            nn.Conv1d(256, 512, 5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(512, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(step)
        )

        # Latent space
        self.flatten = nn.Flatten()
        self.flat_size = 512 * (input_size // (step ** 4))
        
        
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flat_size)

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(512, 256, step, stride=step, output_padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(512, 128, step, stride=step, output_padding=0),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(256, 64, step, stride=step, output_padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(128, 32, step, stride=step, output_padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Final layer
        channels = 32
        self.final = nn.Sequential(
            nn.Conv1d(channels, 1, 1),
            nn.Flatten()
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        
        # Latent space
        latent_input = self.flatten(enc4)
        mu = self.fc_mu(latent_input)
        logvar = self.fc_logvar(latent_input)
        z = self.reparameterize(mu, logvar)

        # Decode from latent space
        dec1 = self.fc_dec(z).view(-1, 512, int(self.input_size // (self.step ** 4)))
        
        # Decoder with skip connections
        dec1 = self.dec1(dec1)
        dec1 = torch.cat([dec1, enc3], dim=1)
        
        dec2 = self.dec2(dec1)
        dec2 = torch.cat([dec2, enc2], dim=1)
        
        dec3 = self.dec3(dec2)
        dec3 = torch.cat([dec3, enc1], dim=1)
        
        dec4 = self.dec4(dec3)
        
        out = self.final(dec4)
        return out, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_denoised_signal(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def get_latent_features(self, x):
        self.eval()
        with torch.no_grad():
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            
            enc1 = self.enc1(x)
            enc2 = self.enc2(enc1)
            enc3 = self.enc3(enc2)
            enc4 = self.enc4(enc3)
            
            latent_input = self.flatten(enc4)
            mu = self.fc_mu(latent_input)

            return mu.cpu().numpy()

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))
        self.eval()

    def vae_loss(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss