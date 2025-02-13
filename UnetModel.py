import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class UnetModel(nn.Module):
    def __init__(self, input_size, input_channels, output_size, dropout=0.2, step=4):
        super(UnetModel, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(input_channels, 16, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(step)
        )

        self.enc2 = nn.Sequential(
            nn.Conv1d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(step)
        )

        self.enc3 = nn.Sequential(
            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(step)
        )

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, step, stride=step, output_padding=0),
            nn.ReLU(),
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(64, 16, step, stride=step, output_padding=0),
            nn.ReLU(),
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(32, 8, step, stride=step, output_padding=0),
            nn.ReLU(),
        )

        # Final layer
        channels = input_channels + 8
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
        dec3 = torch.cat([dec3, x], dim=1)
                 
        #print(f"Before final: {dec4.shape}")
        out = self.final(dec3)

        # Concatenate parameter extraction output with decoder output
        # out = torch.cat((dense_out, out), dim=1)  # Concatenate along the feature dimension
        #print(f"Output shape after concatenation: {out.shape}")

        #print(f"Output shape: {out.shape}")
        return out

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
