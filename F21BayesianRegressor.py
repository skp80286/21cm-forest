import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class BayesianRegressor(nn.Module):
    def __init__(self, input_size, dropout_rate=0.5):
        super(BayesianRegressor, self).__init__()
        self.fc1_mean = nn.Linear(input_size, 64)  # First dense layer mean
        self.fc1_logvar = nn.Linear(input_size, 64)  # First dense layer log variance
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout layer after first dense layer
        
        self.fc2_mean = nn.Linear(64, 32)           # Second dense layer mean
        self.fc2_logvar = nn.Linear(64, 32)         # Second dense layer log variance
        self.dropout2 = nn.Dropout(dropout_rate)  # Dropout layer after second dense layer
        
        self.fc3_mean = nn.Linear(32, 2)            # Output layer mean (2 values: xHI and logfX)
        self.fc3_logvar = nn.Linear(32, 2)          # Output layer log variance

        self.relu = nn.ReLU()                       # Activation function

    def forward(self, x):
        # First layer
        mean1 = self.fc1_mean(x)
        logvar1 = self.fc1_logvar(x)
        std1 = torch.exp(0.5 * logvar1)  # Standard deviation
        eps1 = torch.randn_like(std1)     # Random noise
        x1 = mean1 + eps1 * std1          # Sampled output

        x1 = self.relu(x1)
        x1 = self.dropout1(x1)  # Apply dropout after first dense layer

        # Second layer
        mean2 = self.fc2_mean(x1)
        logvar2 = self.fc2_logvar(x1)
        std2 = torch.exp(0.5 * logvar2)
        eps2 = torch.randn_like(std2)
        x2 = mean2 + eps2 * std2

        x2 = self.relu(x2)
        x2 = self.dropout2(x2)  # Apply dropout after second dense layer

        # Output layer
        mean3 = self.fc3_mean(x2)
        logvar3 = self.fc3_logvar(x2)
        std3 = torch.exp(0.5 * logvar3)

        return mean3, std3  # Return mean and standard deviation for predictions

    def fit(self, X_train, y_train, epochs=100, learning_rate=0.001):
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)

        # Define optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()  # Zero the gradients
            mean, std = self(X_train_tensor)  # Forward pass

            # Calculate the negative log likelihood
            dist = Normal(mean, std)
            nll = -dist.log_prob(y_train_tensor).mean()  # Negative log likelihood

            nll.backward()  # Backward pass
            optimizer.step()  # Update weights

            if (epoch + 1) % 10 == 0:  # Print every 10 epochs
                print(f'Epoch [{epoch + 1}/{epochs}], NLL: {nll.item():.4f}')

    def predict(self, X_test):
        # Convert numpy array to PyTorch tensor
        X_test_tensor = torch.FloatTensor(X_test)
        with torch.no_grad():  # No need to track gradients
            mean, std = self(X_test_tensor)  # Forward pass
        return mean.numpy(), std.numpy()  # Convert back to numpy array

    def save_model(self, file_path):
        # Save the model state dictionary
        torch.save(self.state_dict(), file_path)
        print(f'Model saved to {file_path}')

    def load_model(self, file_path):
        # Load the model state dictionary
        self.load_state_dict(torch.load(file_path))
        self.eval()  # Set the model to evaluation mode
        print(f'Model loaded from {file_path}')