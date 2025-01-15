import torch
import torch.nn as nn
import torch.optim as optim

class NNRegressor(nn.Module):
    def __init__(self, input_size, dropout_rate=0.3):
        super(NNRegressor, self).__init__()
        self.fc0 = nn.Linear(input_size, 256)  # Oth dense layer
        self.dropout0 = nn.Dropout(dropout_rate)  # Dropout layer after first dense layer
        self.fc1 = nn.Linear(256, 128)  # First dense layer
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout layer after first dense layer
        self.fc2 = nn.Linear(128, 64)           # Second dense layer
        self.dropout2 = nn.Dropout(dropout_rate)  # Dropout layer after second dense layer
        self.fc3 = nn.Linear(64, 2)            # Output layer (2 values: xHI and logfX)
        self.tanh = nn.Tanh()                   # Activation function1
        self.relu = nn.ReLU()                   # Activation function2

    def forward(self, x):
        x = self.fc0(x)
        x = self.tanh(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)  # Apply dropout after first dense layer
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)  # Apply dropout after second dense layer
        x = self.fc3(x)
        return x

    def fit(self, X_train, y_train, epochs=2000, learning_rate=0.0001):
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        #optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()  # Zero the gradients
            outputs = self(X_train_tensor)  # Forward pass
            loss = criterion(outputs, y_train_tensor)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            if (epoch + 1) % 10 == 0:  # Print every 10 epochs
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    def predict(self, X_test):
        # Convert numpy array to PyTorch tensor
        X_test_tensor = torch.FloatTensor(X_test)
        with torch.no_grad():  # No need to track gradients
            predictions = self(X_test_tensor)  # Forward pass
        return predictions.numpy()  # Convert back to numpy array
    
    def save_model(self, file_path):
        # Save the model state dictionary
        torch.save(self.state_dict(), file_path)
        print(f'Model saved to {file_path}')

    def load_model(self, file_path):
        # Load the model state dictionary
        self.load_state_dict(torch.load(file_path))
        self.eval()  # Set the model to evaluation mode
        print(f'Model loaded from {file_path}')