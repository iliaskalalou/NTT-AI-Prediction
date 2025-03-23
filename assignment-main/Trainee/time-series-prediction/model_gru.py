import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

class GRUMultivariate(nn.Module):
    """
    Defines a multivariate GRU model for time series forecasting.
    Takes in sequences with 'input_size' features and outputs one forecast value.
    """
    def __init__(self, input_size, hidden_size=100, num_layers=2, output_size=1):
        super(GRUMultivariate, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):

        h0 = torch.zeros(2, x.size(0), 100).to(x.device)
        out, _ = self.gru(x, h0)

        return self.fc(out[:, -1, :])

def create_sequences(data, seq_length):
    """
    Splits the time series 'data' into sequences of length 'seq_length'.
    Returns the sequences (X) and the target values (y).
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)

def train_and_evaluate_gru(series, seq_length, train_ratio, num_epochs, lr=0.001, device=None):
    """
    Trains and evaluates the GRUMultivariate model on the given time series data.

    Parameters:
        series      : np.array, the time series (e.g., closing prices)
        seq_length  : int, length of sequences to create from the time series
        train_ratio : float, ratio of data used for training (e.g., 0.60 means 60% train, 40% test)
        num_epochs  : int, number of epochs to train the GRU model
        lr          : float, learning rate for the optimizer
        device      : optional torch device (e.g., "cuda" or "cpu")

    Returns:
        y_test      : np.array, actual target values from the test set
        predictions : np.array, model predictions on the test set
        rmse        : float, Root Mean Squared Error of the predictions
        mae         : float, Mean Absolute Error of the predictions
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X, y = create_sequences(series, seq_length)
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    X_test_tensor  = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor  = torch.tensor(y_test, dtype=torch.float32)
    
    model = GRUMultivariate(input_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in tqdm(range(num_epochs), desc="Training GRU"):
        inputs = X_train_tensor.to(device)
        targets = y_train_tensor.to(device).unsqueeze(1)  # Shape: [batch, 1]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor.to(device)).cpu().numpy().flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    return y_test, predictions, rmse, mae


