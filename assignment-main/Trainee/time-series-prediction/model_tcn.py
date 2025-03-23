import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

class TemporalBlock(nn.Module):
    """
    Defines a single Temporal Block for the TCN.
    This block applies two dilated 1D convolutions with ReLU and dropout, and includes a residual connection.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.padding = padding
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.init_weights()
    
    def init_weights(self):
        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.normal_(self.conv2.weight, 0, 0.01)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0, 0.01)
    
    def forward(self, x):
        out = self.conv1(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        if res.size(2) > out.size(2):
            res = res[:, :, :out.size(2)]
        return out + res

class TCN(nn.Module):
    """
    Defines the Temporal Convolutional Network (TCN) model.
    It stacks multiple TemporalBlocks and applies a final fully-connected layer to generate the forecast.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size,
                                          stride=1, dilation=dilation_size,
                                          padding=padding, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)
    
    def forward(self, x):
        x = x.transpose(1, 2)  # Change shape from (batch, seq_length, num_inputs) to (batch, num_inputs, seq_length)
        y = self.network(x)
        y = y[:, :, -1]  # Use the output from the last time step
        return self.fc(y)

def create_sequences(data, seq_length):
    """
    Splits the time series 'data' into sequences of length 'seq_length'.
    
    Returns:
        X: Array of input sequences.
        y: Array of target values (next value after each sequence).
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i + seq_length])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)

def train_and_evaluate_tcn(series, seq_length, train_ratio, num_epochs, lr=0.001, device=None):
    """
    Trains and evaluates the TCN model on the given time series.
    
    Parameters:
      - series: np.array, the time series data (normalized or not)
      - seq_length: int, length of the sequences
      - train_ratio: float, proportion of data used for training
      - num_epochs: int, number of training epochs
      - lr: float, learning rate
      - device: torch.device (optional)
    
    Returns:
      - y_test: Actual target values from the test set
      - predictions: Forecasted values by the TCN model
      - rmse: Root Mean Squared Error
      - mae: Mean Absolute Error
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X, y = create_sequences(series, seq_length)
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    model = TCN(num_inputs=1, num_channels=[50, 50], kernel_size=2, dropout=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in tqdm(range(num_epochs), desc="Training TCN"):
        inputs = X_train_tensor.to(device)
        targets = y_train_tensor.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"TCN Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor.to(device)).cpu().numpy().flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    return y_test, predictions, rmse, mae
