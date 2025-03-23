import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import data_preprocessing as data_preprocessing 
from sklearn.metrics import mean_squared_error, mean_absolute_error

class LSTMForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMForecast, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialisation des états cachés et cellulaire
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # On prend la sortie du dernier pas temporel
        out = self.fc(out[:, -1, :])
        return out

def train_and_evaluate(series, seq_length, train_ratio, num_epochs, lr=0.001, device=None):
    """
    Entraîne et évalue le modèle LSTM sur la série temporelle donnée.
    Renvoie :
        - y_test : les vraies valeurs de test
        - predictions : les prédictions du modèle
        - rmse : l'erreur quadratique moyenne
        - mae : l'erreur absolue moyenne
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Création des séquences temporelles
    X, y = data_preprocessing.create_sequences(series, seq_length)
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Conversion en tenseurs PyTorch et ajout de la dimension feature
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # Initialisation du modèle, de la fonction de coût et de l'optimiseur
    model = LSTMForecast().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in tqdm(range(num_epochs), desc="Training LSTM"):
        optimizer.zero_grad()
        outputs = model(X_train_tensor.to(device))
        loss = criterion(outputs, y_train_tensor.to(device).unsqueeze(1))
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor.to(device)).cpu().numpy().flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    return y_test, predictions, rmse, mae

