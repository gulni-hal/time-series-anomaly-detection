"""
Derin Öğrenme Modelleri: LSTM, GRU, 1D-CNN
"""

import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


SEQ_LEN = 30  # Sliding window uzunluğu


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = SEQ_LEN):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def to_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.sigmoid(self.fc(out[:, -1, :])).squeeze()


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, layers, batch_first=True, dropout=dropout)
        self.fc  = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return torch.sigmoid(self.fc(out[:, -1, :])).squeeze()


class CNN1DModel(nn.Module):
    def __init__(self, input_size, seq_len=SEQ_LEN):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return torch.sigmoid(self.fc(self.conv(x).squeeze(-1))).squeeze()


def get_model(name: str, input_size: int):
    name = name.upper()
    if name == "LSTM":  return LSTMModel(input_size)
    if name == "GRU":   return GRUModel(input_size)
    if name == "CNN1D": return CNN1DModel(input_size)
    raise ValueError(f"Bilinmeyen model: {name}")


def train_model(model, train_loader, val_loader, cfg: dict, seed: int):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    best_val  = float('inf')
    patience  = 0
    t0        = time.perf_counter()

    for epoch in range(cfg["training"]["max_epochs"]):
        model.train()
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                val_losses.append(criterion(model(Xb), yb).item())
        avg_val = np.mean(val_losses)

        if avg_val < best_val:
            best_val = avg_val
            patience = 0
        else:
            patience += 1
            if patience >= cfg["training"]["early_stopping_patience"]:
                print(f"    Early stopping @ epoch {epoch+1}")
                break

    return model, time.perf_counter() - t0


def evaluate_model(model, test_loader):
    model.eval()
    preds, ys = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for Xb, yb in test_loader:
            preds.extend(model(Xb).numpy())
            ys.extend(yb.numpy())
    return np.array(preds), np.array(ys), time.perf_counter() - t0
