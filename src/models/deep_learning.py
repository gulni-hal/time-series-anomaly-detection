import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        # Batch_first=True -> (batch, seq_len, features)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)  # Tek çıktı: 0 veya 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)
        out = out[:, -1, :]  # Sadece son zaman adımının çıktısını al
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)  # logit — sigmoid BCEWithLogitsLoss içinde uygulanır
        return out


class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.2):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)  # Tek çıktı: 0 veya 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, hidden = self.gru(x)
        out = out[:, -1, :]  # Sadece son zaman adımının çıktısını al
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)  # logit — sigmoid BCEWithLogitsLoss içinde uygulanır
        return out


class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim, seq_len):
        super(CNN1DClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        # MaxPool sonrası boyut yarıya düşer
        pooled_len = seq_len // 2
        self.fc1 = nn.Linear(64 * pooled_len, 50)
        self.fc2 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # PyTorch Conv1D (batch, channels, seq_len) bekler
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # logit — sigmoid BCEWithLogitsLoss içinde uygulanır
        return x
