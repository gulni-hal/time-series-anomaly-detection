import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(LSTMAutoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Encoder (Kodlayıcı)
        self.encoder = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder (Çözücü)
        self.decoder = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=input_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, x):
        # x boyutu: (batch_size, sequence_length, input_dim)
        encoded_output, (hidden, cell) = self.encoder(x)
        
        # Decoder'a giriş olarak encoder'ın son durumunu (hidden state) serinin uzunluğu kadar kopyalıyoruz
        seq_len = x.shape[1]
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        
        decoded_output, _ = self.decoder(decoder_input)
        return decoded_output

class CNN1DAutoencoder(nn.Module):
    def __init__(self, input_dim, sequence_length, hidden_dim=64):
        super(CNN1DAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_channels=hidden_dim, out_channels=input_dim, kernel_size=3, padding=1),
            nn.Sigmoid() # Eğer veri 0-1 arası ölçeklendiyse Sigmoid, değilse kaldırılabilir
        )

    def forward(self, x):
        # CNN1D, (batch, channels, length) formatı bekler. Girdiyi (batch, length, input_dim)'den çeviriyoruz
        x = x.transpose(1, 2)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Çıktıyı tekrar orijinal (batch, length, input_dim) formatına döndür
        return decoded.transpose(1, 2)