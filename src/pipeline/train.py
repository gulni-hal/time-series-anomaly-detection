import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def train_deep_learning(model, train_loader, val_loader, config_dl):
    """Derin öğrenme modelini (LSTM/CNN) erken durdurma (Early Stopping) ile eğitir."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config_dl.get('learning_rate', 1e-3))
    
    epochs = config_dl['epochs']
    patience = config_dl['patience']
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            batch_x = batch[0] # HATA ÇÖZÜMÜ BURADA: DataLoader'dan gelen tuple'ın sadece girdisini al
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_x) # Autoencoder: Girdi ile Çıktı arasındaki farkı minimize et
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch_x = batch[0] # HATA ÇÖZÜMÜ BURADA
                output = model(batch_x)
                loss = criterion(output, batch_x)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        # Early Stopping Kontrolü
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # torch.save(model.state_dict(), 'best_model.pth') # Modeli kaydet
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping tetiklendi. Epoch: {epoch+1}")
                break
                
    return model

def train_automata(automata_model, X_train_pca):
    """Otomata modeline PCA uygulanmış 1D eğitim verisini vererek geçiş olasılıklarını fit eder."""
    # X_train_pca verisi genelde (N, 1) şeklindedir, 1D array'e çevirip eğitelim
    data_1d = X_train_pca.flatten()
    automata_model.fit(data_1d)
    return automata_model