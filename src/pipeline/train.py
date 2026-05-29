import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def train_deep_learning(model, train_loader, val_loader, config_dl):
    """Derin öğrenme modelini gözetimli sınıflandırma olarak eğitir."""
    criterion = nn.BCELoss() # HATA ÇÖZÜMÜ: Autoencoder yerine Sınıflandırıcı Kaybı
    optimizer = optim.Adam(model.parameters(), lr=config_dl.get('learning_rate', 1e-3))
    
    epochs = config_dl.get('max_epochs', 50) 
    patience = config_dl.get('early_stopping_patience', 5)
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch_x, batch_y = batch[0], batch[1] # Artık etiketleri (y) de alıyoruz
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output.squeeze(), batch_y) # Tahmin ile gerçek etiketi karşılaştır
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch_x, batch_y = batch[0], batch[1]
                output = model(batch_x)
                loss = criterion(output.squeeze(), batch_y)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
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