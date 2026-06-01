import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

MODEL_TIMEOUT_SECONDS = 300  # Model başına maksimum eğitim süresi (5 dakika)

def train_deep_learning(model, train_loader, val_loader, config_dl, y_train=None):
    """Derin öğrenme modelini gözetimli sınıflandırma olarak eğitir."""
    # Sınıf dengesizliğini düzeltmek için pos_weight hesapla
    if y_train is not None and len(y_train) > 0:
        n_neg = float(np.sum(y_train == 0))
        n_pos = float(np.sum(y_train == 1))
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)])
    else:
        pos_weight = torch.tensor([1.0])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=config_dl.get('learning_rate', 1e-3))

    epochs = config_dl.get('max_epochs', 50)
    patience = config_dl.get('early_stopping_patience', 5)

    best_loss = float('inf')
    patience_counter = 0
    t_start = time.time()

    for epoch in range(epochs):
        # Zaman aşımı kontrolü: 5 dakika geçtiyse eğitimi durdur
        if time.time() - t_start > MODEL_TIMEOUT_SECONDS:
            print(f"  Model timeout ({MODEL_TIMEOUT_SECONDS}s). Epoch: {epoch+1}")
            break

        model.train()
        for batch in train_loader:
            batch_x, batch_y = batch[0], batch[1] # Artık etiketleri (y) de alıyoruz

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output.squeeze(-1), batch_y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch_x, batch_y = batch[0], batch[1]
                output = model(batch_x)
                loss = criterion(output.squeeze(-1), batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping tetiklendi. Epoch: {epoch+1}")
                break

    return model

def train_automata(automata_model, X_train_pca):
    """Otomata modeline PCA uygulanmış 1D eğitim verisini vererek geçiş olasılıklarını fit eder."""
    # X_train_pca verisi genelde (N, 1) şeklindedir, 1D array'e çevirip eğitelim
    data_1d = X_train_pca.flatten()
    automata_model.fit(data_1d)
    return automata_model