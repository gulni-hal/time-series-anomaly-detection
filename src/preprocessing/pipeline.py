import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=1) # Otomata için veriyi tek boyuta düşürür
        
    def fit_transform(self, X_train):
        """
        Sadece eğitim verisi üzerinde fit işlemi yapar ve uygular.
        Derin öğrenme için ölçeklenmiş veriyi, Otomata için PCA uygulanmış 1D veriyi döner.
        """
        # Veriyi standartlaştır
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Otomata için veriyi 1 boyuta (PC1) indirge
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        
        return X_train_scaled, X_train_pca
        
    def transform(self, X):
        """
        Eğitim verisinden öğrenilen parametreleri kullanarak Validation ve Test verisini dönüştürür.
        Böylece Data Leakage (Veri sızıntısı) engellenmiş olur.
        """
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        return X_scaled, X_pca