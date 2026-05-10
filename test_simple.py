import sys, numpy as np, torch
sys.path.insert(0, '.')
from src.pipeline.data_loader import load_config, prepare_wadi
from src.models.deep_learning import make_sequences, to_loader

cfg = load_config('configs/default_config.yaml')
cfg['training']['max_epochs'] = 5
data = prepare_wadi(cfg)

print('X_train shape:', data['X_train'].shape)
print('y_train anomali:', data['y_train'].sum())

X_tr, y_tr = make_sequences(data['X_train'], data['y_train'], seq_len=10)
print('Sequence shape:', X_tr.shape)
print('Sequence anomali:', y_tr.sum())

# Basit model testi
import torch.nn as nn
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(10 * data['X_train'].shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)
Xb = torch.tensor(X_tr[:32])
out = model(Xb)
print('Output range:', out.min().item(), '-', out.max().item())
