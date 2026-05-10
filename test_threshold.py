import sys, numpy as np
sys.path.insert(0, '.')
from src.pipeline.data_loader import load_config, prepare_wadi
from src.models.deep_learning import get_model, make_sequences, to_loader, train_model, evaluate_model
from src.utils.metrics import compute_metrics

cfg = load_config('configs/default_config.yaml')
cfg['training']['max_epochs'] = 10
data = prepare_wadi(cfg)

X_tr, y_tr = make_sequences(data['X_train'], data['y_train'])
X_vl, y_vl = make_sequences(data['X_val'],   data['y_val'])
X_te, y_te = make_sequences(data['X_test'],  data['y_test'])

tr_l = to_loader(X_tr, y_tr, 32)
vl_l = to_loader(X_vl, y_vl, 32, shuffle=False)
te_l = to_loader(X_te, y_te, 32, shuffle=False)

pos_w = float((data['y_train']==0).sum()) / max(float((data['y_train']==1).sum()), 1)
model = get_model('LSTM', data['X_train'].shape[1])
model, _ = train_model(model, tr_l, vl_l, cfg, 42, pos_w)
preds, y_true, _ = evaluate_model(model, te_l)

print('Pred min:', round(float(preds.min()), 4))
print('Pred max:', round(float(preds.max()), 4))

for t in [0.01, 0.02, 0.03, 0.04, 0.05]:
    y_bin = (preds > t).astype(int)
    m = compute_metrics(y_true, y_bin)
    print('Esik=' + str(t) + ' F1=' + str(m['f1']) + ' R=' + str(m['recall']))
