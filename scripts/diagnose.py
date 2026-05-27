"""Diagnóstico rápido do pipeline ML."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np

data = joblib.load('models/trained_models.pkl')
print("Keys:", list(data.keys()))
print(f"Train size: {data.get('train_size')}, Test size: {data.get('test_size')}")
print()

for name, info in data['models'].items():
    acc = info['accuracy']
    f1 = info['f1']
    rps_val = info['rps']
    n_feats = len(info['feature_columns'])
    print(f"{name}: Acc={acc:.4f}, F1={f1:.4f}, RPS={rps_val:.4f}, Features={n_feats}")
    print(f"  Cols: {info['feature_columns'][:5]}...")

print()
print("Seasonal results:")
for season, models in data.get('seasonal_results', {}).items():
    print(f"\n  {season}:")
    for name, m in models.items():
        print(f"    {name}: Acc={m['accuracy']:.4f}, Prec={m['precision']:.4f}, Rec={m['recall']:.4f}, F1={m['f1']:.4f}")
