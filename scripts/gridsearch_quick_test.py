#!/usr/bin/env python
"""
GridSearch Rápido - Teste de hiperparâmetros com grid reduzido
==============================================================

Versão rápida (5-10 min) do gridsearch_advanced.py.
Use para validar se o pipeline funciona antes de rodar o completo.
"""

import sys
import io
from pathlib import Path

# Forçar UTF-8 no Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.preprocessing import load_all_data
from src.feature_engineering import calculate_team_stats
from src.train_models import prepare_features_by_model

def rps(y_true, y_prob):
    """Ranked Probability Score"""
    y_true = y_true.astype(int)
    y_true_onehot = np.eye(3)[y_true]
    y_true_cum = np.cumsum(y_true_onehot, axis=1)
    y_prob_cum = np.cumsum(y_prob, axis=1)
    k_minus_1 = y_prob.shape[1] - 1 if y_prob.shape[1] > 1 else 1
    return np.mean(np.sum((y_true_cum - y_prob_cum)**2, axis=1)) / k_minus_1

class RPSScorer:
    def __call__(self, estimator, X, y):
        return -rps(y, estimator.predict_proba(X))


print("="*60)
print("GRIDSEARCH RÁPIDO - TESTE DE HIPERPARÂMETROS")
print("="*60)

# Carregar dados
df = load_all_data()
features = calculate_team_stats(df)

train = features[features['Season'] <= 2014].reset_index(drop=True)
test = features[features['Season'] > 2014].reset_index(drop=True)

rps_scorer_fn = RPSScorer()
tscv = TimeSeriesSplit(n_splits=3)  # 3 splits para ser mais rápido

# Class B features (SVM, RF, XGB)
df_train_b = prepare_features_by_model(train, 'SVM')
X_train_b = df_train_b.drop(['Result', 'Season'], axis=1)
y_train_b = df_train_b['Result']

# Class A features (NaiveBayes)
df_train_a = prepare_features_by_model(train, 'NaiveBayes')
X_train_a = df_train_a.drop(['Result', 'Season'], axis=1)
y_train_a = df_train_a['Result']

print(f"\nTreino: {len(train)} partidas | Teste: {len(test)} partidas")
print(f"CV: TimeSeriesSplit com 3 splits (rápido)")
print(f"Features B: {X_train_b.shape[1]} | Features A: {X_train_a.shape[1]}\n")

# ============================================================
# Grids reduzidos (teste rápido)
# ============================================================
grids = {
    'SVM': {
        'estimator': SVC(probability=True, random_state=42, class_weight='balanced'),
        'params': {
            'C': [0.1, 1.0, 5.0],
            'gamma': ['scale', 0.01],
            'kernel': ['rbf']
        },
        'X_train': X_train_b,
        'y_train': y_train_b,
        'features': 'B'
    },
    'RandomForest': {
        'estimator': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_leaf': [1, 4]
        },
        'X_train': X_train_b,
        'y_train': y_train_b,
        'features': 'B'
    },
    'XGBoost': {
        'estimator': XGBClassifier(eval_metric='mlogloss', random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8]
        },
        'X_train': X_train_b,
        'y_train': y_train_b,
        'features': 'B'
    },
    'NaiveBayes': {
        'estimator': GaussianNB(),
        'params': {
            'var_smoothing': [1e-9, 1e-7, 1e-5]
        },
        'X_train': X_train_a,
        'y_train': y_train_a,
        'features': 'A'
    },
}

results = {}

for name, config in grids.items():
    print(f"\n{'='*50}")
    print(f"[{name}] (Class {config['features']})")
    print(f"{'='*50}")
    
    grid = GridSearchCV(
        config['estimator'],
        config['params'],
        cv=tscv,
        scoring=rps_scorer_fn,
        n_jobs=-1,
        verbose=0
    )
    
    grid.fit(config['X_train'], config['y_train'])
    
    print(f"✓ Melhores parâmetros:")
    for param, value in grid.best_params_.items():
        print(f"  {param}: {value}")
    print(f"✓ Melhor RPS (CV): {-grid.best_score_:.4f}")
    
    results[name] = {
        'best_params': grid.best_params_,
        'best_rps': -grid.best_score_,
        'features': config['features']
    }

# ============================================================
# Resumo
# ============================================================
print("\n" + "="*60)
print("RESUMO - GRIDSEARCH RÁPIDO")
print("="*60)
print(f"\n{'Modelo':<18} {'Class':>6} {'RPS (CV)':>10}")
print("-" * 40)

for name, info in results.items():
    print(f"{name:<18} {info['features']:>6} {info['best_rps']:>10.4f}")

print("\n💡 Para resultados mais precisos, execute:")
print("   python scripts/gridsearch_advanced.py")
print("\n💡 Copie os melhores parâmetros para src/train_models.py")
print("   e execute 'python main.py' para retreinar.")
