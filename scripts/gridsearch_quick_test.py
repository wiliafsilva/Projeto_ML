#!/usr/bin/env python
"""Teste rápido do GridSearch - versão simplificada para validar o scorer"""

import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVC
from src.preprocessing import load_data
from src.feature_engineering import calculate_team_stats

class RPSScorer:
    """Scorer RPS customizado para GridSearch"""
    
    def __call__(self, estimator, X, y):
        """Calcula o RPS score"""
        y_pred_proba = estimator.predict_proba(X)
        y_true_onehot = np.eye(3)[y]
        y_true_cum = np.cumsum(y_true_onehot, axis=1)
        y_prob_cum = np.cumsum(y_pred_proba, axis=1)
        return -np.mean(np.sum((y_true_cum - y_prob_cum)**2, axis=1))

print("="*60)
print("TESTE RÁPIDO DO GRIDSEARCH")
print("="*60)

# Carregar dados
df = load_data('data/epl.csv')
features = calculate_team_stats(df)

# Usar apenas dados de treino
train = features[features['Season'] <= 2018]
X_train = train.drop(['Result','Season'], axis=1)
y_train = train['Result']

# Scorer e CV
rps_scorer_fn = RPSScorer()
tscv = TimeSeriesSplit(n_splits=3)  # Apenas 3 splits para ser mais rápido

print(f"\nDataset de treino: {len(X_train)} partidas")
print(f"Cross-validation: TimeSeriesSplit com 3 splits")

# GridSearch simplificado (apenas 2 combinações)
print("\n" + "="*60)
print("SVM - Teste Rápido (2 combinações)")
print("="*60)

svm_param_grid = {
    'C': [0.5, 1.0],
    'gamma': ['scale'],
    'kernel': ['rbf']
}

svm_grid = GridSearchCV(
    SVC(probability=True, random_state=42, class_weight='balanced'),
    svm_param_grid,
    cv=tscv,
    scoring=rps_scorer_fn,
    n_jobs=-1,
    verbose=2
)

svm_grid.fit(X_train, y_train)

print(f"\n✓ Melhores parâmetros SVM:")
for param, value in svm_grid.best_params_.items():
    print(f"  {param}: {value}")
print(f"✓ Melhor RPS (CV): {-svm_grid.best_score_:.4f}")

print("\n" + "="*60)
print("✓ TESTE CONCLUÍDO COM SUCESSO!")
print("="*60)
print("\nAgora você pode executar: python scripts\\gridsearch_advanced.py")
