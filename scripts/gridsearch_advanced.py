#!/usr/bin/env python
"""Script para GridSearch avançado com validação temporal"""

import sys
from pathlib import Path

# Adicionar o diretório raiz ao path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

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
        # Retorna negativo porque queremos minimizar RPS, mas GridSearch maximiza
        return -np.mean(np.sum((y_true_cum - y_prob_cum)**2, axis=1))
    
    def _score_func(self, *args, **kwargs):
        """Para compatibilidade com make_scorer"""
        return self.__call__(*args, **kwargs)


print("="*60)
print("GRIDSEARCH AVANÇADO - OTIMIZAÇÃO DE HIPERPARÂMETROS")
print("="*60)

# Carregar dados
df = load_data('data/epl.csv')
features = calculate_team_stats(df)

# Usar apenas dados de treino para GridSearch
train = features[features['Season'] <= 2018]
X_train = train.drop(['Result','Season'], axis=1)
y_train = train['Result']

# Sample weights para modelos que suportam
sample_weights = compute_sample_weight('balanced', y_train)

# Scorer customizado (RPS negativo para minimizar)
rps_scorer_fn = RPSScorer()

# Cross-validation temporal (evita data leakage)
tscv = TimeSeriesSplit(n_splits=5)

print(f"\nDataset de treino: {len(X_train)} partidas")
print(f"Cross-validation: TimeSeriesSplit com 5 splits")
print(f"Métrica de otimização: RPS (Ranked Probability Score)\n")

# ============================================================
# 1. SVM - GridSearch
# ============================================================
print("="*60)
print("[1] SVM - Otimizando hiperparâmetros")
print("="*60)

svm_param_grid = {
    'C': [0.1, 0.5, 1.0, 2.0, 5.0],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf']
}

print(f"Testando {len(svm_param_grid['C']) * len(svm_param_grid['gamma'])} combinações...")

svm_grid = GridSearchCV(
    SVC(probability=True, random_state=42, class_weight='balanced'),
    svm_param_grid,
    cv=tscv,
    scoring=rps_scorer_fn,
    n_jobs=-1,
    verbose=1
)

svm_grid.fit(X_train, y_train)

print(f"\n✓ Melhores parâmetros SVM:")
for param, value in svm_grid.best_params_.items():
    print(f"  {param}: {value}")
print(f"✓ Melhor RPS (CV): {-svm_grid.best_score_:.4f}")

# ============================================================
# 2. RandomForest - GridSearch
# ============================================================
print("\n" + "="*60)
print("[2] RandomForest - Otimizando hiperparâmetros")
print("="*60)

rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print(f"Testando {len(rf_param_grid['n_estimators']) * len(rf_param_grid['max_depth']) * len(rf_param_grid['min_samples_split']) * len(rf_param_grid['min_samples_leaf'])} combinações...")

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    rf_param_grid,
    cv=tscv,
    scoring=rps_scorer_fn,
    n_jobs=-1,
    verbose=1
)

rf_grid.fit(X_train, y_train)

print(f"\n✓ Melhores parâmetros RandomForest:")
for param, value in rf_grid.best_params_.items():
    print(f"  {param}: {value}")
print(f"✓ Melhor RPS (CV): {-rf_grid.best_score_:.4f}")

# ============================================================
# 3. XGBoost - GridSearch
# ============================================================
print("\n" + "="*60)
print("[3] XGBoost - Otimizando hiperparâmetros")
print("="*60)

xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

print(f"Testando {len(xgb_param_grid['n_estimators']) * len(xgb_param_grid['max_depth']) * len(xgb_param_grid['learning_rate']) * len(xgb_param_grid['subsample']) * len(xgb_param_grid['colsample_bytree'])} combinações...")

xgb_grid = GridSearchCV(
    XGBClassifier(eval_metric='mlogloss', random_state=42),
    xgb_param_grid,
    cv=tscv,
    scoring=rps_scorer_fn,
    n_jobs=-1,
    verbose=1
)

# XGBoost - fit sem sample_weight para compatibilidade com GridSearch
xgb_grid.fit(X_train, y_train)

print(f"\n✓ Melhores parâmetros XGBoost:")
for param, value in xgb_grid.best_params_.items():
    print(f"  {param}: {value}")
print(f"✓ Melhor RPS (CV): {-xgb_grid.best_score_:.4f}")

# ============================================================
# Salvar resultados
# ============================================================
print("\n" + "="*60)
print("SALVANDO RESULTADOS")
print("="*60)

best_models = {
    'SVM_optimized': {
        'model': svm_grid.best_estimator_,
        'params': svm_grid.best_params_,
        'cv_rps': -svm_grid.best_score_
    },
    'RandomForest_optimized': {
        'model': rf_grid.best_estimator_,
        'params': rf_grid.best_params_,
        'cv_rps': -rf_grid.best_score_
    },
    'XGBoost_optimized': {
        'model': xgb_grid.best_estimator_,
        'params': xgb_grid.best_params_,
        'cv_rps': -xgb_grid.best_score_
    }
}

joblib.dump(best_models, 'models/optimized_models.pkl')
print("✓ Modelos otimizados salvos em: models/optimized_models.pkl")

# Salvar resultados detalhados
results_df = pd.DataFrame({
    'SVM': [svm_grid.best_params_, -svm_grid.best_score_],
    'RandomForest': [rf_grid.best_params_, -rf_grid.best_score_],
    'XGBoost': [xgb_grid.best_params_, -xgb_grid.best_score_]
}, index=['best_params', 'best_rps_cv']).T

print(f"\n✓ Resumo dos resultados:")
print(results_df)

results_df.to_csv('models/gridsearch_results.csv')
print(f"✓ Resultados salvos em: models/gridsearch_results.csv")

print("\n" + "="*60)
print("GRIDSEARCH CONCLUÍDO!")
print("="*60)
print("\n💡 Próximos passos:")
print("   1. Execute 'python main.py' para retreinar com os novos parâmetros")
print("   2. Compare os resultados no Streamlit")
