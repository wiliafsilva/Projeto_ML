#!/usr/bin/env python
"""Script para GridSearch avançado com validação temporal"""

import sys
import io
from pathlib import Path

# Forçar UTF-8 no Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Adicionar o diretório raiz ao path
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
from sklearn.utils.class_weight import compute_sample_weight

from src.preprocessing import load_all_data
from src.feature_engineering import calculate_team_stats
from src.train_models import prepare_features_by_model

def rps(y_true, y_prob):
    """Ranked Probability Score - métrica principal do artigo"""
    y_true = y_true.astype(int)
    y_true_onehot = np.eye(3)[y_true]
    y_true_cum = np.cumsum(y_true_onehot, axis=1)
    y_prob_cum = np.cumsum(y_prob, axis=1)
    k_minus_1 = y_prob.shape[1] - 1 if y_prob.shape[1] > 1 else 1
    return np.mean(np.sum((y_true_cum - y_prob_cum)**2, axis=1)) / k_minus_1

class RPSScorer:
    """Scorer RPS customizado para GridSearch"""
    
    def __call__(self, estimator, X, y):
        """Calcula o RPS score"""
        # Retorna negativo porque queremos minimizar RPS, mas GridSearch maximiza
        return -rps(y, estimator.predict_proba(X))


print("="*60)
print("GRIDSEARCH AVANÇADO - OTIMIZAÇÃO DE HIPERPARÂMETROS")
print("="*60)

# Carregar dados (usar conjunto combinado para garantir continuidade temporal)
df = load_all_data()
features = calculate_team_stats(df)

# Separar treino (2005-2014) e teste (2014-2016)
train_mask = features['Season'] <= 2014
test_mask = features['Season'] > 2014

train = features[train_mask].reset_index(drop=True)
test = features[test_mask].reset_index(drop=True)

# Scorer customizado (RPS negativo para minimizar)
rps_scorer_fn = RPSScorer()

# Cross-validation temporal (evita data leakage)
tscv = TimeSeriesSplit(n_splits=5)

print(f"\nDataset de treino: {len(train)} partidas (2005-2014)")
print(f"Dataset de teste: {len(test)} partidas (2014-2016)")
print(f"Cross-validation: TimeSeriesSplit com 5 splits")
print(f"Métrica de otimização: RPS (Ranked Probability Score)\n")

# ============================================================
# Preparar features por tipo (Class A vs Class B)
# ============================================================
# Class B: usado por SVM, RandomForest, XGBoost
df_train_b = prepare_features_by_model(train, 'SVM')
X_train_b = df_train_b.drop(['Result', 'Season'], axis=1)
y_train_b = df_train_b['Result']

df_test_b = prepare_features_by_model(test, 'SVM')
X_test_b = df_test_b.drop(['Result', 'Season'], axis=1)
y_test_b = df_test_b['Result']

# Class A: usado por NaiveBayes
df_train_a = prepare_features_by_model(train, 'NaiveBayes')
X_train_a = df_train_a.drop(['Result', 'Season'], axis=1)
y_train_a = df_train_a['Result']

df_test_a = prepare_features_by_model(test, 'NaiveBayes')
X_test_a = df_test_a.drop(['Result', 'Season'], axis=1)
y_test_a = df_test_a['Result']

print(f"\nFeatures Class B (SVM/RF/XGB): {X_train_b.shape[1]} features")
print(f"Features Class A (NaiveBayes): {X_train_a.shape[1]} features")

# ============================================================
# 1. SVM - GridSearch (Class B features)
# ============================================================
print("\n" + "="*60)
print("[1] SVM - Otimizando hiperparâmetros (Class B)")
print("="*60)

svm_param_grid = {
    'C': [0.1, 0.5, 1.0, 2.0, 5.0],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf']
}

total_combos = len(svm_param_grid['C']) * len(svm_param_grid['gamma'])
print(f"Testando {total_combos} combinações...")

svm_grid = GridSearchCV(
    SVC(probability=True, random_state=42, class_weight='balanced'),
    svm_param_grid,
    cv=tscv,
    scoring=rps_scorer_fn,
    n_jobs=-1,
    verbose=1
)

svm_grid.fit(X_train_b, y_train_b)

print(f"\n✓ Melhores parâmetros SVM:")
for param, value in svm_grid.best_params_.items():
    print(f"  {param}: {value}")
print(f"✓ Melhor RPS (CV): {-svm_grid.best_score_:.4f}")

# ============================================================
# 2. RandomForest - GridSearch (Class B features)
# ============================================================
print("\n" + "="*60)
print("[2] RandomForest - Otimizando hiperparâmetros (Class B)")
print("="*60)

rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

total_combos = (len(rf_param_grid['n_estimators']) * len(rf_param_grid['max_depth']) *
                len(rf_param_grid['min_samples_split']) * len(rf_param_grid['min_samples_leaf']))
print(f"Testando {total_combos} combinações...")

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    rf_param_grid,
    cv=tscv,
    scoring=rps_scorer_fn,
    n_jobs=-1,
    verbose=1
)

rf_grid.fit(X_train_b, y_train_b)

print(f"\n✓ Melhores parâmetros RandomForest:")
for param, value in rf_grid.best_params_.items():
    print(f"  {param}: {value}")
print(f"✓ Melhor RPS (CV): {-rf_grid.best_score_:.4f}")

# ============================================================
# 3. XGBoost - GridSearch (Class B features)
# ============================================================
print("\n" + "="*60)
print("[3] XGBoost - Otimizando hiperparâmetros (Class B)")
print("="*60)

xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

total_combos = (len(xgb_param_grid['n_estimators']) * len(xgb_param_grid['max_depth']) *
                len(xgb_param_grid['learning_rate']) * len(xgb_param_grid['subsample']) *
                len(xgb_param_grid['colsample_bytree']))
print(f"Testando {total_combos} combinações...")

xgb_grid = GridSearchCV(
    XGBClassifier(eval_metric='mlogloss', random_state=42),
    xgb_param_grid,
    cv=tscv,
    scoring=rps_scorer_fn,
    n_jobs=-1,
    verbose=1
)

xgb_grid.fit(X_train_b, y_train_b)

print(f"\n✓ Melhores parâmetros XGBoost:")
for param, value in xgb_grid.best_params_.items():
    print(f"  {param}: {value}")
print(f"✓ Melhor RPS (CV): {-xgb_grid.best_score_:.4f}")

# ============================================================
# 4. NaiveBayes - GridSearch (Class A features)
# ============================================================
print("\n" + "="*60)
print("[4] NaiveBayes (GaussianNB) - Otimizando hiperparâmetros (Class A)")
print("="*60)

nb_param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
}

print(f"Testando {len(nb_param_grid['var_smoothing'])} combinações...")

nb_grid = GridSearchCV(
    GaussianNB(),
    nb_param_grid,
    cv=tscv,
    scoring=rps_scorer_fn,
    n_jobs=-1,
    verbose=1
)

nb_grid.fit(X_train_a, y_train_a)

print(f"\n✓ Melhores parâmetros NaiveBayes:")
for param, value in nb_grid.best_params_.items():
    print(f"  {param}: {value}")
print(f"✓ Melhor RPS (CV): {-nb_grid.best_score_:.4f}")

# ============================================================
# 5. AVALIAÇÃO NO TESTE (POR TEMPORADA)
# ============================================================
print("\n" + "="*60)
print("AVALIAÇÃO POR TEMPORADA (2014-2015, 2015-2016, ALL)")
print("="*60)
print("\nMetodologia do artigo: Avaliar separadamente em cada temporada de teste")
print("-"*60)

seasons_info = [
    ('2014-2015', 2015),
    ('2015-2016', 2016),
    ('All', None)
]

seasonal_results = []

best_models = {
    'SVM': {
        'model': svm_grid.best_estimator_,
        'params': svm_grid.best_params_,
        'cv_rps': -svm_grid.best_score_,
        'features': 'B'
    },
    'RandomForest': {
        'model': rf_grid.best_estimator_,
        'params': rf_grid.best_params_,
        'cv_rps': -rf_grid.best_score_,
        'features': 'B'
    },
    'XGBoost': {
        'model': xgb_grid.best_estimator_,
        'params': xgb_grid.best_params_,
        'cv_rps': -xgb_grid.best_score_,
        'features': 'B'
    },
    'NaiveBayes': {
        'model': nb_grid.best_estimator_,
        'params': nb_grid.best_params_,
        'cv_rps': -nb_grid.best_score_,
        'features': 'A'
    },
}

for season_name, season_value in seasons_info:
    print(f"\n{'='*50}")
    print(f"TEMPORADA: {season_name}")
    print(f"{'='*50}")
    
    # Filtrar dados da temporada
    if season_value is None:
        test_season = test
    else:
        test_season = test[test['Season'] == season_value]
    
    print(f"Total de jogos: {len(test_season)}\n")
    print(f"{'Modelo':<20} {'Accuracy':>10} {'F1 (macro)':>12} {'RPS':>10}")
    print("-" * 55)
    
    for model_name, model_info in best_models.items():
        model = model_info['model']
        
        # Usar features Class A ou Class B conforme o modelo
        if model_info['features'] == 'A':
            df_season_model = prepare_features_by_model(test_season, 'NaiveBayes')
        else:
            df_season_model = prepare_features_by_model(test_season, model_name)
        
        X_season = df_season_model.drop(['Result', 'Season'], axis=1)
        y_season = df_season_model['Result']
        
        y_pred = model.predict(X_season)
        y_pred_proba = model.predict_proba(X_season)
        
        acc = accuracy_score(y_season, y_pred)
        f1 = f1_score(y_season, y_pred, average='macro', zero_division=0)
        season_rps = rps(y_season.values, y_pred_proba)
        
        print(f"{model_name:<20} {acc:>10.4f} {f1:>12.4f} {season_rps:>10.4f}")
        
        seasonal_results.append({
            'Temporada': season_name,
            'Modelo': model_name,
            'Accuracy': acc,
            'F1': f1,
            'RPS': season_rps,
            'Jogos': len(y_season)
        })

# ============================================================
# 6. SALVAR RESULTADOS
# ============================================================
print("\n" + "="*60)
print("SALVANDO RESULTADOS")
print("="*60)

# Salvar modelos otimizados
joblib.dump(best_models, 'models/optimized_models.pkl')
print("✓ Modelos otimizados salvos em: models/optimized_models.pkl")

# Salvar resumo dos melhores parâmetros
results_summary = []
for name, info in best_models.items():
    row = {'Modelo': name, 'CV_RPS': info['cv_rps'], 'Features': info['features']}
    row.update(info['params'])
    results_summary.append(row)

results_df = pd.DataFrame(results_summary)
results_df.to_csv('models/gridsearch_results.csv', index=False)
print(f"✓ Parâmetros salvos em: models/gridsearch_results.csv")

# Salvar resultados por temporada
df_seasonal = pd.DataFrame(seasonal_results)
df_seasonal.to_csv('models/gridsearch_advanced_por_temporada.csv', index=False)
print(f"✓ Resultados por temporada salvos em: models/gridsearch_advanced_por_temporada.csv")

# Exibir tabela formatada de melhores parâmetros
print("\n" + "="*60)
print("RESUMO - MELHORES HIPERPARÂMETROS ENCONTRADOS")
print("="*60)
for name, info in best_models.items():
    print(f"\n  {name} (Class {info['features']}):")
    print(f"    CV RPS: {info['cv_rps']:.4f}")
    for param, value in info['params'].items():
        print(f"    {param}: {value}")

print("\n" + "="*60)
print("GRIDSEARCH CONCLUÍDO!")
print("="*60)
print("\n💡 Próximos passos:")
print("   1. Copie os melhores parâmetros para src/train_models.py")
print("   2. Execute 'python main.py' para retreinar com os novos parâmetros")
print("   3. Execute os scripts de análise para atualizar tabelas")
