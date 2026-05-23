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
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

from src.preprocessing import load_data
from src.feature_engineering import calculate_team_stats

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
    
    def _score_func(self, *args, **kwargs):
        """Para compatibilidade com make_scorer"""
        return self.__call__(*args, **kwargs)


print("="*60)
print("GRIDSEARCH AVANÇADO - OTIMIZAÇÃO DE HIPERPARÂMETROS")
print("="*60)

# Carregar dados (usar conjunto combinado de temporadas presente em data/)
from src.preprocessing import load_all_data
df = load_all_data()
features = calculate_team_stats(df)

# Separar treino (2011-2023) e teste (2023-2025)
train = features[features['Season'] <= 2023]
test = features[features['Season'] > 2023]

X_train = train.drop(['Result','Season'], axis=1)
y_train = train['Result']
X_test = test.drop(['Result','Season'], axis=1)
y_test = test['Result']

# Sample weights para modelos que suportam
sample_weights = compute_sample_weight('balanced', y_train)

# Scorer customizado (RPS negativo para minimizar)
rps_scorer_fn = RPSScorer()

# Cross-validation temporal (evita data leakage)
tscv = TimeSeriesSplit(n_splits=5)

print(f"\nDataset de treino: {len(X_train)} partidas (2011-2023)")
print(f"Dataset de teste: {len(X_test)} partidas (2023-2025)")
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
# 4. NaiveBayes - GridSearch
# ============================================================
print("\n" + "="*60)
print("[4] NaiveBayes (GaussianNB) - Otimizando hiperparâmetros")
print("="*60)

nb_param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
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

nb_grid.fit(X_train, y_train)

print(f"\n✓ Melhores parâmetros NaiveBayes:")
for param, value in nb_grid.best_params_.items():
    print(f"  {param}: {value}")
print(f"✓ Melhor RPS (CV): {-nb_grid.best_score_:.4f}")

# ============================================================
# 5. AVALIAÇÃO POR TEMPORADA (ARTIGO CIENTÍFICO)
# ============================================================
print("\n" + "="*60)
print("AVALIAÇÃO POR TEMPORADA (2023-2024, 2024-2025, ALL)")
print("="*60)
print("\nMetodologia do artigo: Avaliar separadamente em cada temporada de teste")
print("-"*60)

# Temporadas de teste
seasons_info = [
    ('2023-2024', 2024),
    ('2024-2025', 2025),
    ('All', None)
]

# Resultados por temporada
seasonal_results = []

# Preparar dicionário de modelos otimizados (necessário para avaliação por temporada)
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
    },
    'NaiveBayes_optimized': {
        'model': nb_grid.best_estimator_,
        'params': nb_grid.best_params_,
        'cv_rps': -nb_grid.best_score_
    },
}

for season_name, season_value in seasons_info:
    print(f"\n{'='*50}")
    print(f"TEMPORADA: {season_name}")
    print(f"{'='*50}")
    
    # Filtrar dados da temporada
    if season_value is None:
        # Todas as temporadas
        test_season = test
    else:
        # Temporada específica
        test_season = test[test['Season'] == season_value]
    
    X_season = test_season.drop(['Result', 'Season'], axis=1)
    y_season = test_season['Result']
    
    print(f"Total de jogos: {len(y_season)}\n")
    print(f"{'Modelo':<25} {'RPS':>10}")
    print("-" * 50)
    
    # Avaliar cada modelo
    for model_name, model_info in best_models.items():
        model = model_info['model']
        y_pred_proba = model.predict_proba(X_season)
        
        # Calcular RPS
        season_rps = rps(y_season.values, y_pred_proba)
        
        # Nome do modelo para display (remover _optimized)
        display_name = model_name.replace('_optimized', '')
        
        print(f"{display_name:<25} {season_rps:>10.4f}")
        
        # Armazenar resultado
        seasonal_results.append({
            'Temporada': season_name,
            'Modelo': display_name,
            'RPS': season_rps,
            'Jogos': len(y_season)
        })

"""
Após avaliação por temporada, salvamos os resultados detalhados e os modelos otimizados.
"""

# Adicionar resultados por temporada aos modelos
for model_info in best_models.values():
    model_info['seasonal_results'] = []

joblib.dump(best_models, 'models/optimized_models.pkl')
print("✓ Modelos otimizados salvos em: models/optimized_models.pkl")

# Salvar resultados detalhados
results_df = pd.DataFrame({
    'SVM': [svm_grid.best_params_, -svm_grid.best_score_],
    'RandomForest': [rf_grid.best_params_, -rf_grid.best_score_],
    'XGBoost': [xgb_grid.best_params_, -xgb_grid.best_score_],
    'NaiveBayes': [nb_grid.best_params_, -nb_grid.best_score_],
}, index=['best_params', 'best_rps_cv']).T

print(f"\n✓ Resumo dos resultados (CV):")
print(results_df)

results_df.to_csv('models/gridsearch_results.csv')
print(f"✓ Resultados de CV salvos em: models/gridsearch_results.csv")

# Salvar resultados por temporada
df_seasonal = pd.DataFrame(seasonal_results)

# Reorganizar para formato do artigo (temporadas como linhas, modelos como colunas)
df_pivot = df_seasonal.pivot(index='Temporada', columns='Modelo', values='RPS')

# Ordenar colunas
model_order = ['SVM', 'RandomForest', 'XGBoost', 'NaiveBayes']
df_pivot = df_pivot[[col for col in model_order if col in df_pivot.columns]]

# Ordenar linhas (2023-2024, 2024-2025, All)
season_order = ['2023-2024', '2024-2025', 'All']
df_pivot = df_pivot.reindex(season_order)

print("\n📊 RESULTADOS POR TEMPORADA:")
print(df_pivot.to_string())

# Salvar CSV formato artigo
df_pivot.to_csv('models/gridsearch_advanced_por_temporada.csv')
print("\n✓ Resultados por temporada salvos em: models/gridsearch_advanced_por_temporada.csv")

print("\n" + "="*60)
print("GRIDSEARCH CONCLUÍDO!")
print("="*60)
print("\n💡 Próximos passos:")
print("   1. Execute 'python main.py' para retreinar com os novos parâmetros")
print("   2. Compare os resultados no Streamlit")
