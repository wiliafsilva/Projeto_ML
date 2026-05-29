"""
Confidence Intervals Bootstrap - VERSAO RAPIDA
==============================================

Calcula intervalos de confiança (95%) via bootstrap com 100 iterações.
Versão rápida para validação estatística.

Autor: Projeto_ML
Data: Março 2026
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score
from src.preprocessing import load_all_data
from src.feature_engineering import calculate_team_stats
from src.train_models import prepare_features_by_model

# Configuração bootstrap
N_ITERATIONS = 100  # Rápido para validação
CONFIDENCE_LEVEL = 95
RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Confidence intervals (bootstrap)")
    parser.add_argument("--model-path", default="models/trained_models.pkl")
    parser.add_argument("--output-dir", default="models")
    return parser.parse_args()

def rps_score(y_true, y_proba):
    """Ranked Probability Score (menor = melhor)"""
    n_classes = y_proba.shape[1]
    rps = 0
    
    for i, true_class in enumerate(y_true):
        # CDF observada
        y_true_cdf = np.zeros(n_classes)
        y_true_cdf[int(true_class):] = 1
        
        # CDF predita
        y_pred_cdf = np.cumsum(y_proba[i])
        
        # RPS = soma dos quadrados das diferenças
        rps += np.sum((y_pred_cdf - y_true_cdf) ** 2)
    
    return rps / len(y_true)

print("="*80)
print("INTERVALOS DE CONFIANCA (BOOTSTRAP - VERSAO RAPIDA)")
print("="*80)
print(f"Iterações: {N_ITERATIONS}")
print(f"Confiança: {CONFIDENCE_LEVEL}%")
print()

# Carregar dados
print("Carregando dados...")
df_all = load_all_data()
df_features = calculate_team_stats(df_all)

# Split treino/teste
df_train = df_all[df_all['Season'] <= 2014].copy().reset_index(drop=True)
df_test = df_all[df_all['Season'] > 2014].copy().reset_index(drop=True)

df_features_test = df_features[df_all['Season'] > 2014].reset_index(drop=True)

print(f"   Teste: {len(df_test)} amostras")
print()

# Carregar modelos
print("Carregando modelos treinados...")
args = parse_args()
results_metadata = joblib.load(args.model_path)
models_info = results_metadata['models']
print(f"   {len(models_info)} modelos")
print()

# Temporadas de teste
seasons_test = [
    ('2014-2015', df_test['Season'] == 2015),
    ('2015-2016', df_test['Season'] == 2016),
    ('All', df_test['Season'] > 2014)
]

all_ci_results = []

# Bootstrap para cada modelo e cada temporada
for season_name, season_mask in seasons_test:
    print(f"\n{'='*80}")
    print(f"TEMPORADA: {season_name}")
    print(f"{'='*80}\n")
    
    # Filtrar dados da temporada
    y_season = df_test[season_mask]['Result']
    df_features_season = df_features_test[season_mask].reset_index(drop=True)
    
    print(f"Total de jogos: {len(y_season)}")
    print()
    
    for model_name in ['RandomForest', 'XGBoost', 'NaiveBayes', 'SVM']:
        if model_name not in models_info:
            continue
        
        print(f"Bootstrap {model_name}...")
        
        # Preparar features específicas
        df_season_model = prepare_features_by_model(df_features_season, model_name)
        X_season = df_season_model.drop(['Result', 'Season'], axis=1)
        
        # Modelo
        model = models_info[model_name]['model']
        
        # Armazenar métricas de cada iteração
        acc_bootstrap = []
        f1_bootstrap = []
        rps_bootstrap = []
        
        np.random.seed(RANDOM_STATE)
        
        for i in range(N_ITERATIONS):
            # Resample com reposição
            X_boot, y_boot = resample(X_season, y_season, random_state=i)
            
            # Predições
            y_pred = model.predict(X_boot)
            y_proba = model.predict_proba(X_boot)
            
            # Métricas
            acc = accuracy_score(y_boot, y_pred)
            f1 = f1_score(y_boot, y_pred, average='macro')
            rps = rps_score(y_boot, y_proba)
            
            acc_bootstrap.append(acc)
            f1_bootstrap.append(f1)
            rps_bootstrap.append(rps)
            
            if (i+1) % 25 == 0:
                print(f"   Iteracao {i+1}/{N_ITERATIONS}")
        
        # Calcular percentis para CI
        alpha = (100 - CONFIDENCE_LEVEL) / 2
        
        result = {
            'Temporada': season_name,
            'Modelo': model_name,
            'Accuracy_Mean': np.mean(acc_bootstrap),
            'Accuracy_CI_Lower': np.percentile(acc_bootstrap, alpha),
            'Accuracy_CI_Upper': np.percentile(acc_bootstrap, 100 - alpha),
            'F1_Mean': np.mean(f1_bootstrap),
            'F1_CI_Lower': np.percentile(f1_bootstrap, alpha),
            'F1_CI_Upper': np.percentile(f1_bootstrap, 100 - alpha),
            'RPS_Mean': np.mean(rps_bootstrap),
            'RPS_CI_Lower': np.percentile(rps_bootstrap, alpha),
            'RPS_CI_Upper': np.percentile(rps_bootstrap, 100 - alpha)
        }
        
        all_ci_results.append(result)
        
        print(f"   ✓ Concluido!")
        print()

# Criar DataFrame
ci_df = pd.DataFrame(all_ci_results)

# Salvar
output_path = os.path.join(args.output_dir, 'confidence_intervals.csv')
ci_df.to_csv(output_path, index=False)

print("\n" + "="*80)
print("INTERVALOS DE CONFIANCA (95%) - RESUMO")
print("="*80)
print()

# Mostrar resultados por temporada
for season_name, _ in seasons_test:
    print(f"\n{season_name}:")
    print("-" * 80)
    season_data = ci_df[ci_df['Temporada'] == season_name]
    for _, row in season_data.iterrows():
        modelo = row['Modelo']
        acc_mean = row['Accuracy_Mean']
        acc_lower = row['Accuracy_CI_Lower']
        acc_upper = row['Accuracy_CI_Upper']
        
        print(f"   {modelo:15s}: {acc_mean:.4f} [{acc_lower:.4f}, {acc_upper:.4f}]")

print()
print(f"💾 Salvo em: {output_path}")
print()
print("✅ BOOTSTRAP CONCLUIDO!")

