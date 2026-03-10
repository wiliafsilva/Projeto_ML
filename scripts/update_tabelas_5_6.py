"""
Update Tabelas 5 e 6 - Performance Temporal e Classificação por Classe
======================================================================

Regenera Tabelas 5 e 6 usando modelos atuais (trained_models.pkl).

Tabela 5: Performance por temporada (accuracy por season)
Tabela 6: Classificação por classe (precision/recall/f1 para H/D/A)

Autor: Projeto_ML
Data: Março 2026
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.preprocessing import load_all_data
from src.feature_engineering import calculate_team_stats
from src.train_models import prepare_features_by_model

print("="*80)
print("ATUALIZANDO TABELAS 5 E 6")
print("="*80)
print()

# Carregar dados
print("📂 Carregando dados...")
df_all = load_all_data()
df_features = calculate_team_stats(df_all)

# Split treino/teste
df_train = df_all[df_all['Season'] <= 2014].copy().reset_index(drop=True)
df_test = df_all[df_all['Season'] > 2014].copy().reset_index(drop=True)

df_features_train = df_features[df_all['Season'] <= 2014].reset_index(drop=True)
df_features_test = df_features[df_all['Season'] > 2014].reset_index(drop=True)

y_test = df_test['Result']
print(f"   ✓ Teste: {len(y_test)} amostras")
print()

# Carregar modelos
print("📂 Carregando modelos treinados...")
results_metadata = joblib.load('models/trained_models.pkl')
models_info = results_metadata['models']
print(f"   ✓ {len(models_info)} modelos")
print()

# ============================================================================
# TABELA 5: PERFORMANCE POR TEMPORADA
# ============================================================================
print("="*80)
print("TABELA 5: PERFORMANCE POR TEMPORADA")
print("="*80)
print()

# Calcular baseline por temporada
from collections import Counter
y_train = df_train['Result']
baseline_pred = Counter(y_train).most_common(1)[0][0]

# Agrupar por temporada
seasons = sorted(df_test['Season'].unique())
table5_data = []

for season in seasons:
    season_mask = df_test['Season'] == season
    y_season = df_test[season_mask]['Result']
    n_games = len(y_season)
    
    # Baseline
    baseline_preds = np.full(n_games, baseline_pred)
    baseline_acc = accuracy_score(y_season, baseline_preds)
    
    row = {
        'Temporada': f"{int(season)-1}-{int(season)}",
        'Jogos': n_games,
        'Baseline': f"{baseline_acc*100:.2f}%"
    }
    
    # Modelos ML
    for model_name in ['SVM', 'RandomForest', 'XGBoost', 'NaiveBayes']:
        if model_name in models_info:
            # Preparar features específicas
            df_season_model = prepare_features_by_model(
                df_features_test[df_test['Season'] == season].reset_index(drop=True),
                model_name
            )
            X_season = df_season_model.drop(['Result', 'Season'], axis=1)
            
            # Predições
            model = models_info[model_name]['model']
            y_pred = model.predict(X_season)
            
            # Accuracy
            acc = accuracy_score(y_season, y_pred)
            row[model_name] = f"{acc*100:.2f}%"
    
    table5_data.append(row)
    print(f"   ✓ {row['Temporada']}: {row['Jogos']} jogos")

# Salvar Tabela 5
table5_df = pd.DataFrame(table5_data)
output_path5 = 'models/tabela5_performance_temporada.csv'
table5_df.to_csv(output_path5, index=False)

print()
print("TABELA 5:")
print(table5_df.to_string(index=False))
print()
print(f"💾 Salva em: {output_path5}")
print()

# ============================================================================
# TABELA 6: CLASSIFICAÇÃO POR CLASSE (4 tabelas, uma por modelo)
# ============================================================================
print("="*80)
print("TABELA 6: CLASSIFICAÇÃO POR CLASSE")
print("="*80)
print()

class_names = {0: 'Home Win (H)', 1: 'Draw (D)', 2: 'Away Win (A)'}

for model_name in ['RandomForest', 'XGBoost', 'NaiveBayes', 'SVM']:
    if model_name not in models_info:
        continue
    
    print(f"📊 {model_name}...")
    
    # Preparar features específicas
    df_test_model = prepare_features_by_model(df_features_test, model_name)
    X_test = df_test_model.drop(['Result', 'Season'], axis=1)
    
    # Predições
    model = models_info[model_name]['model']
    y_pred = model.predict(X_test)
    
    # Métricas por classe
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    # Confusion matrix para contar suporte
    cm = confusion_matrix(y_test, y_pred)
    support_per_class = cm.sum(axis=1)  # Total real de cada classe
    
    # Construir tabela
    table6_data = []
    for class_id in [0, 1, 2]:
        table6_data.append({
            'Classe': class_names[class_id],
            'Precision': f"{precision_per_class[class_id]:.4f}",
            'Recall': f"{recall_per_class[class_id]:.4f}",
            'F1-Score': f"{f1_per_class[class_id]:.4f}",
            'Support': int(support_per_class[class_id])
        })
    
    # Adicionar média macro
    table6_data.append({
        'Classe': 'Macro Avg',
        'Precision': f"{precision_per_class.mean():.4f}",
        'Recall': f"{recall_per_class.mean():.4f}",
        'F1-Score': f"{f1_per_class.mean():.4f}",
        'Support': int(support_per_class.sum())
    })
    
    # Salvar
    table6_df = pd.DataFrame(table6_data)
    output_path6 = f'models/tabela6_classificacao_{model_name.lower()}.csv'
    table6_df.to_csv(output_path6, index=False)
    
    print(f"   ✓ Salvo em: {output_path6}")

print()
print("="*80)
print("✅ TABELAS 5 E 6 ATUALIZADAS!")
print("="*80)
print()
print("📁 Arquivos gerados:")
print("   - models/tabela5_performance_temporada.csv")
print("   - models/tabela6_classificacao_randomforest.csv")
print("   - models/tabela6_classificacao_xgboost.csv")
print("   - models/tabela6_classificacao_naivebayes.csv")
print("   - models/tabela6_classificacao_svm.csv")
