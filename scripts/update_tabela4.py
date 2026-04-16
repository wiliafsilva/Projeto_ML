"""
Update Tabela 4 - Confusion Matrices
====================================

Regenera matrizes de confusão (3x3) para todos os modelos.

Autor: Projeto_ML
Data: Março 2026
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix
from src.preprocessing import load_all_data
from src.feature_engineering import calculate_team_stats
from src.train_models import prepare_features_by_model

print("="*80)
print("ATUALIZANDO TABELA 4 - CONFUSION MATRICES")
print("="*80)
print()

# Carregar dados
print("Carregando dados...")
df_all = load_all_data()
df_features = calculate_team_stats(df_all)

# Split treino/teste
df_train = df_all[df_all['Season'] <= 2014].copy().reset_index(drop=True)
df_test = df_all[df_all['Season'] > 2014].copy().reset_index(drop=True)

df_features_test = df_features[df_all['Season'] > 2014].reset_index(drop=True)
y_test = df_test['Result']

print(f"   Teste: {len(y_test)} amostras")
print()

# Carregar modelos
print("Carregando modelos treinados...")
results_metadata = joblib.load('models/trained_models.pkl')
models_info = results_metadata['models']
print(f"   {len(models_info)} modelos")
print()

# Labels das classes
class_labels = ['Vitoria Casa', 'Empate', 'Vitoria Visitante']

# Gerar CM para cada modelo
for model_name in ['RandomForest', 'XGBoost', 'NaiveBayes', 'SVM']:
    if model_name not in models_info:
        continue
    
    print(f"Gerando CM: {model_name}...")
    
    # Preparar features específicas
    df_test_model = prepare_features_by_model(df_features_test, model_name)
    X_test = df_test_model.drop(['Result', 'Season'], axis=1)
    
    # Predições
    model = models_info[model_name]['model']
    y_pred = model.predict(X_test)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    
    # Criar DataFrame formatado
    cm_df = pd.DataFrame(cm, 
                         index=[f'Real: {label}' for label in class_labels],
                         columns=[f'Pred: {label}' for label in class_labels])
    
    # Adicionar coluna Total
    cm_df['Total'] = cm_df.sum(axis=1)
    
    # Salvar
    output_path = f'models/tabela4_cm_{model_name.lower()}.csv'
    cm_df.to_csv(output_path)
    
    # Calcular accuracy da diagonal
    accuracy = np.trace(cm) / len(y_test)
    
    print(f"   Accuracy: {accuracy*100:.2f}%")
    print(f"   Salvo em: {output_path}")
    print()

print("="*80)
print("CONFUSION MATRICES GERADAS!")
print("="*80)
print()

# Mostrar exemplo (RandomForest)
print("Exemplo - RandomForest:")
cm_rf = pd.read_csv('models/tabela4_cm_randomforest.csv', index_col=0)
print(cm_rf)
print()

print("Arquivos gerados:")
print("   - models/tabela4_cm_randomforest.csv")
print("   - models/tabela4_cm_xgboost.csv")
print("   - models/tabela4_cm_naivebayes.csv")
print("   - models/tabela4_cm_svm.csv")
