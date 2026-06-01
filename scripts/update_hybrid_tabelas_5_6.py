"""
Update Tabelas 5 e 6 - Performance Temporal e Classificação (DECODER HYBRID)
===========================================================================

Tabela 5: Performance por temporada (accuracy, F1, RPS por season)
Tabela 6: Classificação por classe (precision/recall/f1 para H/D/A)

Autor: Projeto_ML
Data: Maio 2026
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.preprocessing import load_multiple_seasons
from src.feature_engineering import calculate_team_stats

print("="*80)
print("TABELAS 5 E 6 (DECODER HYBRID)")
print("="*80)
print()


def rps_score(y_true, y_proba):
    """Ranked Probability Score (menor = melhor)"""
    n_classes = y_proba.shape[1]
    rps = 0
    
    for i, true_class in enumerate(y_true):
        y_true_cdf = np.zeros(n_classes)
        y_true_cdf[int(true_class):] = 1
        y_pred_cdf = np.cumsum(y_proba[i])
        rps += np.sum((y_pred_cdf - y_true_cdf) ** 2)
    
    return rps / len(y_true)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Atualizar tabelas 5 e 6 hybrid")
    parser.add_argument("--hybrid-path", default="models/autoencoder_decoder_hybrid/trained_models_hybrid.pkl")
    parser.add_argument("--output-dir", default="models/autoencoder_decoder_hybrid")
    return parser.parse_args()


args = parse_args()
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# Carregar dados
print("📂 Carregando dados...")
train_dir = "data/data_2005_2014"
test_dir = "data/data_2014_2016"

df_train = load_multiple_seasons(train_dir)
df_test = load_multiple_seasons(test_dir)

# Calcular features
print("🔧 Calculando features...")
features_train = calculate_team_stats(df_train)
features_test = calculate_team_stats(df_test)

print(f"   ✓ Treino: {len(features_train)} amostras")
print(f"   ✓ Teste: {len(features_test)} amostras")
print()

# Carregar modelos
print("📂 Carregando modelos Hybrid...")
try:
    hybrid_meta = joblib.load(args.hybrid_path)
    models_info = hybrid_meta['models']
    print(f"   ✓ {len(models_info)} modelos")
except Exception as e:
    print(f"   ❌ Erro: {e}")
    exit(1)

print()

# Preparar dados de teste com features híbridas
X_test = features_test.drop(['Result', 'Season'], axis=1)
y_test = features_test['Result']

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

scaler_path = os.path.join(os.path.dirname(args.hybrid_path), 'scaler_hybrid.joblib')
encoder_path = os.path.join(os.path.dirname(args.hybrid_path), 'encoder_hybrid.keras')
decoder_path = os.path.join(os.path.dirname(args.hybrid_path), 'decoder_hybrid.keras')

scaler = joblib.load(scaler_path)
encoder = tf.keras.models.load_model(encoder_path)
decoder = tf.keras.models.load_model(decoder_path)

X_test_scaled = scaler.transform(X_test.values.astype(np.float32))
X_test_latent = encoder(X_test_scaled).numpy()
X_test_reconstructed = decoder(X_test_latent).numpy()
X_test_reconstruction_error = np.mean(np.abs(X_test_scaled - X_test_reconstructed), axis=1, keepdims=True)

# UPDATED: usar apenas as features reconstruídas do decoder
X_test_hybrid = X_test_reconstructed

# ============================================================================
# TABELA 5: PERFORMANCE POR TEMPORADA
# ============================================================================
print("="*80)
print("TABELA 5: PERFORMANCE POR TEMPORADA")
print("="*80)
print()

# Seasons de teste
seasons = sorted(features_test['Season'].unique())
season_names = {
    2015: '2014-2015',
    2016: '2015-2016',
}

table5_data = []

for model_name in ['RandomForest', 'XGBoost', 'NaiveBayes', 'SVM']:
    if model_name not in models_info:
        continue
    
    model = models_info[model_name]['model']
    
    for season in seasons:
        season_mask = features_test['Season'] == season
        y_season = y_test[season_mask]
        X_season_hybrid = X_test_hybrid[season_mask.values]
        
        y_pred = model.predict(X_season_hybrid)
        y_proba = model.predict_proba(X_season_hybrid)
        
        acc = accuracy_score(y_season, y_pred)
        f1 = f1_score(y_season, y_pred, average='macro', zero_division=0)
        rps = rps_score(y_season.values, y_proba)
        
        table5_data.append({
            'Temporada': season_names.get(season, str(season)),
            'Modelo': model_name,
            'Accuracy': f"{acc:.4f}",
            'F1-Score': f"{f1:.4f}",
            'RPS': f"{rps:.4f}",
            'N_Amostras': season_mask.sum()
        })

# Também incluir "All" (todas as temporadas de teste)
for model_name in ['RandomForest', 'XGBoost', 'NaiveBayes', 'SVM']:
    if model_name not in models_info:
        continue
    
    model = models_info[model_name]['model']
    
    y_pred = model.predict(X_test_hybrid)
    y_proba = model.predict_proba(X_test_hybrid)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    rps = rps_score(y_test.values, y_proba)
    
    table5_data.append({
        'Temporada': 'All (2014-2016)',
        'Modelo': model_name,
        'Accuracy': f"{acc:.4f}",
        'F1-Score': f"{f1:.4f}",
        'RPS': f"{rps:.4f}",
        'N_Amostras': len(y_test)
    })

table5_df = pd.DataFrame(table5_data)
output_path5 = os.path.join(output_dir, 'tabela5_hybrid_performance_temporada.csv')
table5_df.to_csv(output_path5, index=False)

print(table5_df.to_string(index=False))
print(f"\n✓ Salvo em: {output_path5}\n")

# ============================================================================
# TABELA 6: CLASSIFICAÇÃO POR CLASSE
# ============================================================================
print("="*80)
print("TABELA 6: CLASSIFICAÇÃO POR CLASSE (Precision/Recall/F1)")
print("="*80)
print()

class_names = {0: 'H (Home Win)', 1: 'D (Draw)', 2: 'A (Away Win)'}
table6_data = []

for model_name in ['RandomForest', 'XGBoost', 'NaiveBayes', 'SVM']:
    if model_name not in models_info:
        continue
    
    model = models_info[model_name]['model']
    y_pred = model.predict(X_test_hybrid)
    
    for class_idx in [0, 1, 2]:
        prec = precision_score(y_test, y_pred, labels=[class_idx], average='micro', zero_division=0)
        rec = recall_score(y_test, y_pred, labels=[class_idx], average='micro', zero_division=0)
        f1 = f1_score(y_test, y_pred, labels=[class_idx], average='micro', zero_division=0)
        
        # Contar support (quantas amostras dessa classe)
        support = (y_test == class_idx).sum()
        
        table6_data.append({
            'Modelo': model_name,
            'Classe': class_names[class_idx],
            'Precision': f"{prec:.4f}",
            'Recall': f"{rec:.4f}",
            'F1-Score': f"{f1:.4f}",
            'Support': support
        })

table6_df = pd.DataFrame(table6_data)
output_path6 = os.path.join(output_dir, 'tabela6_hybrid_classificacao_classe.csv')
table6_df.to_csv(output_path6, index=False)

print(table6_df.to_string(index=False))
print(f"\n✓ Salvo em: {output_path6}\n")

print("="*80)
print("✅ TABELAS 5 E 6 CONCLUÍDAS!")
print("="*80)
