"""
Update Tabela 4 - Confusion Matrices (DECODER HYBRID)
====================================================

Regenera matrizes de confusão (3x3) para todos os modelos Hybrid.

Autor: Projeto_ML
Data: Maio 2026
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix
from src.preprocessing import load_multiple_seasons
from src.feature_engineering import calculate_team_stats

print("="*80)
print("TABELA 4 - CONFUSION MATRICES (DECODER HYBRID)")
print("="*80)
print()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Atualizar tabela 4 hybrid")
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

# Labels das classes
class_labels = ['Vitória Casa (H)', 'Empate (D)', 'Vitória Visitante (A)']

# Carregar scaler e autoencoder
scaler_path = os.path.join(os.path.dirname(args.hybrid_path), 'scaler_hybrid.joblib')
encoder_path = os.path.join(os.path.dirname(args.hybrid_path), 'encoder_hybrid.keras')
decoder_path = os.path.join(os.path.dirname(args.hybrid_path), 'decoder_hybrid.keras')

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

scaler = joblib.load(scaler_path)
encoder = tf.keras.models.load_model(encoder_path)
decoder = tf.keras.models.load_model(decoder_path)

print("✓ Autoencoder components carregados")
print()

# Preparar dados de teste com features híbridas
X_test = features_test.drop(['Result', 'Season'], axis=1)
y_test = features_test['Result']

X_test_scaled = scaler.transform(X_test.values.astype(np.float32))

# Gerar features híbridas
X_test_latent = encoder(X_test_scaled).numpy()
X_test_reconstructed = decoder(X_test_latent).numpy()
X_test_reconstruction_error = np.mean(np.abs(X_test_scaled - X_test_reconstructed), axis=1, keepdims=True)

X_test_hybrid = np.hstack([
    X_test_latent,
    X_test_reconstructed,
    X_test_reconstruction_error
])

print(f"📊 Gerando Confusion Matrices...")
print("-" * 80)
print()

# Gerar CM para cada modelo
for model_name in ['RandomForest', 'XGBoost', 'NaiveBayes', 'SVM']:
    if model_name not in models_info:
        continue
    
    print(f"Gerando CM: {model_name}...")
    
    # Predições
    model = models_info[model_name]['model']
    y_pred = model.predict(X_test_hybrid)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    
    # Converter para DataFrame
    cm_df = pd.DataFrame(
        cm,
        index=[f'Verdadeiro: {label}' for label in class_labels],
        columns=[f'Predito: {label}' for label in class_labels]
    )
    
    # Salvar
    output_path = os.path.join(output_dir, f'tabela4_cm_hybrid_{model_name.lower()}.csv')
    cm_df.to_csv(output_path)
    
    print(f"\n{model_name}:")
    print(cm_df.to_string())
    print(f"\n✓ Salvo em: {output_path}\n")

print("="*80)
print("✅ TABELA 4 CONCLUÍDA!")
print("="*80)
