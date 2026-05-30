"""
Baseline Comparison - DECODER HYBRID
====================================

Compara modelos Hybrid com modelo trivial (baseline).

Autor: Projeto_ML
Data: Maio 2026
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib

from src.preprocessing import load_multiple_seasons
from src.feature_engineering import calculate_team_stats

print("="*80)
print("BASELINE COMPARISON - DECODER HYBRID")
print("="*80)
print()

# Carregar dados
print("📂 Carregando dados...")
train_dir = "data/data_2005_2014"
test_dir = "data/data_2014_2016"

df_train = load_multiple_seasons(train_dir)
df_test = load_multiple_seasons(test_dir)

features_train = calculate_team_stats(df_train)
features_test = calculate_team_stats(df_test)

y_train = features_train['Result']
y_test = features_test['Result']

print(f"   ✓ Treino: {len(y_train)} amostras")
print(f"   ✓ Teste: {len(y_test)} amostras")
print()

# Distribuição de classes
print("Distribuição de Classes (Conjunto de Teste):")
print("-" * 60)
class_counts = y_test.value_counts().sort_index()
class_names = {0: 'Home Win (H)', 1: 'Draw (D)', 2: 'Away Win (A)'}

total = len(y_test)
for cls, count in class_counts.items():
    pct = count / total * 100
    print(f"   {class_names[cls]}: {count:3d} jogos ({pct:5.1f}%)")

print()
print(f"   Classe majoritária: {class_names[class_counts.idxmax()]} "
      f"({class_counts.max() / total * 100:.1f}%)")
print()

# ============================================================================
# BASELINE MODELS
# ============================================================================
print("="*80)
print("BASELINES")
print("="*80)
print()

X_train = features_train.drop(['Result', 'Season'], axis=1)
X_test = features_test.drop(['Result', 'Season'], axis=1)

baseline_results = []

# 1. Most Frequent
print("BASELINE 1: Sempre Prever Classe Majoritária")
print("-" * 60)
baseline_most = DummyClassifier(strategy='most_frequent', random_state=42)
baseline_most.fit(X_train, y_train)
y_pred = baseline_most.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

print(f"   Predição: Sempre '{class_names[y_pred[0]]}'")
print(f"   Accuracy:  {acc:.4f}")
print(f"   Precision: {prec:.4f}")
print(f"   Recall:    {rec:.4f}")
print(f"   F1-Score:  {f1:.4f}")
print()

baseline_results.append({
    'Tipo': 'Baseline',
    'Modelo': 'Most Frequent',
    'Accuracy': acc,
    'Precision': prec,
    'Recall': rec,
    'F1': f1,
})

# 2. Stratified
print("BASELINE 2: Predição Estratificada")
print("-" * 60)
baseline_strat = DummyClassifier(strategy='stratified', random_state=42)
baseline_strat.fit(X_train, y_train)
y_pred = baseline_strat.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

print(f"   Accuracy:  {acc:.4f}")
print(f"   Precision: {prec:.4f}")
print(f"   Recall:    {rec:.4f}")
print(f"   F1-Score:  {f1:.4f}")
print()

baseline_results.append({
    'Tipo': 'Baseline',
    'Modelo': 'Stratified',
    'Accuracy': acc,
    'Precision': prec,
    'Recall': rec,
    'F1': f1,
})

# ============================================================================
# MODELOS HYBRID
# ============================================================================
print("="*80)
print("DECODER HYBRID MODELS")
print("="*80)
print()

hybrid_path = "models/autoencoder_decoder_hybrid/trained_models_hybrid.pkl"

try:
    hybrid_meta = joblib.load(hybrid_path)
    models_info = hybrid_meta['models']
    print(f"📂 Carregando {len(models_info)} modelos Hybrid...")
    print()
except Exception as e:
    print(f"❌ Erro ao carregar modelos: {e}")
    exit(1)

# Preparar features híbridas
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

scaler_path = "models/autoencoder_decoder_hybrid/scaler_hybrid.joblib"
encoder_path = "models/autoencoder_decoder_hybrid/encoder_hybrid.keras"
decoder_path = "models/autoencoder_decoder_hybrid/decoder_hybrid.keras"

scaler = joblib.load(scaler_path)
encoder = tf.keras.models.load_model(encoder_path)
decoder = tf.keras.models.load_model(decoder_path)

X_test_scaled = scaler.transform(X_test.values.astype(np.float32))
X_test_latent = encoder(X_test_scaled).numpy()
X_test_reconstructed = decoder(X_test_latent).numpy()
X_test_reconstruction_error = np.mean(np.abs(X_test_scaled - X_test_reconstructed), axis=1, keepdims=True)

X_test_hybrid = np.hstack([
    X_test_latent,
    X_test_reconstructed,
    X_test_reconstruction_error
])

for model_name in ['RandomForest', 'XGBoost', 'NaiveBayes', 'SVM']:
    if model_name not in models_info:
        continue
    
    model = models_info[model_name]['model']
    y_pred = model.predict(X_test_hybrid)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"{model_name}:")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print()
    
    baseline_results.append({
        'Tipo': 'ML (Hybrid)',
        'Modelo': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
    })

# ============================================================================
# SALVAR RESULTADOS
# ============================================================================
baseline_df = pd.DataFrame(baseline_results)
output_path = "models/autoencoder_decoder_hybrid/hybrid_baseline_comparison.csv"
baseline_df.to_csv(output_path, index=False)

print("="*80)
print("COMPARAÇÃO RESUMIDA")
print("="*80)
print(baseline_df.to_string(index=False))
print(f"\n✓ Salvo em: {output_path}")
print()

# Melhor modelo
ml_models = baseline_df[baseline_df['Tipo'] == 'ML (Hybrid)']
best_model = ml_models.loc[ml_models['F1'].idxmax()]
print(f"🏆 Melhor modelo: {best_model['Modelo']} (F1: {best_model['F1']:.4f})")
print()

print("="*80)
print("✅ BASELINE COMPARISON CONCLUÍDO!")
print("="*80)
