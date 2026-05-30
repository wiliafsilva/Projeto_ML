"""
Radar Chart - DECODER HYBRID
============================

Gera radar charts com as 5 métricas principais por modelo e temporada.

Autor: Projeto_ML
Data: Maio 2026
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from math import pi
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.preprocessing import load_multiple_seasons
from src.feature_engineering import calculate_team_stats

print("="*80)
print("RADAR CHART - DECODER HYBRID")
print("="*80)
print()


def rps_score(y_true, y_proba):
    """RPS Score"""
    n_classes = y_proba.shape[1]
    rps = 0
    for i, true_class in enumerate(y_true):
        y_true_cdf = np.zeros(n_classes)
        y_true_cdf[int(true_class):] = 1
        y_pred_cdf = np.cumsum(y_proba[i])
        rps += np.sum((y_pred_cdf - y_true_cdf) ** 2)
    return rps / len(y_true)


def generate_radar_chart(season_name, season_mask, models_info, X_hybrid, y_test, output_dir):
    """Gera radar chart para uma temporada específica"""
    
    print(f"\n📊 Gerando radar chart para: {season_name}")
    
    # Filtrar dados da temporada
    y_season = y_test[season_mask.values]
    X_season = X_hybrid[season_mask.values]
    
    # Modelos
    model_names = ['RandomForest', 'XGBoost', 'NaiveBayes', 'SVM']
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', '1-RPS']
    
    # Coletar métricas
    metrics_data = {name: [] for name in metrics_names}
    
    for model_name in model_names:
        if model_name not in models_info:
            continue
        
        model = models_info[model_name]['model']
        y_pred = model.predict(X_season)
        y_proba = model.predict_proba(X_season)
        
        acc = accuracy_score(y_season, y_pred)
        prec = precision_score(y_season, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_season, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_season, y_pred, average='macro', zero_division=0)
        rps = rps_score(y_season.values, y_proba)
        
        metrics_data['Accuracy'].append(acc)
        metrics_data['Precision'].append(prec)
        metrics_data['Recall'].append(rec)
        metrics_data['F1-Score'].append(f1)
        metrics_data['1-RPS'].append(1 - rps)  # Inverter para que maior seja melhor
    
    # Preparar dados para radar
    num_vars = len(metrics_names)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    
    # Cores
    colors = ['#FF6B6B', '#4ECDC4', '#FFD93D', '#6C5CE7']
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for idx, model_name in enumerate(model_names):
        if model_name not in models_info:
            continue
        
        vals = [metrics_data[metric][idx] for metric in metrics_names]
        vals += vals[:1]
        
        ax.plot(angles, vals, 'o-', linewidth=2.5, label=model_name, color=colors[idx])
        ax.fill(angles, vals, alpha=0.15, color=colors[idx])
    
    # Labels e styling
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names, size=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=10, color='gray')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.set_title(f'Decoder Hybrid - {season_name}\n({len(y_season)} jogos)', 
                 size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11, framealpha=0.9)
    
    # Salvar figura
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'radar_chart_hybrid_{season_name.replace(" ", "_")}.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Salvo em: {output_path}")


# Carregar dados
print("📂 Carregando dados...")
train_dir = "data/data_2005_2014"
test_dir = "data/data_2014_2016"

df_train = load_multiple_seasons(train_dir)
df_test = load_multiple_seasons(test_dir)

features_train = calculate_team_stats(df_train)
features_test = calculate_team_stats(df_test)

print(f"   ✓ Teste: {len(features_test)} amostras")
print()

y_test = features_test['Result']
X_test = features_test.drop(['Result', 'Season'], axis=1)

# Carregar modelos
print("📂 Carregando modelos Hybrid...")
hybrid_path = "models/autoencoder_decoder_hybrid/trained_models_hybrid.pkl"

try:
    hybrid_meta = joblib.load(hybrid_path)
    models_info = hybrid_meta['models']
    print(f"   ✓ {len(models_info)} modelos")
except Exception as e:
    print(f"   ❌ Erro: {e}")
    exit(1)

print()

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

print("🔧 Gerando radar charts...")
print()

output_dir = "models/autoencoder_decoder_hybrid/figures"

# Gerar para cada temporada
seasons = {
    '2014-2015': features_test['Season'] == 2015,
    '2015-2016': features_test['Season'] == 2016,
    'All': features_test['Season'].isin([2015, 2016]),
}

for season_name, season_mask in seasons.items():
    generate_radar_chart(season_name, season_mask, models_info, X_test_hybrid, y_test, output_dir)

print("\n" + "="*80)
print("✅ RADAR CHARTS CONCLUÍDOS!")
print("="*80)
