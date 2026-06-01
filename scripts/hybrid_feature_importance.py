"""
Feature Importance - DECODER HYBRID
====================================

Gera ranking de importância das 43 features reconstruídas do pipeline Hybrid
 (USANDO APENAS O DECODER).

Autor: Projeto_ML
Data: Maio 2026
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from src.preprocessing import load_multiple_seasons
from src.feature_engineering import calculate_team_stats

print("="*80)
print("FEATURE IMPORTANCE - DECODER HYBRID (43D Reconstructed Features)")
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

y_test = features_test['Result']
X_test = features_test.drop(['Result', 'Season'], axis=1)

print(f"   ✓ Teste: {len(X_test)} amostras, {X_test.shape[1]} features")
print()

# Carregar modelos e autoencoder
print("📂 Carregando modelos Hybrid...")
hybrid_path = "models/autoencoder_decoder_hybrid/trained_models_hybrid.pkl"
scaler_path = "models/autoencoder_decoder_hybrid/scaler_hybrid.joblib"
encoder_path = "models/autoencoder_decoder_hybrid/encoder_hybrid.keras"
decoder_path = "models/autoencoder_decoder_hybrid/decoder_hybrid.keras"

hybrid_meta = joblib.load(hybrid_path)
models_info = hybrid_meta['models']

scaler = joblib.load(scaler_path)
encoder = tf.keras.models.load_model(encoder_path)
decoder = tf.keras.models.load_model(decoder_path)

print(f"   ✓ {len(models_info)} modelos carregados")
print()

# Preparar features híbridas
print("🔧 Preparando features híbridas...")
X_test_scaled = scaler.transform(X_test.values.astype(np.float32))
X_test_latent = encoder(X_test_scaled).numpy()
X_test_reconstructed = decoder(X_test_latent).numpy()
X_test_reconstruction_error = np.mean(np.abs(X_test_scaled - X_test_reconstructed), axis=1, keepdims=True)

X_test_hybrid = X_test_reconstructed

print(f"   ✓ Features Hybrid: {X_test_hybrid.shape}")
print()

print("="*80)
print("FEATURE IMPORTANCE - RandomForest (Reconstructed Only)")
print("="*80)
print()

# Extrair RandomForest
rf_model = models_info['RandomForest']['model']

# Se for CalibratedClassifierCV, extrair modelo base
if hasattr(rf_model, 'calibrated_classifiers_'):
    rf_base = rf_model.calibrated_classifiers_[0].estimator
    print("ℹ️  Extraindo RandomForest de CalibratedClassifierCV")
else:
    rf_base = rf_model

# Feature importance
importances = rf_base.feature_importances_

print(f"ℹ️  Importances size: {len(importances)}")

importances = rf_base.feature_importances_

print(f"ℹ️  Importances size: {len(importances)}")

# Mapear todos como 'Reconstructed' (apenas saídas do decoder são usadas)
importance_type = ['Reconstructed' for _ in range(len(importances))]

# Criar feature names simplificados
simple_features = [f'Recon_{i}' for i in range(len(importances))]

# DataFrame
importance_df = pd.DataFrame({
    'Feature': simple_features,
    'Importance': importances,
    'Type': importance_type
}).sort_values('Importance', ascending=False).reset_index(drop=True)

print()
print("="*80)
print("TOP 20 FEATURES")
print("="*80)
print()
print(f"{'Rank':<6} {'Feature':<35} {'Type':<15} {'Importance':<12} {'Bar'}")
print("-"*100)

max_imp = importance_df['Importance'].max()
for idx, row in importance_df.head(20).iterrows():
    bar_len = int(row['Importance'] / max_imp * 30)
    print(f"#{idx+1:<5} {row['Feature']:<35} {row['Type']:<15} {row['Importance']:.6f}     {'='*bar_len}")

print()
print("="*80)
print("IMPORTÂNCIA POR GRUPO")
print("="*80)
print()

for ftype in ['Reconstructed']:
    type_data = importance_df[importance_df['Type'] == ftype]
    total_imp = type_data['Importance'].sum()
    pct = total_imp / importance_df['Importance'].sum() * 100
    print(f"{ftype:<20} | Importância Total: {total_imp:.6f} ({pct:5.1f}%) | {len(type_data):2d} features")

print()

 # Salvar como CSV
output_dir = "models/autoencoder_decoder_hybrid"
csv_path = os.path.join(output_dir, "hybrid_feature_importance.csv")
importance_df.to_csv(csv_path, index=False)
print(f"💾 Salvo em: {csv_path}")
print()

# Gerar visualização 1: Top 20 features
print("🎨 Gerando visualização 1: Top 20 Features...")

fig, ax = plt.subplots(figsize=(12, 8))

top_20 = importance_df.head(20).iloc[::-1]  # Invertido para melhor visualização

colors = {
    'Latent': '#FF6B6B',
    'Reconstructed': '#4ECDC4',
    'Error': '#FFD93D'
}

bar_colors = [colors[t] for t in top_20['Type']]

bars = ax.barh(range(len(top_20)), top_20['Importance'].values, color=bar_colors, alpha=0.8, edgecolor='black')

ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20['Feature'].values, fontsize=10)
ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance - Top 20 (Decoder Hybrid - RandomForest)', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#4ECDC4', edgecolor='black', label='Reconstructed (43 features)')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

plt.tight_layout()

figures_dir = os.path.join(output_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)
fig_path = os.path.join(figures_dir, "hybrid_feature_importance_top20.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ Salvo em: {fig_path}")
print()

# Gerar visualização 2: Importância por grupo
print("🎨 Gerando visualização 2: Importância por Tipo...")

# Determinar dinamicamente os tipos presentes (ex.: Reconstructed, Latent, Error)
types_to_plot = importance_df['Type'].unique().tolist()

# Criar subplots dinamicamente para evitar eixos vazios
n_types = len(types_to_plot)
fig_width = max(6, 5 * n_types)
fig, axes = plt.subplots(1, n_types, figsize=(fig_width, 5))
if n_types == 1:
    axes = [axes]

for ax, ftype in zip(axes, types_to_plot):
    type_data = importance_df[importance_df['Type'] == ftype].head(15).sort_values('Importance', ascending=True)

    ax.barh(range(len(type_data)), type_data['Importance'].values, 
            color=colors.get(ftype, '#4ECDC4'), alpha=0.8, edgecolor='black')

    ax.set_yticks(range(len(type_data)))
    ax.set_yticklabels(type_data['Feature'].values, fontsize=9)
    ax.set_xlabel('Importance', fontsize=10, fontweight='bold')
    ax.set_title(f'{ftype}\n({len(importance_df[importance_df["Type"] == ftype])} features)', 
                 fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

fig.suptitle('Feature Importance por Tipo (Decoder Hybrid - RandomForest)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()

fig_path2 = os.path.join(figures_dir, "hybrid_feature_importance_by_type.png")
plt.savefig(fig_path2, dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ Salvo em: {fig_path2}")
print()

# Gerar visualização 3: Distribuição de importância
print("🎨 Gerando visualização 3: Distribuição de Importância...")

fig, ax = plt.subplots(figsize=(12, 6))

for ftype in types_to_plot:
    type_data = importance_df[importance_df['Type'] == ftype]
    ax.hist(type_data['Importance'], bins=20, alpha=0.8, label=ftype, color=colors.get(ftype, '#4ECDC4'))

ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Distribuição de Importância por Tipo (Decoder Hybrid)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3, linestyle='--')

fig_path3 = os.path.join(figures_dir, "hybrid_feature_importance_distribution.png")
plt.savefig(fig_path3, dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ Salvo em: {fig_path3}")
print()

print("="*80)
print("✅ FEATURE IMPORTANCE CONCLUÍDO!")
print("="*80)
print()
print("📊 Arquivos gerados:")
print(f"   ✓ CSV: {csv_path}")
print(f"   ✓ PNG 1: {fig_path}")
print(f"   ✓ PNG 2: {fig_path2}")
print(f"   ✓ PNG 3: {fig_path3}")
print()
