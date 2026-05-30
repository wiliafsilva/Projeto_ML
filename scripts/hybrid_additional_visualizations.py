"""
Visualizações Adicionais - DECODER HYBRID
==========================================

Gera visualizações complementares:
- Comparação de performance por modelo e temporada
- Evolução temporal das métricas
- Distribuição de predições
- Matriz de confusão agregada

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
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from src.preprocessing import load_multiple_seasons
from src.feature_engineering import calculate_team_stats

print("="*80)
print("VISUALIZAÇÕES ADICIONAIS - DECODER HYBRID")
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

y_test = features_test['Result'].reset_index(drop=True)
X_test = features_test.drop(['Result', 'Season'], axis=1).reset_index(drop=True)
seasons = features_test['Season'].reset_index(drop=True)

print(f"   ✓ Teste: {len(X_test)} amostras")
print()

# Carregar modelos
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

X_test_hybrid = np.hstack([
    X_test_latent,
    X_test_reconstructed,
    X_test_reconstruction_error
])

print(f"   ✓ Features: {X_test_hybrid.shape}")
print()

output_dir = "models/autoencoder_decoder_hybrid"
figures_dir = os.path.join(output_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

# ============================================================================
print("🎨 Gerando Visualização 1: Comparação de Performance por Modelo e Temporada")
print()

season_data = []

for model_name in models_info.keys():
    model = models_info[model_name]['model']
    
    # Geral
    y_pred_all = model.predict(X_test_hybrid)
    from sklearn.metrics import accuracy_score, f1_score
    acc_all = accuracy_score(y_test, y_pred_all)
    f1_all = f1_score(y_test, y_pred_all, average='macro', zero_division=0)
    
    season_data.append({
        'Model': model_name,
        'Season': 'All (2014-2016)',
        'Accuracy': acc_all,
        'F1-Score': f1_all,
        'Count': len(y_test)
    })
    
    # Por temporada
    for season in sorted(seasons.unique()):
        mask = seasons == season
        X_season = X_test_hybrid[mask.values]
        y_season = y_test[mask.values]
        
        y_pred = model.predict(X_season)
        acc = accuracy_score(y_season, y_pred)
        f1 = f1_score(y_season, y_pred, average='macro', zero_division=0)
        
        season_data.append({
            'Model': model_name,
            'Season': f'{int(season)}-{int(season)+1}',
            'Accuracy': acc,
            'F1-Score': f1,
            'Count': len(y_season)
        })

season_df = pd.DataFrame(season_data)

fig, ax = plt.subplots(figsize=(12, 6))

models = season_df['Model'].unique()
seasons_list = season_df['Season'].unique()
x = np.arange(len(seasons_list))
width = 0.2

colors = ['#FF6B6B', '#4ECDC4', '#FFD93D', '#6C5CE7']

for i, model in enumerate(models):
    model_data = season_df[season_df['Model'] == model]
    bars = ax.bar(x + i*width, model_data['Accuracy'], width, label=model, 
                  color=colors[i], alpha=0.8, edgecolor='black')
    ax.bar_label(bars, fmt='%.4f', fontsize=9, padding=3)

ax.set_xlabel('Temporada', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Accuracy por Modelo e Temporada (Decoder Hybrid)', 
             fontsize=13, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(seasons_list, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim([0.4, 0.6])

plt.tight_layout()
fig_path1 = os.path.join(figures_dir, "hybrid_performance_by_season.png")
plt.savefig(fig_path1, dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ Salvo: {fig_path1}")

# ============================================================================
print("🎨 Gerando Visualização 2: Matrizes de Confusão (Todos os Modelos)")
print()

models_list = list(models_info.keys())

for model_name in models_list:
    model = models_info[model_name]['model']
    
    y_pred = model.predict(X_test_hybrid)
    cm = confusion_matrix(y_test, y_pred)
    
    acc = np.trace(cm) / np.sum(cm)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Home', 'Draw', 'Away'],
                yticklabels=['Home', 'Draw', 'Away'],
                cbar_kws={'label': 'Count'},
                ax=ax,
                annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    
    ax.set_xlabel('Predito', fontsize=12, fontweight='bold')
    ax.set_ylabel('Verdadeiro', fontsize=12, fontweight='bold')
    ax.set_title(f'Matriz de Confusão - {model_name} (Decoder Hybrid)\nAccuracy: {acc:.4f}', 
                 fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    fig_path = os.path.join(figures_dir, f"hybrid_confusion_matrix_{model_name.lower()}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ {model_name}: Accuracy {acc:.4f}")

# ============================================================================
print("🎨 Gerando Visualização 3: Distribuição de Probabilidades Preditas")
print()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for model_idx, (model_name, ax) in enumerate(zip(models_info.keys(), axes)):
    model = models_info[model_name]['model']
    y_proba = model.predict_proba(X_test_hybrid)
    
    # Probabilidade máxima por predição
    max_proba = np.max(y_proba, axis=1)
    
    ax.hist(max_proba, bins=30, color=['#FF6B6B', '#4ECDC4', '#FFD93D', '#6C5CE7'][model_idx], 
            alpha=0.7, edgecolor='black')
    
    ax.axvline(np.mean(max_proba), color='red', linestyle='--', linewidth=2, 
               label=f'Média: {np.mean(max_proba):.3f}')
    
    ax.set_xlabel('Probabilidade Máxima', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequência', fontsize=10, fontweight='bold')
    ax.set_title(f'{model_name}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle='--')

fig.suptitle('Distribuição de Confiança das Predições (Decoder Hybrid)', 
             fontsize=13, fontweight='bold')
plt.tight_layout()

fig_path3 = os.path.join(figures_dir, "hybrid_prediction_confidence.png")
plt.savefig(fig_path3, dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ Salvo: {fig_path3}")

# ============================================================================
print("🎨 Gerando Visualização 4: Heatmap de Performance")
print()

# Criar matriz de performance
perf_data = []
models_list = list(models_info.keys())
seasons_list = ['2014-2015', '2015-2016', 'All (2014-2016)']

for model in models_list:
    accs = season_df[season_df['Model'] == model]['Accuracy'].values
    perf_data.append(accs)

perf_matrix = np.array(perf_data)

fig, ax = plt.subplots(figsize=(8, 5))

sns.heatmap(perf_matrix, annot=True, fmt='.4f', cmap='RdYlGn', 
            xticklabels=seasons_list,
            yticklabels=models_list,
            cbar_kws={'label': 'Accuracy'},
            ax=ax,
            vmin=0.4, vmax=0.6,
            annot_kws={'fontsize': 11, 'fontweight': 'bold'})

ax.set_xlabel('Temporada', fontsize=12, fontweight='bold')
ax.set_ylabel('Modelo', fontsize=12, fontweight='bold')
ax.set_title('Heatmap de Accuracy por Modelo e Temporada (Decoder Hybrid)', 
             fontsize=13, fontweight='bold')

plt.tight_layout()
fig_path4 = os.path.join(figures_dir, "hybrid_performance_heatmap.png")
plt.savefig(fig_path4, dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ Salvo: {fig_path4}")

# Salvar season_df como CSV para referência
csv_path = os.path.join(output_dir, "hybrid_performance_by_season.csv")
season_df.to_csv(csv_path, index=False)
print(f"   ✓ CSV salvo: {csv_path}")

print()

print("="*80)
print("✅ VISUALIZAÇÕES ADICIONAIS CONCLUÍDAS!")
print("="*80)
print()
print("📊 Arquivos gerados:")
print(f"   ✓ PNG 1: {fig_path1}")
print(f"   ✓ PNG 2: {fig_path2}")
print(f"   ✓ PNG 3: {fig_path3}")
print(f"   ✓ PNG 4: {fig_path4}")
print(f"   ✓ CSV: {csv_path}")
print()
