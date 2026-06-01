"""
Confidence Intervals Bootstrap - DECODER HYBRID
================================================

Calcula intervalos de confiança (95%) via bootstrap com 100 iterações
para validar estabilidade dos modelos Hybrid.

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
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from src.preprocessing import load_multiple_seasons
from src.feature_engineering import calculate_team_stats

print("="*80)
print("INTERVALOS DE CONFIANÇA (BOOTSTRAP) - DECODER HYBRID")
print("="*80)
print()

# Configuração bootstrap
N_ITERATIONS = 100
CONFIDENCE_LEVEL = 95
RANDOM_STATE = 42

print(f"Configuração:")
print(f"   Iterações: {N_ITERATIONS}")
print(f"   Confiança: {CONFIDENCE_LEVEL}%")
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

print(f"   ✓ Teste: {len(X_test)} amostras")
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

# UPDATED: usar apenas as features reconstruídas do decoder
X_test_hybrid = X_test_reconstructed

print(f"   ✓ Features: {X_test_hybrid.shape}")
print()

def rps_score(y_true, y_proba):
    """Ranked Probability Score"""
    n_classes = y_proba.shape[1]
    rps = 0
    for i, true_class in enumerate(y_true):
        y_true_cdf = np.zeros(n_classes)
        y_true_cdf[int(true_class):] = 1
        y_pred_cdf = np.cumsum(y_proba[i])
        rps += np.sum((y_pred_cdf - y_true_cdf) ** 2)
    return rps / len(y_true)

# Bootstrap
print("🔄 Executando bootstrap...")
print()

results = {model_name: {'accuracy': [], 'f1': [], 'rps': []} for model_name in models_info.keys()}

np.random.seed(RANDOM_STATE)

for iteration in range(N_ITERATIONS):
    # Resample
    indices = resample(range(len(X_test_hybrid)), n_samples=len(X_test_hybrid), random_state=RANDOM_STATE + iteration)
    
    X_boot = X_test_hybrid[indices]
    y_boot = y_test.iloc[indices].values
    
    # Avaliar cada modelo
    for model_name, model_data in models_info.items():
        model = model_data['model']
        
        y_pred = model.predict(X_boot)
        y_proba = model.predict_proba(X_boot)
        
        acc = accuracy_score(y_boot, y_pred)
        f1 = f1_score(y_boot, y_pred, average='macro', zero_division=0)
        rps = rps_score(y_boot, y_proba)
        
        results[model_name]['accuracy'].append(acc)
        results[model_name]['f1'].append(f1)
        results[model_name]['rps'].append(rps)
    
    if (iteration + 1) % 20 == 0:
        print(f"   ✓ Iteração {iteration + 1}/{N_ITERATIONS}")

print()

# Calcular intervalos
print("="*80)
print("INTERVALOS DE CONFIANÇA (95%)")
print("="*80)
print()

ci_data = []

for model_name in models_info.keys():
    for metric in ['accuracy', 'f1', 'rps']:
        values = np.array(results[model_name][metric])
        
        mean = np.mean(values)
        std = np.std(values)
        ci_lower = np.percentile(values, (100 - CONFIDENCE_LEVEL) / 2)
        ci_upper = np.percentile(values, 100 - (100 - CONFIDENCE_LEVEL) / 2)
        margin = (ci_upper - ci_lower) / 2
        
        ci_data.append({
            'Model': model_name,
            'Metric': metric.upper(),
            'Mean': mean,
            'Std': std,
            'CI_Lower': ci_lower,
            'CI_Upper': ci_upper,
            'Margin': margin
        })

ci_df = pd.DataFrame(ci_data)

print(f"{'Model':<15} {'Metric':<12} {'Mean':>10} {'Std':>10} {'CI_Lower':>10} {'CI_Upper':>10} {'Margin':>10}")
print("-"*90)

for _, row in ci_df.iterrows():
    print(f"{row['Model']:<15} {row['Metric']:<12} {row['Mean']:>10.4f} {row['Std']:>10.4f} {row['CI_Lower']:>10.4f} {row['CI_Upper']:>10.4f} {row['Margin']:>10.4f}")

print()

# Salvar CSV
output_dir = "models/autoencoder_decoder_hybrid"
csv_path = os.path.join(output_dir, "hybrid_confidence_intervals.csv")
ci_df.to_csv(csv_path, index=False)
print(f"💾 Salvo em: {csv_path}")
print()

# Visualizações
print("🎨 Gerando visualizações...")

# Plot 1: Boxplots por modelo
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ['accuracy', 'f1', 'rps']
titles = ['Accuracy', 'F1-Score', 'RPS']

for ax, metric, title in zip(axes, metrics, titles):
    data_to_plot = [results[model][metric] for model in models_info.keys()]
    
    bp = ax.boxplot(data_to_plot, labels=models_info.keys(), patch_artist=True)
    
    colors = ['#FF6B6B', '#4ECDC4', '#FFD93D', '#6C5CE7']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel(title, fontsize=11, fontweight='bold')
    ax.set_title(f'{title} - Bootstrap Distribution\n(N={N_ITERATIONS} iterações)', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([np.min([np.min(d) for d in data_to_plot]) - 0.02,
                 np.max([np.max(d) for d in data_to_plot]) + 0.02])

plt.tight_layout()

figures_dir = os.path.join(output_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)
fig_path1 = os.path.join(figures_dir, "hybrid_confidence_intervals_boxplot.png")
plt.savefig(fig_path1, dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ Salvo: {fig_path1}")

# Plot 2: Barplot com error bars
fig, ax = plt.subplots(figsize=(12, 6))

metrics_summary = ci_df[ci_df['Metric'] == 'ACCURACY']
x_pos = np.arange(len(metrics_summary))

ax.bar(x_pos, metrics_summary['Mean'], 
       yerr=metrics_summary['Margin'],
       capsize=10,
       color=['#FF6B6B', '#4ECDC4', '#FFD93D', '#6C5CE7'],
       alpha=0.8,
       edgecolor='black',
       error_kw={'elinewidth': 2, 'capthick': 2})

ax.set_xticks(x_pos)
ax.set_xticklabels(metrics_summary['Model'], fontsize=11, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title(f'Accuracy com Intervalos de Confiança 95% (Bootstrap N={N_ITERATIONS})', 
             fontsize=13, fontweight='bold')
ax.set_ylim([0.4, 0.6])
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Adicionar valores nos tops
for i, (x, y) in enumerate(zip(x_pos, metrics_summary['Mean'])):
    ax.text(x, y + metrics_summary.iloc[i]['Margin'] + 0.01, f'{y:.4f}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()

fig_path2 = os.path.join(figures_dir, "hybrid_confidence_intervals_barplot.png")
plt.savefig(fig_path2, dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ Salvo: {fig_path2}")

# Plot 3: Distribuições histogramas
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

model_names = list(models_info.keys())
colors = ['#FF6B6B', '#4ECDC4', '#FFD93D', '#6C5CE7']

for idx, (model_name, color) in enumerate(zip(model_names, colors)):
    ax = axes[idx]
    
    acc_values = results[model_name]['accuracy']
    
    ax.hist(acc_values, bins=20, color=color, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(acc_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(acc_values):.4f}')
    ax.axvline(np.percentile(acc_values, 2.5), color='green', linestyle=':', linewidth=2, label='CI 95%')
    ax.axvline(np.percentile(acc_values, 97.5), color='green', linestyle=':', linewidth=2)
    
    ax.set_xlabel('Accuracy', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax.set_title(f'{model_name} (N={N_ITERATIONS} bootstrap samples)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linestyle='--')

plt.suptitle('Distribuição de Accuracy - Bootstrap (Decoder Hybrid)', 
             fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()

fig_path3 = os.path.join(figures_dir, "hybrid_confidence_intervals_distribution.png")
plt.savefig(fig_path3, dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ Salvo: {fig_path3}")
print()

print("="*80)
print("✅ BOOTSTRAP CONFIDENCE INTERVALS CONCLUÍDO!")
print("="*80)
print()
print("📊 Arquivos gerados:")
print(f"   ✓ CSV: {csv_path}")
print(f"   ✓ PNG 1: {fig_path1}")
print(f"   ✓ PNG 2: {fig_path2}")
print(f"   ✓ PNG 3: {fig_path3}")
print()
