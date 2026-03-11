#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script para gerar visualizações avançadas para o artigo científico"""

import sys
import os
from pathlib import Path

# Forçar UTF-8 no Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Adicionar o diretório raiz ao path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing import load_all_data, load_multiple_seasons
from src.feature_engineering import calculate_team_stats
from sklearn.metrics import accuracy_score

# Configurar estilo
plt.style.use('default')
sns.set_palette("husl")

print("="*80)
print("GERAÇÃO DE VISUALIZAÇÕES AVANÇADAS PARA ARTIGO CIENTÍFICO")
print("="*80)

# Criar pasta para salvar gráficos
import os
os.makedirs('models/figures', exist_ok=True)

# Carregar dados
df_all = load_all_data()
df_test = load_multiple_seasons("data/data_2014_2016")
features_test = calculate_team_stats(df_test)
features_all = calculate_team_stats(df_all)

X_test = features_test.drop(['Result', 'Season'], axis=1)
y_test = features_test['Result']

# Carregar modelos
try:
    results_metadata = joblib.load("models/trained_models.pkl")
    models = results_metadata.get('models', results_metadata)
except:
    print("\n⚠️  ERRO: Modelos não encontrados. Execute 'python main.py' primeiro.")
    sys.exit(1)

# ============================================================================
# GRÁFICO 1: Comparação Multi-Métrica (Radar Chart)
# ============================================================================
print("\n[1/6] Gerando Radar Chart de Comparação Multi-Métrica...")

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from math import pi

# Coletar métricas
metrics_data = {}
for name, info in models.items():
    model = info['model']
    
    # Filtrar features para corresponder ao modelo
    feature_columns = info.get('feature_columns', None)
    if feature_columns is not None:
        X_test_model = X_test[feature_columns]
    else:
        X_test_model = X_test
    
    preds = model.predict(X_test_model)
    probs = model.predict_proba(X_test_model)
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro', zero_division=0)
    rec = recall_score(y_test, preds, average='macro', zero_division=0)
    f1 = f1_score(y_test, preds, average='macro', zero_division=0)
    
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    try:
        roc_auc = roc_auc_score(y_bin, probs, average='macro', multi_class='ovr')
    except:
        roc_auc = 0.5
    
    # Normalizar RPS (quanto menor, melhor) -> inverter para radar
    rps_normalized = 1 - info.get('rps', 0.5)
    
    metrics_data[name] = [acc, prec, rec, f1, roc_auc, rps_normalized]

# Criar radar chart
categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC', 'RPS (inv.)']
N = len(categories)

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

for name, values in metrics_data.items():
    values += values[:1]  # Fechar o círculo
    ax.plot(angles, values, 'o-', linewidth=2, label=name)
    ax.fill(angles, values, alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=10)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
ax.grid(True)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.set_title('Comparação Multi-Métrica dos Modelos', size=14, y=1.08)

plt.tight_layout()
plt.savefig('models/figures/fig1_radar_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Salvo: models/figures/fig1_radar_comparison.png")
plt.close()

# ============================================================================
# GRÁFICO 2: Heatmap de Correlação entre Features
# ============================================================================
print("\n[2/6] Gerando Heatmap de Correlação entre Features...")

feature_cols = ['gd_diff', 'streak_diff', 'weighted_diff']
corr_matrix = features_all[feature_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlação entre Features', fontsize=14, pad=15)
ax.set_xticklabels(['Goal Diff', 'Streak Diff', 'Weighted Diff'], rotation=45, ha='right')
ax.set_yticklabels(['Goal Diff', 'Streak Diff', 'Weighted Diff'], rotation=0)

plt.tight_layout()
plt.savefig('models/figures/fig2_feature_correlation.png', dpi=300, bbox_inches='tight')
print("✓ Salvo: models/figures/fig2_feature_correlation.png")
plt.close()

# ============================================================================
# GRÁFICO 3: Boxplots de Features por Resultado
# ============================================================================
print("\n[3/6] Gerando Boxplots de Features por Resultado...")

# Mapear resultados
resultado_map = {0: 'Vitória Casa', 1: 'Empate', 2: 'Vitória Visitante'}
features_all['Resultado'] = features_all['Result'].map(resultado_map)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, col in enumerate(feature_cols):
    ax = axes[idx]
    data_to_plot = [features_all[features_all['Result'] == i][col].dropna() for i in range(3)]
    
    bp = ax.boxplot(data_to_plot, tick_labels=['Vitória Casa', 'Empate', 'Vitória Visitante'],
                    patch_artist=True, showmeans=True)
    
    # Colorir boxes
    colors = ['#ff9999', '#ffcc99', '#99ccff']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_title(col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_ylabel('Valor' if idx == 0 else '')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(['Casa', 'Empate', 'Fora'], rotation=15, ha='right')

fig.suptitle('Distribuição das Features por Resultado da Partida', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('models/figures/fig3_boxplots_by_result.png', dpi=300, bbox_inches='tight')
print("✓ Salvo: models/figures/fig3_boxplots_by_result.png")
plt.close()

# ============================================================================
# GRÁFICO 4: Comparação de Importância de Features (RF vs XGBoost)
# ============================================================================
print("\n[4/6] Gerando Comparação de Feature Importance...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, model_name in enumerate(['RandomForest', 'XGBoost']):
    if model_name not in models:
        continue
    
    model = models[model_name]['model']
    
    # Extrair modelo base se for calibrado
    base_model = model
    if hasattr(model, 'calibrated_classifiers_'):
        base_model = model.calibrated_classifiers_[0].estimator
    
    if hasattr(base_model, 'feature_importances_'):
        importances = base_model.feature_importances_
        
        # Usar os nomes reais de features do modelo
        feature_columns = models[model_name].get('feature_columns', [])
        if not feature_columns:
            feature_columns = [f'feature_{i}' for i in range(len(importances))]
        
        # Top 10 features mais importantes
        indices = np.argsort(importances)[::-1][:10]
        top_importances = importances[indices]
        top_features = [feature_columns[i] for i in indices]
        
        ax = axes[idx]
        colors_palette = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_importances)))
        ax.bar(range(len(top_importances)), top_importances, color=colors_palette)
        ax.set_xticks(range(len(top_importances)))
        ax.set_xticklabels(top_features, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Importância')
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Adicionar valores no topo das barras
        for i, v in enumerate(top_importances):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

fig.suptitle('Comparação de Importância das Features', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('models/figures/fig4_feature_importance_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Salvo: models/figures/fig4_feature_importance_comparison.png")
plt.close()

# ============================================================================
# GRÁFICO 5: Comparação de Curvas de Calibração (todos os modelos)
# ============================================================================
print("\n[5/6] Gerando Comparação de Curvas de Calibração...")

from sklearn.calibration import calibration_curve

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
class_names = ['Vitória Casa', 'Empate', 'Vitória Visitante']

for class_idx, class_name in enumerate(class_names):
    ax = axes[class_idx]
    
    y_bin = (y_test == class_idx).astype(int)
    
    for name, info in models.items():
        model = info['model']
        
        # Filtrar features para corresponder ao modelo
        feature_columns = info.get('feature_columns', None)
        if feature_columns is not None:
            X_test_model = X_test[feature_columns]
        else:
            X_test_model = X_test
        
        probs = model.predict_proba(X_test_model)
        prob_pos = probs[:, class_idx]
        
        frac_pos, mean_pred = calibration_curve(y_bin, prob_pos, n_bins=5, strategy='quantile')
        
        ax.plot(mean_pred, frac_pos, marker='o', label=name, linewidth=2.5, markersize=8)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfeitamente Calibrado')
    ax.set_xlabel('Probabilidade Média Predita')
    ax.set_ylabel('Fração de Positivos' if class_idx == 0 else '')
    ax.set_title(class_name, fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

fig.suptitle('Curvas de Calibração por Classe', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('models/figures/fig5_calibration_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Salvo: models/figures/fig5_calibration_comparison.png")
plt.close()

# ============================================================================
# GRÁFICO 6: Comparação de Métricas (Barras Agrupadas)
# ============================================================================
print("\n[6/6] Gerando Gráfico de Barras de Comparação de Métricas...")

# Preparar dados
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics_names))

# Ajuste dinâmico da largura das barras com base no número de modelos
n_models = len(models)
width = 0.8 / n_models if n_models > 0 else 0.25

fig, ax = plt.subplots(figsize=(12, 6))

for idx, (name, info) in enumerate(models.items()):
    model = info['model']
    
    # Filtrar features para corresponder ao modelo
    feature_columns = info.get('feature_columns', None)
    if feature_columns is not None:
        X_test_model = X_test[feature_columns]
    else:
        X_test_model = X_test
    
    preds = model.predict(X_test_model)
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro', zero_division=0)
    rec = recall_score(y_test, preds, average='macro', zero_division=0)
    f1 = f1_score(y_test, preds, average='macro', zero_division=0)
    
    values = [acc, prec, rec, f1]

    # Centralizar os grupos: calcular offset para cada modelo
    offset = (idx - (n_models - 1) / 2) * width
    bars = ax.bar(x + offset, values, width, label=name)

    # Preparar offsets escalonados para reduzir sobreposição entre modelos
    max_stagger = 0.04
    if n_models > 1:
        stagger_offsets = np.linspace(-max_stagger, max_stagger, n_models)
    else:
        stagger_offsets = np.array([0.0])

    base_padding = 0.01
    alt_step = 0.018  # alternância por coluna (metric) para distribuir labels
    for bar, v in zip(bars, values):
        height = bar.get_height()
        # Centralizado, logo acima da barra, fonte menor
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f"{v:.3f}", ha='center', va='bottom', fontsize=7, clip_on=False)

ax.set_xlabel('Métricas', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Comparação de Métricas entre Modelos', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('models/figures/fig6_metrics_comparison_bars.png', dpi=300, bbox_inches='tight')
print("✓ Salvo: models/figures/fig6_metrics_comparison_bars.png")
plt.close()

# ============================================================================
# RESUMO
# ============================================================================
print("\n" + "="*80)
print("✅ TODAS AS VISUALIZAÇÕES FORAM GERADAS COM SUCESSO!")
print("="*80)
print("\nArquivos criados na pasta 'models/figures/' (300 DPI):")
print("  1. fig1_radar_comparison.png - Radar chart multi-métrica")
print("  2. fig2_feature_correlation.png - Heatmap de correlação")
print("  3. fig3_boxplots_by_result.png - Boxplots por resultado")
print("  4. fig4_feature_importance_comparison.png - Importância RF vs XGBoost")
print("  5. fig5_calibration_comparison.png - Curvas de calibração")
print("  6. fig6_metrics_comparison_bars.png - Barras de comparação")
print("\n" + "="*80)
print("📊 INFORMAÇÕES DAS FIGURAS GERADAS")
print("="*80)
print("\n✓ Resolução: 300 DPI (qualidade para publicação)")
print("✓ Formato: PNG com transparência")
print("✓ Tamanho médio: ~100-500 KB por figura")
print("✓ Pronto para: Artigos científicos, apresentações, relatórios")

# Listar tamanhos dos arquivos
import os
print("\n📁 Tamanho dos arquivos:")
for i in range(1, 7):
    fig_path = f'models/figures/fig{i}_*.png'
    import glob
    matching_files = glob.glob(fig_path)
    if matching_files:
        for file in matching_files:
            size_kb = os.path.getsize(file) / 1024
            print(f"   {os.path.basename(file)}: {size_kb:.1f} KB")

print("\n" + "="*80)
print("💡 Para visualizar todas as figuras de forma interativa:")
print("   streamlit run app.py")
print("   → Navegue até 'Análise Científica Consolidada'")
print("="*80)
