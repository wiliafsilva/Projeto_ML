"""
Correlation Heatmap - Feature Analysis
=======================================

Gera heatmap de correlação entre todas as 43 features para identificar:
- Multicolinearidade (correlações > 0.8)
- Features redundantes
- Grupos de features relacionadas

Autor: Projeto_ML
Data: Março 2026
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing import load_all_data
from src.feature_engineering import calculate_team_stats

print("="*80)
print("CORRELATION HEATMAP - ANÁLISE DE FEATURES")
print("="*80)
print()

# Carregar dados e calcular features
print("📂 Carregando dados e calculando features...")
df_all = load_all_data()
df_features = calculate_team_stats(df_all)

# Remove Result e Season para ter apenas features
feature_cols = [col for col in df_features.columns if col not in ['Result', 'Season']]
X = df_features[feature_cols]

print(f"   ✓ {len(feature_cols)} features")
print(f"   ✓ {len(X)} amostras")
print()

# Calcular matriz de correlação
print("🔧 Calculando matriz de correlação...")
corr_matrix = X.corr()
print(f"   ✓ Matriz {corr_matrix.shape[0]}x{corr_matrix.shape[1]}")
print()

# Identificar correlações altas
print("📊 Análise de Multicolinearidade:")
print("-" * 60)
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr.append({
                'Feature 1': corr_matrix.columns[i],
                'Feature 2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })

if high_corr:
    print(f"   ⚠️  {len(high_corr)} pares com correlação > 0.8:")
    for pair in high_corr[:10]:  # Mostrar top 10
        print(f"      {pair['Feature 1']:<25} ↔ {pair['Feature 2']:<25} : {pair['Correlation']:+.3f}")
    if len(high_corr) > 10:
        print(f"      ... e mais {len(high_corr) - 10} pares")
else:
    print("   ✅ Nenhuma correlação > 0.8 (sem multicolinearidade severa)")
print()

# Salvar matriz de correlação como CSV
corr_path = 'models/correlation_matrix.csv'
corr_matrix.to_csv(corr_path)
print(f"💾 Matriz salva em: {corr_path}")
print()

# Criar heatmap
print("🎨 Gerando visualização...")

# Figura grande para acomodar 43 features
fig, ax = plt.subplots(figsize=(20, 18))

# Heatmap com máscara triangular superior (evitar redundância)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Colormap divergente (azul negativo, branco zero, vermelho positivo)
sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap='coolwarm',
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8, "label": "Correlação de Pearson"},
    annot=False,  # Não anotar (43x43 fica ilegível)
    fmt='.2f',
    ax=ax
)

# Títulos e labels
ax.set_title('Correlation Heatmap - 43 Features', fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('Features', fontsize=12, fontweight='bold')
ax.set_ylabel('Features', fontsize=12, fontweight='bold')

# Rotacionar labels para legibilidade
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)

# Layout ajustado
plt.tight_layout()

# Salvar figura
output_path = 'models/figures/correlation_heatmap.png'
os.makedirs('models/figures', exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"📊 Heatmap salvo em: {output_path}")
print()

# Estatísticas sobre correlações
print("📈 ESTATÍSTICAS DA MATRIZ DE CORRELAÇÃO:")
print("-" * 60)

# Flatten a matriz (pegando apenas triângulo inferior para evitar duplicatas)
mask_lower = np.tril(np.ones_like(corr_matrix, dtype=bool), k=-1)
corr_values = corr_matrix.values[mask_lower]

print(f"   Total de pares de features: {len(corr_values)}")
print(f"   Correlação média (absoluta): {np.mean(np.abs(corr_values)):.3f}")
print(f"   Correlação máxima: {np.max(corr_values):+.3f}")
print(f"   Correlação mínima: {np.min(corr_values):+.3f}")
print()

# Distribuição de correlações
n_high = np.sum(np.abs(corr_values) > 0.8)
n_medium = np.sum((np.abs(corr_values) > 0.5) & (np.abs(corr_values) <= 0.8))
n_low = np.sum((np.abs(corr_values) > 0.3) & (np.abs(corr_values) <= 0.5))
n_very_low = np.sum(np.abs(corr_values) <= 0.3)

print(f"   Pares com |corr| > 0.8 (alta):         {n_high:4d} ({n_high/len(corr_values)*100:.1f}%)")
print(f"   Pares com 0.5 < |corr| ≤ 0.8 (média):  {n_medium:4d} ({n_medium/len(corr_values)*100:.1f}%)")
print(f"   Pares com 0.3 < |corr| ≤ 0.5 (baixa):  {n_low:4d} ({n_low/len(corr_values)*100:.1f}%)")
print(f"   Pares com |corr| ≤ 0.3 (muito baixa): {n_very_low:4d} ({n_very_low/len(corr_values)*100:.1f}%)")
print()

print("="*80)
print("✅ CORRELATION HEATMAP CONCLUÍDO!")
print("="*80)
print()
print("💡 PRÓXIMOS PASSOS:")
print("   - Revisar features com alta correlação (> 0.8)")
print("   - Considerar remover features redundantes")
print("   - Calcular VIF (Variance Inflation Factor) para análise detalhada")
