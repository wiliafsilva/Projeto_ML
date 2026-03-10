"""
Radar Chart - Multi-Metric Model Comparison
===========================================

Cria radar chart (spider plot) comparando 4 modelos em 5 dimensões:
- Accuracy
- F1-Score (macro)
- Precision (macro)
- Recall (macro)
- RPS (invertido: 1-RPS para que maior = melhor)

Autor: Projeto_ML
Data: Março 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from math import pi

print("="*80)
print("RADAR CHART - COMPARAÇÃO MULTI-MÉTRICA")
print("="*80)
print()

# Carregar baseline_comparison.csv
print("📂 Carregando dados...")
df = pd.read_csv('models/baseline_comparison.csv')

# Filtrar apenas modelos ML (remover baselines)
df_ml = df[df['Tipo'] == 'ML'].copy()

# Carregar RPS do trained_models.pkl
results_metadata = joblib.load('models/trained_models.pkl')
models_info = results_metadata['models']

# Adicionar coluna RPS
df_ml['RPS'] = df_ml['Modelo'].apply(lambda x: models_info[x]['rps'])

print(f"   ✓ {len(df_ml)} modelos ML")
print()

# Preparar dados para radar
models = df_ml['Modelo'].tolist()
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 (macro)', 'RPS']

# Extrair valores
data = []
for metric in metrics:
    if metric == 'RPS':
        # Inverter RPS: menor RPS = melhor, então usar (1 - RPS) para radar
        # Normalizar para [0, 1]: assumindo RPS típico entre 0.3-0.6
        values = [1 - float(val) for val in df_ml[metric]]
    else:
        values = df_ml[metric].values
    data.append(values)

# Converter para numpy array
data = np.array(data)

print("🎨 Gerando Radar Chart...")

# Número de variáveis
num_vars = len(metrics)

# Ângulos para cada eixo (dividir círculo em partes iguais)
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]  # Fechar o círculo

# Figura
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Cores para cada modelo
colors = ['#FF6B6B', '#4ECDC4', '#FFD93D', '#6C5CE7']

# Loop através dos modelos
for idx, model in enumerate(models):
    values = data[:, idx].tolist()
    values += values[:1]  # Fechar o círculo
    
    ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx])
    ax.fill(angles, values, alpha=0.15, color=colors[idx])

# Labels dos eixos
ax.set_xticks(angles[:-1])
metric_labels_adjusted = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'RPS\n(1-RPS)']
ax.set_xticklabels(metric_labels_adjusted, size=12, fontweight='bold')

# Limites dos eixos (0 a 1 normalizado)
ax.set_ylim(0, 1)

# Grid
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=10, color='gray')
ax.grid(True, linestyle='--', alpha=0.7)

# Título e legenda
ax.set_title('Radar Chart - Comparação Multi-Métrica dos Modelos ML', 
             size=16, fontweight='bold', pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

# Salvar
output_path = 'models/figures/radar_chart.png'
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"📊 Salvo em: {output_path}")
plt.close()
print()

# Tabela com valores usados
print("📊 VALORES USADOS NO RADAR CHART:")
print("="*80)
print(f"{'Modelo':<15} {'Accuracy':>10} {'Precision':>11} {'Recall':>10} {'F1-Score':>10} {'1-RPS':>10}")
print("-"*80)
for i, model in enumerate(models):
    print(f"{model:<15} {data[0,i]:>10.4f} {data[1,i]:>11.4f} {data[2,i]:>10.4f} {data[3,i]:>10.4f} {data[4,i]:>10.4f}")
print()

print("="*80)
print("✅ RADAR CHART CONCLUÍDO!")
print("="*80)
print()
print("💡 INTERPRETAÇÃO:")
print("   - Polígono maior = melhor performance geral")
print("   - Polígono simétrico = desempenho balanceado")
print("   - Polígono assimétrico = forte em algumas métricas, fraco em outras")
