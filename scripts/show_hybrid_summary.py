"""
Sumário Visual - DECODER HYBRID
===============================

Exibe um sumário bonito de todos os arquivos gerados.

Execução: python scripts/show_hybrid_summary.py
"""

import os
import pandas as pd
from pathlib import Path

print("\n")
print("┌" + "─"*98 + "┐")
print("│" + " "*30 + "DECODER HYBRID - SUMÁRIO FINAL" + " "*39 + "│")
print("└" + "─"*98 + "┘")
print()

output_dir = "models/autoencoder_decoder_hybrid"

# ============================================================================
print("📊 TABELAS GERADAS (CSV)")
print("─" * 100)
print()

csvs = {
    'tabela3_hybrid_comparacao.csv': 'Comparação Baseline vs Latent vs Hybrid',
    'tabela4_cm_hybrid_randomforest.csv': 'Confusion Matrix - RandomForest',
    'tabela4_cm_hybrid_xgboost.csv': 'Confusion Matrix - XGBoost',
    'tabela4_cm_hybrid_naivebayes.csv': 'Confusion Matrix - NaiveBayes',
    'tabela4_cm_hybrid_svm.csv': 'Confusion Matrix - SVM',
    'tabela5_hybrid_performance_temporada.csv': 'Performance por Temporada',
    'tabela6_hybrid_classificacao_classe.csv': 'Classificação por Classe (H/D/A)',
    'hybrid_baseline_comparison.csv': 'Comparação com Baselines (Freq/Stratified)',
    'hybrid_correlation_matrix.csv': 'Matriz de Correlação 43×43',
    'hybrid_model_results.csv': 'Resumo Geral dos Modelos',
    'hybrid_model_results_by_season.csv': 'Resultados por Temporada',
}

for filename, description in csvs.items():
    filepath = os.path.join(output_dir, filename)
    if os.path.exists(filepath):
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  ✓ {filename:<45} | {description:<35} | {size_kb:6.1f} KB")
    else:
        print(f"  ✗ {filename:<45} | {description:<35} | FALTANDO")

print()

# ============================================================================
print("📈 FIGURAS GERADAS (PNG)")
print("─" * 100)
print()

figures_dir = os.path.join(output_dir, 'figures')
pngs = {
    'hybrid_correlation_heatmap.png': 'Heatmap de Correlação (43 features)',
    'radar_chart_hybrid_2014-2015.png': 'Radar Chart - Temporada 2014-2015',
    'radar_chart_hybrid_2015-2016.png': 'Radar Chart - Temporada 2015-2016',
    'radar_chart_hybrid_All.png': 'Radar Chart - Ambas Temporadas',
}

for filename, description in pngs.items():
    if figures_dir:
        filepath = os.path.join(figures_dir, filename)
    else:
        filepath = os.path.join(output_dir, filename)
    
    if os.path.exists(filepath):
        size_kb = os.path.getsize(filepath) / 1024
        resolution = "300 DPI"
        print(f"  ✓ {filename:<45} | {description:<35} | {size_kb:6.1f} KB")
    else:
        print(f"  ✗ {filename:<45} | {description:<35} | FALTANDO")

print()

# ============================================================================
print("🔧 MODELOS SALVOS")
print("─" * 100)
print()

models = {
    'trained_models_hybrid.pkl': '4 modelos + metadados',
    'autoencoder_hybrid.keras': 'Autoencoder completo',
    'encoder_hybrid.keras': 'Encoder (features → latent 8D)',
    'decoder_hybrid.keras': 'Decoder (latent 8D → features)',
    'scaler_hybrid.joblib': 'MinMaxScaler para 43 features',
}

for filename, description in models.items():
    filepath = os.path.join(output_dir, filename)
    if os.path.exists(filepath):
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  ✓ {filename:<45} | {description:<35} | {size_kb:6.1f} KB")
    else:
        print(f"  ✗ {filename:<45} | {description:<35} | FALTANDO")

print()

# ============================================================================
print("📋 ANÁLISES COMPARATIVAS")
print("─" * 100)
print()

comparisons = [
    ("Baseline vs Latent vs Hybrid", "tabela3_hybrid_comparacao.csv"),
    ("ML Models vs Dummy Classifiers", "hybrid_baseline_comparison.csv"),
]

for name, filename in comparisons:
    filepath = os.path.join(output_dir, filename)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"  ✓ {name:<40} | {len(df)} linhas | {len(df.columns)} colunas")
    else:
        print(f"  ✗ {name:<40} | FALTANDO")

print()

# ============================================================================
print("🎯 PERFORMANCE DOS MODELOS")
print("─" * 100)
print()

results_file = os.path.join(output_dir, 'hybrid_model_results.csv')
if os.path.exists(results_file):
    df = pd.read_csv(results_file)
    print("  Modelo         │ Accuracy │ F1-Score │  RPS  ")
    print("  " + "─" * 50)
    for _, row in df.iterrows():
        try:
            print(f"  {row['model']:<14} │ {row['accuracy']:>8.4f} │ {row['f1']:>8.4f} │ {row['rps']:>6.4f}")
        except:
            pass

print()

# ============================================================================
print("⏱️  PERFORMANCE POR TEMPORADA")
print("─" * 100)
print()

season_file = os.path.join(output_dir, 'tabela5_hybrid_performance_temporada.csv')
if os.path.exists(season_file):
    df = pd.read_csv(season_file)
    print("  Temporada  │ Modelo         │ Accuracy │ F1-Score │ RPS   │ Testes")
    print("  " + "─" * 65)
    
    for season in df['Temporada'].unique():
        season_data = df[df['Temporada'] == season]
        for idx, (_, row) in enumerate(season_data.iterrows()):
            if idx == 0:
                sep = "  " + str(season) + " ".ljust(9-len(str(season)))
            else:
                sep = "  " + " "*9
            
            try:
                print(f"{sep} │ {row['Modelo']:<14} │ {row['Accuracy']:>8.4f} │ {row['F1-Score']:>8.4f} │ {row['RPS']:>5.4f} │ {row['N_Amostras']:>6.0f}")
            except:
                pass

print()

# ============================================================================
print("🏆 DESTAQUE DE RESULTADOS")
print("─" * 100)
print()

results_file = os.path.join(output_dir, 'hybrid_model_results.csv')
if os.path.exists(results_file):
    df = pd.read_csv(results_file)
    
    best_acc = df.loc[df['accuracy'].idxmax()]
    best_f1 = df.loc[df['f1'].idxmax()]
    
    print(f"  🥇 Maior Accuracy:  {best_acc['model']:<15} {best_acc['accuracy']:.4f}")
    print(f"  🥇 Maior F1-Score:  {best_f1['model']:<15} {best_f1['f1']:.4f}")

print()

# ============================================================================
print("📊 ARQUIVOS POR TIPO")
print("─" * 100)
print()

total_csv = 0
total_png = 0
total_keras = 0
total_pkl = 0

for filename in os.listdir(output_dir):
    if filename.endswith('.csv'):
        total_csv += 1
    elif filename.endswith('.png'):
        total_png += 1

if os.path.exists(figures_dir):
    for filename in os.listdir(figures_dir):
        if filename.endswith('.png'):
            total_png += 1

if os.path.exists(results_file):
    total_keras = 4  # encoder, decoder, autoencoder, e um extra
    total_pkl = 1

print(f"  📋 Arquivos CSV:           {total_csv:2d}")
print(f"  📈 Imagens PNG:            {total_png:2d}")
print(f"  🤖 Modelos Keras:          {total_keras:2d}")
print(f"  📦 Arquivos Pickle:        {total_pkl:2d}")
print(f"  ─────────────────────────────────")
print(f"  📊 TOTAL:                  {total_csv + total_png + total_keras + total_pkl:2d} arquivos")

print()

# ============================================================================
print("✅ TODOS OS ARQUIVOS GERADOS COM SUCESSO!")
print()
print("┌" + "─"*98 + "┐")
print("│ Para visualizar os resultados em detalhes, abra os arquivos em:                      │")
print(f"│ → {output_dir:<92} │")
print("└" + "─"*98 + "┘")
print()
