"""
Update Tabela 3 - Comparação Completa de Modelos
================================================

Regenera Tabela 3 usando modelos atuais (trained_models.pkl)
e baseline_comparison.csv já gerado.

Autor: Projeto_ML
Data: Março 2026
"""

import pandas as pd
import joblib
import numpy as np

print("="*80)
print("ATUALIZANDO TABELA 3: COMPARAÇÃO COMPLETA DE MODELOS")
print("="*80)
print()

# Carregar baseline_comparison.csv
print("📂 Carregando baseline_comparison.csv...")
baseline_df = pd.read_csv('models/baseline_comparison.csv')
print(f"   ✓ {len(baseline_df)} modelos carregados")
print()

# Carregar trained_models.pkl para RPS
print("📂 Carregando trained_models.pkl...")
results_metadata = joblib.load('models/trained_models.pkl')
models_info = results_metadata['models']
print(f"   ✓ {len(models_info)} modelos treinados")
print()

# Construir Tabela 3
print("🔧 Construindo Tabela 3...")
table3_data = []

# Adicionar Baseline
baseline_row = baseline_df[baseline_df['Modelo'] == 'Baseline (Most Frequent)'].iloc[0]
table3_data.append({
    'Modelo': 'Baseline (Majoritário)',
    'Accuracy': f"{baseline_row['Accuracy']:.4f}",
    'Precision': '-',
    'Recall': '-',
    'F1': '-',
    'RPS': '-',
    'Brier': '-',
    'ROC AUC': '-'
})

# Adicionar modelos ML
ml_models = ['RandomForest', 'XGBoost', 'NaiveBayes', 'SVM']
for model_name in ml_models:
    # Dados do baseline_comparison.csv
    model_row = baseline_df[baseline_df['Modelo'] == model_name].iloc[0]
    
    # RPS do trained_models.pkl
    rps_value = models_info[model_name]['rps']
    
    table3_data.append({
        'Modelo': model_name,
        'Accuracy': f"{model_row['Accuracy']:.4f}",
        'Precision': f"{model_row['Precision']:.4f}",
        'Recall': f"{model_row['Recall']:.4f}",
        'F1': f"{model_row['F1 (macro)']:.4f}",
        'RPS': f"{rps_value:.4f}",
        'Brier': '-',  # Não calculado ainda
        'ROC AUC': '-'  # Não calculado ainda
    })

# Converter para DataFrame e salvar
table3_df = pd.DataFrame(table3_data)
output_path = 'models/tabela3_comparacao_modelos.csv'
table3_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print()
print("="*80)
print("TABELA 3 ATUALIZADA")
print("="*80)
print(table3_df.to_string(index=False))
print()
print(f"💾 Salva em: {output_path}")
print()
print("✅ TABELA 3 REGENERADA COM SUCESSO!")
print("="*80)
