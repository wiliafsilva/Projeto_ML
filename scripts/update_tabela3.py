"""
Update Tabela 3 - Comparação Completa de Modelos
================================================

Regenera Tabela 3 usando modelos atuais (trained_models.pkl)
e baseline_comparison.csv já gerado.

Autor: Projeto_ML
Data: Março 2026
"""

import argparse
import os
import pandas as pd
import joblib
import numpy as np

print("="*80)
print("ATUALIZANDO TABELA 3: COMPARAÇÃO COMPLETA DE MODELOS")
print("="*80)
print()


def parse_args():
    parser = argparse.ArgumentParser(description="Atualizar tabela 3")
    parser.add_argument("--model-path", default="models/trained_models.pkl")
    parser.add_argument("--output-dir", default="models")
    return parser.parse_args()


args = parse_args()
output_dir = args.output_dir

# Carregar baseline_comparison.csv
print("📂 Carregando baseline_comparison.csv...")
baseline_df = pd.read_csv(os.path.join(output_dir, 'baseline_comparison.csv'))
print(f"   ✓ {len(baseline_df)} modelos carregados")
print()

# Carregar trained_models.pkl para RPS
print("📂 Carregando trained_models.pkl...")
results_metadata = joblib.load(args.model_path)
models_info = results_metadata['models']
print(f"   ✓ {len(models_info)} modelos treinados")
print()

# Construir Tabela 3
print("🔧 Construindo Tabela 3...")
table3_data = []

# Adicionar Baseline (buscar nome robustamente)
baseline_candidates = baseline_df[baseline_df['Tipo'].str.lower() == 'baseline']
baseline_row = None
if not baseline_candidates.empty:
    # Preferência por nome que contenha 'Most'
    mask_most = baseline_candidates['Modelo'].str.contains('Most', case=False, na=False)
    if mask_most.any():
        baseline_row = baseline_candidates[mask_most].iloc[0]
    else:
        baseline_row = baseline_candidates.iloc[0]

if baseline_row is None:
    print('⚠️ Baseline não encontrado em baseline_comparison.csv — pulando linha baseline.')
else:
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
    # Dados do baseline_comparison.csv — procurar linha do modelo de forma robusta
    model_rows = baseline_df[baseline_df['Modelo'] == model_name]
    if model_rows.empty:
        # tentar busca por substring
        model_rows = baseline_df[baseline_df['Modelo'].str.contains(model_name, case=False, na=False)]

    if model_rows.empty:
        print(f"⚠️ Modelo {model_name} não encontrado em baseline_comparison.csv — pulando.")
        continue

    model_row = model_rows.iloc[0]

    # RPS do trained_models.pkl (proteção caso a chave não exista)
    rps_value = models_info.get(model_name, {}).get('rps', np.nan)

    # F1 pode estar em coluna 'F1' ou 'F1 (macro)'
    if 'F1 (macro)' in model_row.index:
        f1_val = model_row['F1 (macro)']
    elif 'F1' in model_row.index:
        f1_val = model_row['F1']
    else:
        f1_val = np.nan

    table3_data.append({
        'Modelo': model_name,
        'Accuracy': f"{model_row['Accuracy']:.4f}",
        'Precision': f"{model_row.get('Precision', np.nan):.4f}" if pd.notna(model_row.get('Precision', np.nan)) else '-',
        'Recall': f"{model_row.get('Recall', np.nan):.4f}" if pd.notna(model_row.get('Recall', np.nan)) else '-',
        'F1': f"{f1_val:.4f}" if pd.notna(f1_val) else '-',
        'RPS': f"{rps_value:.4f}" if pd.notna(rps_value) else '-',
        'Brier': '-',  # Não calculado ainda
        'ROC AUC': '-'  # Não calculado ainda
    })

# Converter para DataFrame e salvar
table3_df = pd.DataFrame(table3_data)
output_path = os.path.join(output_dir, 'tabela3_comparacao_modelos.csv')
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
