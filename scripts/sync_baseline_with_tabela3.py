"""
Sincroniza baseline_comparison.csv com tabela3_comparacao_modelos.csv
======================================================================
Este script atualiza a temporada 'All' no baseline_comparison.csv
com os valores recém-calculados da tabela3_comparacao_modelos.csv.
"""

import sys
import io

# Forçar UTF-8 no Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import os

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Ler tabela3 (valores corretos recém-calculados)
tabela3_path = os.path.join(BASE, 'models', 'tabela3_comparacao_modelos.csv')
baseline_path = os.path.join(BASE, 'models', 'baseline_comparison.csv')

print('='*70)
print('SINCRONIZAÇÃO DE baseline_comparison.csv com tabela3')
print('='*70)

if not os.path.exists(tabela3_path):
    print(f'Erro: {tabela3_path} não encontrado.')
    print('Execute primeiro: python scripts/generate_tables.py')
    exit(1)

if not os.path.exists(baseline_path):
    print(f'Erro: {baseline_path} não encontrado.')
    exit(1)

# Ler ambos os CSVs
df_tabela3 = pd.read_csv(tabela3_path)
df_baseline = pd.read_csv(baseline_path)

print(f'\n📂 Lendo {tabela3_path}')
print(f'   Modelos encontrados: {list(df_tabela3["Modelo"])}')

print(f'\n📂 Lendo {baseline_path}')
print(f'   Temporadas: {list(df_baseline["Temporada"].unique())}')

# Map model names (tabela3 pode ter nomes ligeiramente diferentes)
model_map = {
    'SVM': 'SVM',
    'RandomForest': 'RandomForest',
    'XGBoost': 'XGBoost',
    'NaiveBayes': 'NaiveBayes',
    'Voting_Equal': 'Voting_Equal',
    'Voting_Weighted': 'Voting_Weighted',
    'Stacking': 'Stacking'
}

# Atualizar apenas a temporada "All"
print('\n🔄 Atualizando temporada "All"...')

updated_count = 0
for idx, row in df_baseline[df_baseline['Temporada'] == 'All'].iterrows():
    model_name = row['Modelo']
    
    # Buscar valores na tabela3
    tabela3_row = df_tabela3[df_tabela3['Modelo'] == model_name]
    
    if not tabela3_row.empty:
        # Atualizar valores
        for col in ['Accuracy', 'Precision', 'Recall', 'F1']:
            if col in tabela3_row.columns and col in df_baseline.columns:
                new_val = tabela3_row[col].values[0]
                old_val = df_baseline.at[idx, col]
                
                # Converter para float, ignorar '-' ou valores inválidos
                try:
                    new_val_float = float(new_val)
                    old_val_float = float(old_val)
                    df_baseline.at[idx, col] = new_val_float
                    print(f'   {model_name:20s} | {col:10s}: {old_val_float:.4f} → {new_val_float:.4f}')
                except (ValueError, TypeError):
                    # Manter valor original se não for numérico
                    pass
        
        # RPS, Brier, ROC AUC (se existirem em tabela3)
        for col in ['RPS', 'Brier', 'ROC AUC']:
            if col in tabela3_row.columns:
                val = tabela3_row[col].values[0]
                try:
                    val_float = float(val)
                    if col in df_baseline.columns:
                        df_baseline.at[idx, col] = val_float
                except (ValueError, TypeError):
                    pass
        
        updated_count += 1

print(f'\n✓ {updated_count} modelos atualizados na temporada "All"')

# Salvar backup
backup_path = baseline_path.replace('.csv', '_backup.csv')
df_backup = pd.read_csv(baseline_path)
df_backup.to_csv(backup_path, index=False)
print(f'\n💾 Backup salvo em: {backup_path}')

# Salvar baseline atualizado
df_baseline.to_csv(baseline_path, index=False)
print(f'✓ Atualizado: {baseline_path}')

print('\n' + '='*70)
print('✅ SINCRONIZAÇÃO CONCLUÍDA!')
print('='*70)
print('\nPróximos passos:')
print('  python scripts/generate_figures.py  # Regenerar figuras')
