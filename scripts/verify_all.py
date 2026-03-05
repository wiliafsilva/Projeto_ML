#!/usr/bin/env python
"""Script para verificar todos os dados do projeto"""

import sys
from pathlib import Path

# Adicionar o diretório raiz ao path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import joblib
import os
import pandas as pd
from src.preprocessing import load_multiple_seasons
from src.feature_engineering import calculate_team_stats

print("="*60)
print("VERIFICAÇÃO COMPLETA DO PROJETO")
print("="*60)

# 1. Verificar dataset de treinamento
print("\n[1] VERIFICANDO DATASET DE TREINAMENTO (2005-2014):")
df_train = load_multiple_seasons('data/data_2005_2014')
print(f"✓ Total de partidas: {len(df_train):,}")
print(f"✓ Temporadas: {df_train['Season'].nunique()} (de {df_train['Season'].min()} a {df_train['Season'].max()})")
print(f"✓ Times únicos: {pd.concat([df_train['HomeTeam'], df_train['AwayTeam']]).nunique()}")
print(f"✓ Distribuição de resultados:")
resultado_map = {0: 'Vitória Casa', 1: 'Empate', 2: 'Vitória Visitante'}
for key, count in df_train['Result'].value_counts().sort_index().items():
    print(f"  - {resultado_map[key]}: {count} ({count/len(df_train)*100:.1f}%)")

# 2. Verificar dataset de teste
print("\n[2] VERIFICANDO DATASET DE TESTE (2014-2016):")
df_test = load_multiple_seasons('data/data_2014_2016')
print(f"✓ Total de partidas: {len(df_test):,}")
print(f"✓ Temporadas: {df_test['Season'].nunique()} (de {df_test['Season'].min()} a {df_test['Season'].max()})")
print(f"✓ Times únicos: {pd.concat([df_test['HomeTeam'], df_test['AwayTeam']]).nunique()}")
print(f"✓ Distribuição de resultados:")
for key, count in df_test['Result'].value_counts().sort_index().items():
    print(f"  - {resultado_map[key]}: {count} ({count/len(df_test)*100:.1f}%)")

# 3. Verificar features de treinamento
print("\n[3] VERIFICANDO FEATURES DE TREINAMENTO:")
features_train = calculate_team_stats(df_train)
print(f"✓ Total de features geradas: {len(features_train):,}")
print(f"✓ Colunas: {list(features_train.columns)}")
print(f"\nEstatísticas das features:")
print(features_train[['gd_diff', 'streak_diff', 'weighted_diff']].describe())

print(f"\n✓ Percentual de valores zero:")
print(f"  - gd_diff == 0: {(features_train['gd_diff'] == 0).sum()} ({(features_train['gd_diff'] == 0).sum()/len(features_train)*100:.1f}%)")
print(f"  - streak_diff == 0: {(features_train['streak_diff'] == 0).sum()} ({(features_train['streak_diff'] == 0).sum()/len(features_train)*100:.1f}%)")
print(f"  - weighted_diff == 0: {(features_train['weighted_diff'] == 0).sum()} ({(features_train['weighted_diff'] == 0).sum()/len(features_train)*100:.1f}%)")

# 4. Verificar features de teste
print("\n[4] VERIFICANDO FEATURES DE TESTE:")
features_test = calculate_team_stats(df_test)
print(f"✓ Total de features geradas: {len(features_test):,}")
print(f"\nEstatísticas das features:")
print(features_test[['gd_diff', 'streak_diff', 'weighted_diff']].describe())

# 5. Verificar modelos
print("\n[5] VERIFICANDO MODELOS:")
if os.path.exists('models/trained_models.pkl'):
    print("✓ Arquivo de modelos encontrado")
    results_metadata = joblib.load('models/trained_models.pkl')
    models = results_metadata.get('models', results_metadata)  # Compatibilidade
    print(f"✓ Modelos carregados: {list(models.keys())}")
    
    if 'train_period' in results_metadata:
        print(f"✓ Período de treinamento: {results_metadata['train_period']}")
        print(f"✓ Período de teste: {results_metadata['test_period']}")
    
    print("\nMétricas dos modelos:")
    for name, info in models.items():
        print(f"\n  {name}:")
        print(f"    - Acurácia: {info['accuracy']:.4f} ({info['accuracy']*100:.2f}%)")
        print(f"    - F1 Score: {info['f1']:.4f}")
        print(f"    - RPS: {info['rps']:.4f}")
else:
    print("✗ Arquivo de modelos NÃO encontrado! Execute main.py para treinar.")

# 6. Verificar distribuição treino/teste
print("\n[6] VERIFICANDO DIVISÃO TREINO/TESTE:")
print(f"✓ Treino: {len(features_train)} partidas (2005-2014)")
print(f"✓ Teste: {len(features_test)} partidas (2014-2016)")
total_features = len(features_train) + len(features_test)
print(f"✓ Proporção treino/teste: {len(features_train)/total_features*100:.1f}% / {len(features_test)/total_features*100:.1f}%")

# 7. Verificar distribuição por temporada
print("\n[7] VERIFICANDO DISTRIBUIÇÃO POR TEMPORADA:")
df_combined = pd.concat([df_train, df_test])
jogos_por_temporada = df_combined.groupby('Season').size().sort_index()
print(f"✓ Média de jogos por temporada: {jogos_por_temporada.mean():.1f}")
print(f"✓ Mínimo: {jogos_por_temporada.min()} jogos (temporada {jogos_por_temporada.idxmin()})")
print(f"✓ Máximo: {jogos_por_temporada.max()} jogos (temporada {jogos_por_temporada.idxmax()})")
print(f"\nJogos por temporada:")
for season, count in jogos_por_temporada.items():
    periodo = "TREINO" if season < 2014 or season == 2014 else "TESTE"
    print(f"  {season}: {count} jogos [{periodo}]")

print("\n" + "="*60)
print("VERIFICAÇÃO CONCLUÍDA!")
print("="*60)
