import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_data
from src.feature_engineering import calculate_team_stats
import pandas as pd

# Carregar dados
df = load_data("data/epl.csv")
print("=" * 80)
print("VERIFICAÇÃO DOS DADOS CARREGADOS")
print("=" * 80)
print(f"\nTotal de linhas: {len(df)}")
print(f"\nColunas disponíveis: {list(df.columns)}")
print("\nPrimeiras 10 linhas após preprocessing:")
print(df[['Date', 'Season', 'Season_End_Year', 'HomeTeam', 'FTHG', 'FTAG', 'AwayTeam', 'Result']].head(10))

# Calcular features
features = calculate_team_stats(df)
print("\n" + "=" * 80)
print("FEATURES CALCULADAS")
print("=" * 80)
print("\nPrimeiras 10 linhas de features:")
print(features.head(10))

# Verificar estatísticas das features
print("\n" + "=" * 80)
print("ESTATÍSTICAS DAS FEATURES")
print("=" * 80)
print(features[['gd_diff', 'streak_diff', 'weighted_diff']].describe())

# Contar quantos valores são zero
print("\n" + "=" * 80)
print("CONTAGEM DE ZEROS")
print("=" * 80)
print(f"gd_diff == 0: {(features['gd_diff'] == 0).sum()} de {len(features)} ({(features['gd_diff'] == 0).sum()/len(features)*100:.1f}%)")
print(f"streak_diff == 0: {(features['streak_diff'] == 0).sum()} de {len(features)} ({(features['streak_diff'] == 0).sum()/len(features)*100:.1f}%)")
print(f"weighted_diff == 0: {(features['weighted_diff'] == 0).sum()} de {len(features)} ({(features['weighted_diff'] == 0).sum()/len(features)*100:.1f}%)")

# Verificar se há features não-zero
print("\n" + "=" * 80)
print("AMOSTRA DE FEATURES NÃO-ZERO (SE HOUVER)")
print("=" * 80)
non_zero = features[(features['gd_diff'] != 0) | (features['streak_diff'] != 0) | (features['weighted_diff'] != 0)]
if len(non_zero) > 0:
    print(f"\nEncontradas {len(non_zero)} linhas com features não-zero")
    print("\nPrimeiras 10 linhas não-zero:")
    print(non_zero.head(10))
else:
    print("\n⚠️ TODAS AS FEATURES ESTÃO ZERADAS! ESTE É O PROBLEMA!")

# Verificar problema de Season
print("\n" + "=" * 80)
print("VERIFICAÇÃO DE TEMPORADAS")
print("=" * 80)
print("\nContagem de jogos por Season (ano da data):")
print(df['Season'].value_counts().sort_index().head(10))
print("\nContagem por Season_End_Year (ano final da temporada):")
print(df['Season_End_Year'].value_counts().sort_index().head(10))
