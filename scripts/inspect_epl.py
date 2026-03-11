import os
import sys
import csv
from collections import Counter

# Ensure project root is on sys.path so `from src...` imports work when run as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

path = "data/epl.csv"

if os.path.exists(path):
    # Legacy CSV file path: keep original lightweight CSV reader behavior
    season_counts = Counter()
    zeros_home = 0
    zeros_away = 0
    both_zero = 0
    total = 0

    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            s = row.get('Season_End_Year', 'NA')
            season_counts[s] += 1
            try:
                hg = int(row.get('HomeGoals', 0))
                ag = int(row.get('AwayGoals', 0))
            except ValueError:
                continue
            if hg == 0:
                zeros_home += 1
            if ag == 0:
                zeros_away += 1
            if hg == 0 and ag == 0:
                both_zero += 1

    print(f"Total partidas: {total}")
    print(f"Temporadas encontradas: {len(season_counts)}")
    print("Contagem por temporada (temporada: partidas):")
    for s, c in sorted(season_counts.items()):
        print(f"{s}: {c}")

    print("\nZeros")
    print(f"HomeGoals==0: {zeros_home} ({zeros_home/total:.1%})")
    print(f"AwayGoals==0: {zeros_away} ({zeros_away/total:.1%})")
    print(f"Ambos 0: {both_zero} ({both_zero/total:.1%})")
else:
    # Fallback: use project loader to combine seasons
    from src.preprocessing import load_all_data
    df = load_all_data()
    total = len(df)
    # Season column is created by loader
    seasons = df['Season'].value_counts().sort_index()

    print(f"Total partidas: {total}")
    print(f"Temporadas encontradas: {len(seasons)}")
    print("Contagem por temporada (temporada: partidas):")
    for s, c in seasons.items():
        print(f"{s}: {c}")

    # Columns after preprocessing are FTHG/FTAG (or HomeGoals/AwayGoals)
    if 'FTHG' in df.columns and 'FTAG' in df.columns:
        zeros_home = (df['FTHG'] == 0).sum()
        zeros_away = (df['FTAG'] == 0).sum()
        both_zero = ((df['FTHG'] == 0) & (df['FTAG'] == 0)).sum()
    else:
        zeros_home = (df.get('HomeGoals', 0) == 0).sum() if 'HomeGoals' in df.columns else 0
        zeros_away = (df.get('AwayGoals', 0) == 0).sum() if 'AwayGoals' in df.columns else 0
        both_zero = 0

    print("\nZeros")
    print(f"HomeGoals==0: {zeros_home} ({zeros_home/total:.1%})")
    print(f"AwayGoals==0: {zeros_away} ({zeros_away/total:.1%})")
    print(f"Ambos 0: {both_zero} ({both_zero/total:.1%})")
