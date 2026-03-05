import csv
from collections import Counter

path = "data/epl.csv"
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
