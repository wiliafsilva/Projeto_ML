"""
Radar Chart por Temporada
Gera um radar chart para cada temporada: 2023-2024, 2024-2025 e All

Entrada:
 - models/baseline_comparison.csv (Accuracy, F1, Precision, Recall por temporada)
 - models/trained_models.pkl (RPS por temporada em ['seasonal_results'])

Saída:
 - models/figures/radar_chart_2023-2024.png
 - models/figures/radar_chart_2024-2025.png
 - models/figures/radar_chart_All.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from math import pi


def ensure_output_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def generate_radar_for_season(season, df_base, trained_meta, output_path):
    df_season = df_base[df_base['Temporada'] == season].copy()
    df_ml = df_season[df_season['Tipo'] == 'ML'].copy()
    if df_ml.empty:
        print(f"Nenhum modelo ML encontrado para temporada {season}")
        return

    # Models order
    models = df_ml['Modelo'].tolist()

    # Metrics from CSV: Accuracy, Precision, Recall, F1
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']

    # Build RPS list from trained_meta['seasonal_results']
    seasonal_results = trained_meta.get('seasonal_results', {})
    season_key = season
    if season_key not in seasonal_results:
        # try alternative keys (e.g., 'All')
        print(f"Aviso: temporada {season} não encontrada em trained_models.pkl seasonal_results")

    rps_vals = []
    for m in models:
        r = None
        try:
            r = seasonal_results.get(season_key, {}).get(m, {}).get('rps')
        except Exception:
            r = None
        if r is None:
            # fallback to global model rps
            r = trained_meta['models'].get(m, {}).get('rps', 0.0)
        rps_vals.append(float(r))

    # Prepare data matrix: rows = metrics + RPS(inverted), cols = models
    data = []
    for metric in metrics:
        vals = df_ml[metric].astype(float).values.tolist()
        data.append(vals)

    # RPS inverted (1 - rps)
    data.append([1.0 - v for v in rps_vals])

    data = np.array(data)

    # Plot
    num_vars = data.shape[0]
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    colors = ['#FF6B6B', '#4ECDC4', '#FFD93D', '#6C5CE7']

    for idx, model in enumerate(models):
        vals = data[:, idx].tolist()
        vals += vals[:1]
        color = colors[idx % len(colors)]
        ax.plot(angles, vals, 'o-', linewidth=2, label=model, color=color)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'RPS\n(1-RPS)']
    ax.set_xticklabels(labels, size=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=9, color='gray')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title(f'Radar Chart - {season}', size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    ensure_output_dir(output_path)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Print values table
    print('\nVALORES USADOS PARA:', season)
    print(f"{'Modelo':<15} {'Accuracy':>10} {'Precision':>11} {'Recall':>10} {'F1-Score':>10} {'1-RPS':>10}")
    print('-'*80)
    for i, m in enumerate(models):
        print(f"{m:<15} {data[0,i]:>10.4f} {data[1,i]:>11.4f} {data[2,i]:>10.4f} {data[3,i]:>10.4f} {data[4,i]:>10.4f}")


def main():
    df = pd.read_csv('models/baseline_comparison.csv')
    trained = joblib.load('models/trained_models.pkl')

    seasons = ['2023-2024', '2024-2025', 'All']
    for s in seasons:
        out = f'models/figures/radar_chart_{s}.png'
        generate_radar_for_season(s, df, trained, out)
        print(f"Saved: {out}")


if __name__ == '__main__':
    main()
