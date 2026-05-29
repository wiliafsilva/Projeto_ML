"""
Radar Chart - Multi-Metric Model Comparison
===========================================

Agora o script gera radar charts separados por temporada quando executado:
- 2014-2015
- 2015-2016
- All

Ele usa `models/baseline_comparison.csv` para Accuracy, Precision, Recall e F1,
e `models/trained_models.pkl` para obter o RPS por temporada (em `seasonal_results`).

Saída:
 - models/figures/radar_chart_2014-2015.png
 - models/figures/radar_chart_2015-2016.png
 - models/figures/radar_chart_All.png
"""

import os
import argparse
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
    print('\n' + '='*60)
    print(f'Gerando radar chart para temporada: {season}')
    print('='*60)

    df_season = df_base[df_base['Temporada'] == season].copy()
    df_ml = df_season[df_season['Tipo'] == 'ML'].copy()
    if df_ml.empty:
        print(f"Nenhum modelo ML encontrado para temporada {season}")
        return

    models = df_ml['Modelo'].tolist()
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']

    seasonal_results = trained_meta.get('seasonal_results', {})

    rps_vals = []
    for m in models:
        r = seasonal_results.get(season, {}).get(m, {}).get('rps') if seasonal_results else None
        if r is None:
            r = trained_meta['models'].get(m, {}).get('rps', 0.0)
        rps_vals.append(float(r))

    data = []
    for metric in metrics:
        vals = df_ml[metric].astype(float).values.tolist()
        data.append(vals)

    data.append([1.0 - v for v in rps_vals])
    data = np.array(data)

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

    # Print table (inclui Brier e ROC_AUC se presentes no CSV)
    print('\nVALORES USADOS PARA:', season)
    header_cols = ['Modelo', 'Accuracy', 'Precision', 'Recall', 'F1-Score', '1-RPS']
    extra_cols = []
    if 'Brier' in df_ml.columns:
        extra_cols.append('Brier')
    if 'ROC_AUC' in df_ml.columns:
        extra_cols.append('ROC_AUC')

    cols_display = header_cols + extra_cols
    # build format string
    fmt = f"{{:<15}} {{:>10}} {{:>11}} {{:>10}} {{:>10}} {{:>10}}"
    if 'Brier' in extra_cols:
        fmt += ' {:>10}'
    if 'ROC_AUC' in extra_cols:
        fmt += ' {:>10}'

    print(fmt.format(*cols_display))
    print('-'*100)
    for i, m in enumerate(models):
        row_vals = [m, f"{data[0,i]:.4f}", f"{data[1,i]:.4f}", f"{data[2,i]:.4f}", f"{data[3,i]:.4f}", f"{data[4,i]:.4f}"]
        if 'Brier' in extra_cols:
            b = df_ml.loc[df_ml['Modelo'] == m, 'Brier'].values
            row_vals.append(f"{float(b[0]):.4f}" if len(b) and not pd.isna(b[0]) else '-')
        if 'ROC_AUC' in extra_cols:
            r = df_ml.loc[df_ml['Modelo'] == m, 'ROC_AUC'].values
            row_vals.append(f"{float(r[0]):.4f}" if len(r) and not pd.isna(r[0]) else '-')
        print(fmt.format(*row_vals))


def parse_args():
    parser = argparse.ArgumentParser(description="Radar chart por temporada")
    parser.add_argument("--model-path", default="models/trained_models.pkl")
    parser.add_argument("--output-dir", default="models")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    figures_dir = os.path.join(output_dir, 'figures')

    print('📂 Carregando dados...')
    df = pd.read_csv(os.path.join(output_dir, 'baseline_comparison.csv'))
    trained = joblib.load(args.model_path)

    seasons = ['2014-2015', '2015-2016', 'All']
    for s in seasons:
        os.makedirs(figures_dir, exist_ok=True)
        out = os.path.join(figures_dir, f'radar_chart_{s}.png')
        generate_radar_for_season(s, df, trained, out)
        print(f"📊 Salvo em: {out}")


if __name__ == '__main__':
    main()
 
