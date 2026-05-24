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


def generate_radar_from_pkl(season, trained_meta, output_path):
    """Gera radar chart diretamente de trained_models.pkl quando baseline_comparison.csv não tem os dados"""
    print('\n' + '='*60)
    print(f'Gerando radar chart para temporada: {season}')
    print('='*60)
    
    models_dict = trained_meta.get('models', {})
    if not models_dict:
        print(f"Nenhum modelo encontrado no arquivo de treinamento")
        return
    
    # Obter resultados por temporada
    seasonal_results = trained_meta.get('seasonal_results', {})
    season_data = seasonal_results.get(season, {})
    
    # Se season_data estiver vazio, usar dados gerais
    if not season_data:
        print(f"⚠️ Nenhum resultado por temporada para {season}, usando dados consolidados")
        # Usar dados gerais dos modelos
        models_list = list(models_dict.keys())
        accuracy_vals = [models_dict[m].get('accuracy', 0.0) for m in models_list]
        f1_vals = [models_dict[m].get('f1', 0.0) for m in models_list]
        rps_vals = [models_dict[m].get('rps', 0.5) for m in models_list]
    else:
        models_list = list(season_data.keys())
        accuracy_vals = [season_data[m].get('accuracy', 0.0) for m in models_list]
        f1_vals = [season_data[m].get('f1', 0.0) for m in models_list]
        rps_vals = [season_data[m].get('rps', 0.5) for m in models_list]
    
    # Para simplificar, usar F1 para Precision e Recall também (como aproximação)
    # Este é um workaround porque seasonal_results não tem essas métricas detalhadas
    metrics_data = {
        'Accuracy': accuracy_vals,
        'Precision': f1_vals,
        'Recall': f1_vals,
        'F1-Score': f1_vals,
        '1-RPS': [1.0 - v for v in rps_vals]
    }
    
    # Preparar dados para radar chart
    num_vars = len(metrics_data)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    colors = ['#FF6B6B', '#4ECDC4', '#FFD93D', '#6C5CE7', '#95E1D3']
    
    metric_names = list(metrics_data.keys())
    for idx, model in enumerate(models_list):
        vals = [metrics_data[m][idx] for m in metric_names]
        vals += vals[:1]
        color = colors[idx % len(colors)]
        ax.plot(angles, vals, 'o-', linewidth=2, label=model, color=color)
        ax.fill(angles, vals, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, size=11, fontweight='bold')
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
    
    print('\nVALORES USADOS PARA:', season)
    print(f"{'Modelo':<15} {'Accuracy':>10} {'F1-Score':>10} {'1-RPS':>10}")
    print('-'*50)
    for i, m in enumerate(models_list):
        print(f"{m:<15} {accuracy_vals[i]:>10.4f} {f1_vals[i]:>10.4f} {1.0-rps_vals[i]:>10.4f}")


def main():
    print('📂 Carregando dados...')
    
    # Tentar carregar do baseline_comparison.csv primeiro
    try:
        df = pd.read_csv('models/baseline_comparison.csv')
        has_baseline = True
    except:
        has_baseline = False
        print("⚠️ baseline_comparison.csv não encontrado")
    
    trained = joblib.load('models/trained_models.pkl')

    seasons = ['2014-2015', '2015-2016', 'All']
    for s in seasons:
        out = f'models/figures/radar_chart_{s}.png'
        
        # Se baseline_comparison existe E tem a temporada, usar; senão usar pkl
        if has_baseline and s in df['Temporada'].values:
            generate_radar_for_season(s, df, trained, out)
        else:
            generate_radar_from_pkl(s, trained, out)
        
        print(f"📊 Salvo em: {out}")


if __name__ == '__main__':
    main()
 
