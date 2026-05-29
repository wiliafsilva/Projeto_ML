#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script para analise SHAP (SHapley Additive exPlanations) dos modelos - DIA 8
ANALISE FOCADA NO RANDOMFOREST (melhor modelo: RPS 0.4145)
"""

import sys
import os
import argparse
from pathlib import Path

# Forcar UTF-8 no Windows
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Adicionar o diretorio raiz ao path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.preprocessing import load_multiple_seasons
from src.feature_engineering import calculate_team_stats

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("AVISO: biblioteca 'shap' nao instalada!")
    print("Para instalar: pip install shap")
    print("\nContinuando com analise de importancia basica...\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Analise SHAP")
    parser.add_argument("--model-path", default="models/trained_models.pkl")
    parser.add_argument("--output-dir", default="models")
    return parser.parse_args()


args = parse_args()
output_dir = args.output_dir
figures_dir = os.path.join(output_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)

print("="*80)
print("DIA 8: ANALISE DE IMPORTANCIA DAS FEATURES")
print("="*80)
print("Objetivo: Entender POR QUE RandomForest tem o melhor RPS (0.4145)")
print("="*80)

# Carregar dados da mesma forma que main.py
train_dir = "data/data_2005_2014"
test_dir = "data/data_2014_2016"

print("\n[1] CARREGANDO DADOS...")
df_train = load_multiple_seasons(train_dir)
df_test = load_multiple_seasons(test_dir)

print("Calculando features para treinamento...")
features_train = calculate_team_stats(df_train)

print("Calculando features para teste...")
features_test = calculate_team_stats(df_test)

# Preparar X, y
X_train = features_train.drop(['Result'], axis=1)
y_train = features_train['Result']
X_test = features_test.drop(['Result'], axis=1)
y_test = features_test['Result']

feature_names = X_test.columns.tolist()

# Mapeamento de nomes de features para Portugues (exibicao)
FEATURE_NAME_PT = {
    'h2h_games': 'Historico de Confrontos',
    'B365D': 'Odds Empate (Bet365)',
    'B365H': 'Odds Casa (Bet365)',
    'B365A': 'Odds Visitante (Bet365)',
    'points_diff': 'Diferenca de Pontos',
    'away_position': 'Posicao Visitante',
    'position_diff': 'Diferenca de Posicao',
    'away_points': 'Pontos Visitante',
    'shots_diff': 'Diferenca de Remates',
    'Season': 'Temporada',
    'h2h_home_wins': 'Vitorias (H2H Casa)',
    'home_position': 'Posicao Casa',
    'overall_diff': 'Diferenca Overall (FIFA)',
    'corners_diff': 'Diferenca de Escanteios',
    'gd_diff': 'Diferenca de Gols',
    'h2h_draws': 'Empates (H2H)',
    'midfield_diff': 'Diferenca Meio-campo',
    'home_form': 'Forma Casa',
    'away_form': 'Forma Visitante'
}


def map_feature_name(fname):
    return FEATURE_NAME_PT.get(fname, fname)


mapped_feature_names = [map_feature_name(f) for f in feature_names]

print(f"\n✓ Dataset carregado:")
print(f"  Treino: {len(X_train)} amostras, {len(feature_names)} features")
print(f"  Teste: {len(X_test)} amostras")

# Carregar modelos
results_metadata = joblib.load(args.model_path)
models = results_metadata['models']

print(f"\n[2] IMPORTANCIA BASICA DAS FEATURES")
print("="*80)

# Focar em RandomForest (melhor modelo)
rf_model = models['RandomForest']['model']

# Extrair modelo base (esta dentro do CalibratedClassifierCV)
if hasattr(rf_model, 'calibrated_classifiers_'):
    base_rf = rf_model.calibrated_classifiers_[0].estimator
else:
    base_rf = rf_model

print("\n🌲 RANDOMFOREST - FEATURE IMPORTANCE (Built-in)")
print("="*80)

if hasattr(base_rf, 'feature_importances_'):
    importances = base_rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nTop 15 Features Mais Importantes:")
    print("-"*80)
    print(f"{'Rank':<6} {'Feature':<30} {'Importancia':<15} {'% Cumulativa':<15}")
    print("-"*80)

    cumulative = 0
    for i, idx in enumerate(indices[:15], 1):
        cumulative += importances[idx]
        display_name = map_feature_name(feature_names[idx])
        print(f"{i:<6} {display_name:<30} {importances[idx]:<15.4f} {cumulative*100:<15.1f}%")

    print("\n" + "="*80)
    print(f"✓ Top 15 features explicam {cumulative*100:.1f}% da importancia total")
    print("="*80)

    # Salvar analise detalhada
    importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices],
        'Cumulative': np.cumsum(importances[indices])
    })
    importance_path = os.path.join(output_dir, 'feature_importance_randomforest.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"\n✓ Analise completa salva em: {importance_path}")

    # Criar grafico de barras
    plt.figure(figsize=(12, 8))
    top15_idx = indices[:15]
    top15_names_pt = [map_feature_name(feature_names[i]) for i in top15_idx][::-1]
    plt.barh(range(15), importances[top15_idx][::-1])
    plt.yticks(range(15), top15_names_pt)
    plt.xlabel('Importancia')
    plt.title('RandomForest - Top 15 Features Mais Importantes (RPS 0.4145)')
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, 'feature_importance_randomforest.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Grafico salvo em: {fig_path}")
else:
    print("✗ Modelo nao possui feature_importances_")

# Analise rapida de outros modelos
print(f"\n[3] COMPARACAO COM OUTROS MODELOS")
print("="*80)

for name in ['XGBoost', 'NaiveBayes', 'SVM']:
    if name not in models:
        continue

    model = models[name]['model']
    print(f"\n{name} (RPS {models[name]['rps']:.4f}):")

    # Extrair modelo base
    base_model = model
    if hasattr(model, 'calibrated_classifiers_'):
        base_model = model.calibrated_classifiers_[0].estimator

    # Tentar extrair importancias
    if hasattr(base_model, 'feature_importances_'):
        importances = base_model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("  Top 5 features:")
        for i, idx in enumerate(indices[:5], 1):
            print(f"    {i}. {feature_names[idx]:30} : {importances[idx]:.4f}")

    elif hasattr(base_model, 'coef_'):
        coef = np.abs(base_model.coef_).mean(axis=0)
        indices = np.argsort(coef)[::-1]

        print("  Top 5 features (|coef| medio):")
        for i, idx in enumerate(indices[:5], 1):
            print(f"    {i}. {feature_names[idx]:30} : {coef[idx]:.4f}")

    else:
        print("  (Modelo sem importancias nativas)")

# Analise SHAP (se disponivel)
if SHAP_AVAILABLE:
    print(f"\n[4] ANALISE SHAP - EXPLICABILIDADE AVANCADA")
    print("="*80)
    print("Foco: RandomForest (melhor RPS 0.4145)")
    print("="*80)

    # Usar amostra para SHAP (TreeExplainer e rapido)
    sample_size = min(500, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)

    print(f"\n🔍 Calculando SHAP values para {sample_size} amostras de teste...")

    try:
        # TreeExplainer para RandomForest
        explainer = shap.TreeExplainer(base_rf)
        shap_values = explainer.shap_values(X_sample)

        # Para multiclass, pegar a classe 0 (Home Win)
        if isinstance(shap_values, list):
            shap_home = shap_values[0]  # Classe 0: Home Win
            shap_draw = shap_values[1]  # Classe 1: Draw
            shap_away = shap_values[2]  # Classe 2: Away Win

            # Calcular importancia media absoluta por classe
            mean_shap_home = np.abs(shap_home).mean(axis=0)
            mean_shap_draw = np.abs(shap_draw).mean(axis=0)
            mean_shap_away = np.abs(shap_away).mean(axis=0)
            mean_shap_overall = (mean_shap_home + mean_shap_draw + mean_shap_away) / 3
        else:
            mean_shap_overall = np.abs(shap_values).mean(axis=0)

        indices = np.argsort(mean_shap_overall)[::-1]

        print("\n✓ SHAP values calculados!")
        print("\nTop 15 Features por Impacto SHAP (media absoluta):")
        print("-"*80)
        print(f"{'Rank':<6} {'Feature':<30} {'SHAP Impact':<15}")
        print("-"*80)

        for i, idx in enumerate(indices[:15], 1):
            display_name = map_feature_name(feature_names[idx])
            print(f"{i:<6} {display_name:<30} {mean_shap_overall[idx]:<15.4f}")

        # Salvar analise SHAP
        shap_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'SHAP_Impact': mean_shap_overall[indices]
        })
        shap_path = os.path.join(output_dir, 'shap_importance_randomforest.csv')
        shap_df.to_csv(shap_path, index=False)
        print(f"\n✓ Analise SHAP salva em: {shap_path}")

        # Graficos SHAP
        print("\n📊 Gerando visualizacoes SHAP...")

        # 1. Summary plot (bar) - Overall
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values[0] if isinstance(shap_values, list) else shap_values,
            X_sample,
            feature_names=mapped_feature_names,
            show=False,
            plot_type='bar',
            max_display=15
        )
        plt.title('RandomForest - SHAP Feature Importance (Classe: Home Win)')
        plt.tight_layout()
        bar_path = os.path.join(figures_dir, 'shap_summary_bar.png')
        plt.savefig(bar_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {bar_path}")

        # 2. Summary plot (beeswarm)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values[0] if isinstance(shap_values, list) else shap_values,
            X_sample,
            feature_names=mapped_feature_names,
            show=False,
            max_display=15
        )
        plt.title('RandomForest - SHAP Impact Distribution (Classe: Home Win)')
        plt.tight_layout()
        beeswarm_path = os.path.join(figures_dir, 'shap_summary_beeswarm.png')
        plt.savefig(beeswarm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {beeswarm_path}")

        # 3. Waterfall plot
        sample_idx = 0
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0][sample_idx] if isinstance(shap_values, list) else shap_values[sample_idx],
                base_values=explainer.expected_value[0] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
                data=X_sample.iloc[sample_idx],
                feature_names=mapped_feature_names
            ),
            show=False
        )
        plt.title('RandomForest - Exemplo de Predicao Individual (SHAP Waterfall)')
        plt.tight_layout()
        waterfall_path = os.path.join(figures_dir, 'shap_waterfall_example.png')
        plt.savefig(waterfall_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {waterfall_path}")

        print("\n" + "="*80)
        print("✓ Analise SHAP completa!")
        print("="*80)

    except Exception as e:
        print(f"\n✗ Erro ao calcular SHAP: {e}")
        import traceback
        traceback.print_exc()

else:
    print(f"\n[4] ANALISE SHAP - NAO DISPONIVEL")
    print("="*80)
    print("💡 Para analise avancada, instale SHAP:")
    print("   pip install shap")

print("\n" + "="*80)
print("ANALISE CONCLUIDA - DIA 8")
print("="*80)

print("\n📊 Arquivos gerados:")
print(f"  - {os.path.join(output_dir, 'feature_importance_randomforest.csv')}")
if SHAP_AVAILABLE:
    print(f"  - {os.path.join(output_dir, 'shap_importance_randomforest.csv')}")
    print(f"  - {os.path.join(figures_dir, 'shap_summary_bar.png')}")
    print(f"  - {os.path.join(figures_dir, 'shap_summary_beeswarm.png')}")
    print(f"  - {os.path.join(figures_dir, 'shap_waterfall_example.png')}")
    print("\n✓ Use estes insights para entender POR QUE RandomForest funciona tao bem!")
else:
    print(f"  - {os.path.join(figures_dir, 'feature_importance_randomforest.png')}")
    print("\n💡 Instale SHAP para visualizacoes avancadas de explicabilidade")
