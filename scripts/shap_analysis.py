#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script para análise SHAP (SHapley Additive exPlanations) dos modelos - DIA 8
ANÁLISE FOCADA NO RANDOMFOREST (melhor modelo: RPS 0.4145)
"""

import sys
import os
from pathlib import Path

# Forçar UTF-8 no Windows
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Adicionar o diretório raiz ao path
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
    print("AVISO: biblioteca 'shap' não instalada!")
    print("Para instalar: pip install shap")
    print("\nContinuando com análise de importância básica...\n")

print("="*80)
print("DIA 8: ANÁLISE DE IMPORTÂNCIA DAS FEATURES")
print("="*80)
print("Objetivo: Entender POR QUE RandomForest tem o melhor RPS (0.4145)")
print("="*80)

# Carregar dados da mesma forma que main.py (ajustado para novo split)
train_dir = "data/data_2011_2023"
test_dir = "data/data_2023_2025"

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

# Mapeamento de nomes de features para Português (exibição)
FEATURE_NAME_PT = {
    'h2h_games': 'Histórico de Confrontos',
    'B365D': 'Odds Empate (Bet365)',
    'B365H': 'Odds Casa (Bet365)',
    'B365A': 'Odds Visitante (Bet365)',
    'points_diff': 'Diferença de Pontos',
    'away_position': 'Posição Visitante',
    'position_diff': 'Diferença de Posição',
    'away_points': 'Pontos Visitante',
    'shots_diff': 'Diferença de Remates',
    'Season': 'Temporada',
    'h2h_home_wins': 'Vitórias (H2H Casa)',
    'home_position': 'Posição Casa',
    'overall_diff': 'Diferença Overall (FIFA)',
    'corners_diff': 'Diferença de Escanteios',
    'gd_diff': 'Diferença de Gols',
    'h2h_draws': 'Empates (H2H)',
    'midfield_diff': 'Diferença Meio-campo',
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
data = joblib.load('models/trained_models.pkl')
models = data['models']

print(f"\n[2] IMPORTÂNCIA BÁSICA DAS FEATURES")
print("="*80)

# Focar em RandomForest (melhor modelo)
rf_model = models['RandomForest']['model']

# Extrair modelo base (está dentro do CalibratedClassifierCV)
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
    print(f"{'Rank':<6} {'Feature':<30} {'Importância':<15} {'% Cumulativa':<15}")
    print("-"*80)
    
    cumulative = 0
    for i, idx in enumerate(indices[:15], 1):
        cumulative += importances[idx]
        display_name = map_feature_name(feature_names[idx])
        print(f"{i:<6} {display_name:<30} {importances[idx]:<15.4f} {cumulative*100:<15.1f}%")
    
    print("\n" + "="*80)
    print(f"✓ Top 15 features explicam {cumulative*100:.1f}% da importância total")
    print("="*80)
    
    # Salvar análise detalhada
    importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices],
        'Cumulative': np.cumsum(importances[indices])
    })
    importance_df.to_csv('models/feature_importance_randomforest.csv', index=False)
    print("\n✓ Análise completa salva em: models/feature_importance_randomforest.csv")
    
    # Criar gráfico de barras
    plt.figure(figsize=(12, 8))
    top15_idx = indices[:15]
    top15_names_pt = [map_feature_name(feature_names[i]) for i in top15_idx][::-1]
    plt.barh(range(15), importances[top15_idx][::-1])
    plt.yticks(range(15), top15_names_pt)
    plt.xlabel('Importância')
    plt.title('RandomForest - Top 15 Features Mais Importantes (RPS 0.4145)')
    plt.tight_layout()
    plt.savefig('models/figures/feature_importance_randomforest.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico salvo em: models/figures/feature_importance_randomforest.png")
else:
    print("✗ Modelo não possui feature_importances_")

# Análise rápida de outros modelos
print(f"\n[3] COMPARAÇÃO COM OUTROS MODELOS")
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
    
    # Tentar extrair importâncias
    if hasattr(base_model, 'feature_importances_'):
        importances = base_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("  Top 5 features:")
        for i, idx in enumerate(indices[:5], 1):
            print(f"    {i}. {feature_names[idx]:30} : {importances[idx]:.4f}")
    
    elif hasattr(base_model, 'coef_'):
        coef = np.abs(base_model.coef_).mean(axis=0)
        indices = np.argsort(coef)[::-1]
        
        print("  Top 5 features (|coef| médio):")
        for i, idx in enumerate(indices[:5], 1):
            print(f"    {i}. {feature_names[idx]:30} : {coef[idx]:.4f}")
    
    else:
        print("  (Modelo sem importâncias nativas)")

# Análise SHAP (se disponível)
if SHAP_AVAILABLE:
    print(f"\n[4] ANÁLISE SHAP - EXPLICABILIDADE AVANÇADA")
    print("="*80)
    print("Foco: RandomForest (melhor RPS 0.4145)")
    print("="*80)
    
    # Usar amostra para SHAP (TreeExplainer é rápido)
    sample_size = min(500, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)
    
    print(f"\n🔍 Calculando SHAP values para {sample_size} amostras de teste...")
    
    try:
        # TreeExplainer para RandomForest
        explainer = shap.TreeExplainer(base_rf)
        shap_values = explainer.shap_values(X_sample)
        
        print(f"Debug: type(shap_values)={type(shap_values)}, isinstance(list)={isinstance(shap_values, list)}")
        if isinstance(shap_values, list):
            print(f"Debug: len(shap_values list)={len(shap_values)}, types={[type(s) for s in shap_values]}")
            # Para multiclass, pegar a classe 0 (Home Win)
            shap_home = shap_values[0]  # Classe 0: Home Win
            shap_draw = shap_values[1]  # Classe 1: Draw
            shap_away = shap_values[2]  # Classe 2: Away Win
            
            print(f"Debug: shap_home shape={shap_home.shape}")
            
            # Calcular importância média absoluta por classe
            mean_shap_home = np.abs(shap_home).mean(axis=0)
            mean_shap_draw = np.abs(shap_draw).mean(axis=0)
            mean_shap_away = np.abs(shap_away).mean(axis=0)
            
            print(f"Debug: mean_shap_home shape={mean_shap_home.shape}, ndim={mean_shap_home.ndim}")
            
            # Average across the 3 classes
            mean_shap_overall = (mean_shap_home + mean_shap_draw + mean_shap_away) / 3
            print(f"Debug: mean_shap_overall shape={mean_shap_overall.shape}, ndim={mean_shap_overall.ndim}")
        else:
            # shap_values is (samples, features, classes) for multiclass
            print(f"Debug: shap_values shape={shap_values.shape if hasattr(shap_values, 'shape') else 'no shape'}")
            
            # Mean over samples and classes
            if shap_values.ndim == 3:
                # (samples, features, classes) -> average over samples, then average over classes
                mean_shap_overall = np.abs(shap_values).mean(axis=(0, 2))  # Average over samples and classes
                print(f"Debug: mean_shap_overall from ndim==3: shape={mean_shap_overall.shape}")
            else:
                mean_shap_overall = np.abs(shap_values).mean(axis=0)
                print(f"Debug: mean_shap_overall from other: shape={mean_shap_overall.shape}")
            
            # Ensure it's 1D
            if mean_shap_overall.ndim > 1:
                print(f"Debug: Flattening mean_shap_overall")
                mean_shap_overall = mean_shap_overall.flatten()
                
            print(f"Debug: mean_shap_overall after check: shape={mean_shap_overall.shape}, ndim={mean_shap_overall.ndim}")
        
        print(f"Debug BEFORE argsort: shape={mean_shap_overall.shape}, ndim={mean_shap_overall.ndim}, len={len(mean_shap_overall)}")
        
        indices = np.argsort(mean_shap_overall)[::-1]
        # Convert to Python list
        indices_py = indices.tolist()
        
        print("\n✓ SHAP values calculados!")
        print(f"\nTop 15 Features por Impacto SHAP (média absoluta):")
        print("-"*80)
        print(f"{'Rank':<6} {'Feature':<30} {'SHAP Impact':<15}")
        print("-"*80)
        
        for i, idx in enumerate(indices_py[:15], 1):
            try:
                # idx is a Python int from tolist()
                idx_int = int(idx)
                # Add bounds check
                if idx_int < len(feature_names) and idx_int < len(mean_shap_overall):
                    display_name = map_feature_name(feature_names[idx_int])
                    impact_val = float(mean_shap_overall[idx_int])
                    print(f"{i:<6} {display_name:<30} {impact_val:<15.4f}")
                else:
                    print(f"{i:<6} [INDEX OUT OF BOUNDS: {idx_int}] {' ':<15}")
            except Exception as loop_err:
                print(f"{i:<6} [ERROR: {loop_err}]")
        
        # Salvar análise SHAP
        # Converter indices para lista de ints Python (não numpy arrays)
        indices_list = [int(i) for i in indices_py]
        shap_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices_list],
            'SHAP_Impact': [float(mean_shap_overall[i]) for i in indices_list]
        })
        shap_df.to_csv('models/shap_importance_randomforest.csv', index=False)
        print("\n✓ Análise SHAP salva em: models/shap_importance_randomforest.csv")
        
        # Gráficos SHAP
        print("\n📊 Gerando visualizações SHAP...")
        
        # 1. Summary plot (bar) - Overall
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values[0] if isinstance(shap_values, list) else shap_values, 
                 X_sample, 
                 feature_names=mapped_feature_names,
                         show=False, 
                         plot_type='bar',
                         max_display=15)
        plt.title('RandomForest - SHAP Feature Importance (Classe: Home Win)')
        plt.tight_layout()
        plt.savefig('models/figures/shap_summary_bar.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ models/figures/shap_summary_bar.png")
        
        # 2. Summary plot (beeswarm) - Impacto e direção
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values[0] if isinstance(shap_values, list) else shap_values,
                 X_sample,
                 feature_names=mapped_feature_names,
                         show=False,
                         max_display=15)
        plt.title('RandomForest - SHAP Impact Distribution (Classe: Home Win)')
        plt.tight_layout()
        plt.savefig('models/figures/shap_summary_beeswarm.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ models/figures/shap_summary_beeswarm.png")
        
        # 3. Waterfall plot para uma predição específica
        sample_idx = 0  # Primeira amostra
        class_idx = 0  # Classe 0: Home Win
        plt.figure(figsize=(10, 6))
        
        # Extract correct SHAP values depending on format
        if isinstance(shap_values, list):
            shap_sample = shap_values[class_idx][sample_idx]
            base_value = explainer.expected_value[class_idx]
        else:
            # shap_values shape: (samples, features, classes)
            shap_sample = shap_values[sample_idx, :, class_idx]
            if isinstance(explainer.expected_value, np.ndarray):
                base_value = explainer.expected_value[class_idx]
            else:
                base_value = explainer.expected_value
        
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_sample,
                base_values=base_value,
                data=X_sample.iloc[sample_idx],
                feature_names=mapped_feature_names
            ),
            show=False
        )
        plt.title('RandomForest - Exemplo de Predição Individual (SHAP Waterfall)')
        plt.tight_layout()
        plt.savefig('models/figures/shap_waterfall_example.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ models/figures/shap_waterfall_example.png")
        
        print("\n" + "="*80)
        print("✓ Análise SHAP completa!")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Erro ao calcular SHAP: {e}")

else:
    print(f"\n[4] ANÁLISE SHAP - NÃO DISPONÍVEL")
    print("="*80)
    print("💡 Para análise avançada, instale SHAP:")
    print("   pip install shap")

print("\n" + "="*80)
print("ANÁLISE CONCLUÍDA - DIA 8")
print("="*80)

print("\n📊 Arquivos gerados:")
print("  - models/feature_importance_randomforest.csv")
if SHAP_AVAILABLE:
    print("  - models/shap_importance_randomforest.csv")
    print("  - models/figures/shap_summary_bar.png")
    print("  - models/figures/shap_summary_beeswarm.png")
    print("  - models/figures/shap_waterfall_example.png")
    print("\n✓ Use estes insights para entender POR QUE RandomForest funciona tão bem!")
else:
    print("  - models/figures/feature_importance_randomforest.png")
    print("\n💡 Instale SHAP para visualizações avançadas de explicabilidade")

sys.exit(0)
