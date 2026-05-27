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
        
        # Para multiclass, pegar a classe 0 (Home Win)
        if isinstance(shap_values, list):
            shap_home = shap_values[0]  # Classe 0: Home Win
            shap_draw = shap_values[1]  # Classe 1: Draw
            shap_away = shap_values[2]  # Classe 2: Away Win
            
            # Calcular importância média absoluta por classe
            mean_shap_home = np.abs(shap_home).mean(axis=0)
            mean_shap_draw = np.abs(shap_draw).mean(axis=0)
            mean_shap_away = np.abs(shap_away).mean(axis=0)
            mean_shap_overall = (mean_shap_home + mean_shap_draw + mean_shap_away) / 3
        else:
            mean_shap_overall = np.abs(shap_values).mean(axis=0)

        # Garantir que mean_shap_overall é ndarray 1-D (evita erros ao indexar com arrays)
        mean_shap_overall = np.asarray(mean_shap_overall).ravel()

        # Alinhar tamanho de mean_shap_overall com feature_names para evitar index errors
        n_feat_names = len(feature_names)
        if mean_shap_overall.size != n_feat_names:
            print(f"[AVISO] Tamanho de mean_shap_overall ({mean_shap_overall.size}) difere de feature_names ({n_feat_names}). Alinhando pelo menor tamanho.")
            min_len = min(mean_shap_overall.size, n_feat_names)
            mean_shap_overall = mean_shap_overall[:min_len]
            feature_names = feature_names[:min_len]
            # Recalcular nomes mapeados após possível truncamento
            mapped_feature_names = [map_feature_name(f) for f in feature_names]

        indices = np.argsort(mean_shap_overall)[::-1]
        # Garantir que indices seja um array de inteiros simples
        indices = np.asarray(indices, dtype=int).ravel()
        
        print("\n✓ SHAP values calculados!")
        print("\nTop 15 Features por Impacto SHAP (média absoluta):")
        print("-"*80)
        print(f"{'Rank':<6} {'Feature':<30} {'SHAP Impact':<15}")
        print("-"*80)
        
        for i, idx in enumerate(indices[:15], 1):
            idx_int = int(idx)
            display_name = map_feature_name(feature_names[idx_int])
            print(f"{i:<6} {display_name:<30} {mean_shap_overall[idx_int]:<15.4f}")
        
        # Salvar análise SHAP
        # Assegurar tipos inteiros nos índices antes de usar para indexação de listas/arrays
        indices_int = np.array(indices, dtype=int)
        shap_df = pd.DataFrame({
            'Feature': [feature_names[int(i)] for i in indices_int],
            'SHAP_Impact': mean_shap_overall[indices_int]
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
        
        # 2. Summary plot (beeswarm) - Versão final otimizada
        try:
            import matplotlib.cm as cm
            from matplotlib.colors import Normalize
            
            top_n = 10
            top_indices_shap = np.argsort(mean_shap_overall)[::-1][:top_n]
            
            fig, ax = plt.subplots(figsize=(11, 6), dpi=100)
            
            feature_labels = [mapped_feature_names[int(i)] for i in top_indices_shap]
            y_positions = np.arange(len(feature_labels))
            
            for idx_pos, feat_idx in enumerate(top_indices_shap):
                feat_idx = int(feat_idx)
                shap_vals = shap_values[0][:, feat_idx] if isinstance(shap_values, list) else shap_values[:, feat_idx]
                feature_vals = X_sample.iloc[:, feat_idx].values
                
                # Garantir que tudo está 1-D e do mesmo tamanho
                shap_vals = np.asarray(shap_vals).ravel()[:len(X_sample)]
                feature_vals = np.asarray(feature_vals).ravel()[:len(X_sample)]
                
                # Garantir que têm exatamente o mesmo tamanho
                min_size = min(len(shap_vals), len(feature_vals))
                shap_vals = shap_vals[:min_size]
                feature_vals = feature_vals[:min_size]
                
                norm = Normalize(vmin=feature_vals.min(), vmax=feature_vals.max())
                colors = cm.coolwarm(norm(feature_vals))
                
                jitter = np.random.normal(0, 0.05, size=min_size)
                y = np.full(min_size, idx_pos) + jitter
                
                ax.scatter(shap_vals, y, c=colors, alpha=0.6, s=25, edgecolors='none')
            
            ax.set_yticks(y_positions)
            ax.set_yticklabels(feature_labels, fontsize=10)
            ax.set_xlabel('SHAP Value', fontsize=11, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1.2, alpha=0.7)
            ax.grid(axis='x', alpha=0.25, linestyle=':')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_title('RandomForest - SHAP Feature Impact (Top 10)', 
                         fontsize=12, fontweight='bold', pad=15)
            
            plt.tight_layout()
            plt.savefig('models/figures/shap_summary_beeswarm.png', dpi=200, bbox_inches='tight')
            plt.close()
            print("  ✓ models/figures/shap_summary_beeswarm.png")
        except Exception as e:
            print(f"✗ Erro ao gerar beeswarm: {e}")
        
        # 3. Gráfico horizontal de top-15 SHAP values (alternativa melhor que waterfall)
        try:
            top_n = 15
            top_indices = indices[:top_n]
            top_names = [mapped_feature_names[int(i)] for i in top_indices]
            top_shap = mean_shap_overall[top_indices]
            
            plt.figure(figsize=(12, 8))
            y_pos = np.arange(len(top_names))
            plt.barh(y_pos, top_shap, color='steelblue')
            plt.yticks(y_pos, top_names, fontsize=11)
            plt.xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
            plt.title('RandomForest - Top 15 Features por Impacto SHAP\n(Classe: Home Win)', fontsize=13, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig('models/figures/shap_waterfall_example.png', dpi=200, bbox_inches='tight')
            plt.close()
            print("  ✓ models/figures/shap_waterfall_example.png (top-15 bar chart)")
        except Exception as e:
            print(f"✗ Erro ao gerar gráfico SHAP: {e}")
        
        print("\n" + "="*80)
        print("✓ Análise SHAP completa!")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Erro ao calcular SHAP: {e}")
        import traceback
        traceback.print_exc()

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
