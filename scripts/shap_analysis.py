#!/usr/bin/env python
"""Script para análise SHAP (SHapley Additive exPlanations) dos modelos"""

import sys
from pathlib import Path

# Adicionar o diretório raiz ao path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.preprocessing import load_data
from src.feature_engineering import calculate_team_stats

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("AVISO: biblioteca 'shap' não instalada!")
    print("Para instalar: pip install shap")
    print("\nContinuando com análise de importância básica...\n")

print("="*60)
print("ANÁLISE DE IMPORTÂNCIA DAS FEATURES")
print("="*60)

# Carregar dados
df = load_data('data/epl.csv')
features = calculate_team_stats(df)

# Divisão train/test
train = features[features['Season'] <= 2018]
test = features[features['Season'] > 2018]

X_train = train.drop(['Result','Season'], axis=1)
y_train = train['Result']
X_test = test.drop(['Result','Season'], axis=1)
y_test = test['Result']

feature_names = X_test.columns.tolist()

# Carregar modelos
models = joblib.load('models/trained_models.pkl')

print(f"\n[1] IMPORTÂNCIA BÁSICA DAS FEATURES")
print("="*60)

for name, info in models.items():
    model = info['model']
    print(f"\n{name}:")
    
    # Se for modelo calibrado, pegar o modelo base
    base_model = model
    if hasattr(model, 'base_estimator'):
        base_model = model.base_estimator
    elif hasattr(model, 'estimator'):
        base_model = model.estimator
    elif hasattr(model, 'calibrated_classifiers_'):
        # CalibratedClassifierCV
        base_model = model.calibrated_classifiers_[0].estimator
    
    # Tentar extrair importâncias nativas
    if hasattr(base_model, 'feature_importances_'):
        importances = base_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("  Ranking de importância:")
        for i, idx in enumerate(indices, 1):
            print(f"    {i}. {feature_names[idx]:20} : {importances[idx]:.4f}")
    
    elif hasattr(base_model, 'coef_'):
        # SVM (pegar norma dos coeficientes)
        coef = np.abs(base_model.coef_).mean(axis=0)
        indices = np.argsort(coef)[::-1]
        
        print("  Ranking de importância (|coef| médio):")
        for i, idx in enumerate(indices, 1):
            print(f"    {i}. {feature_names[idx]:20} : {coef[idx]:.4f}")
    
    else:
        print("  Modelo não possui importâncias nativas.")

# Análise SHAP (se disponível)
if SHAP_AVAILABLE:
    print(f"\n[2] ANÁLISE SHAP (EXPLICABILIDADE AVANÇADA)")
    print("="*60)
    
    # Usar amostra menor para SHAP (mais rápido)
    sample_size = min(500, len(X_test))
    X_sample = X_test.sample(sample_size, random_state=42)
    
    for name, info in models.items():
        model = info['model']
        
        print(f"\n{name}:")
        
        try:
            # Criar explainer apropriado
            if name == "XGBoost" and hasattr(model, 'get_booster'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                
            elif name == "RandomForest":
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                
            else:  # SVM ou modelo calibrado
                # Usar KernelExplainer (mais lento mas funciona com qualquer modelo)
                background = shap.sample(X_train, 100)
                explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_values = explainer.shap_values(X_sample)
            
            # Para multiclass, pegar média absoluta entre classes
            if isinstance(shap_values, list):
                shap_array = np.abs(np.array(shap_values)).mean(axis=0)
            else:
                shap_array = np.abs(shap_values)
            
            # Calcular importâncias médias
            mean_shap = shap_array.mean(axis=0)
            indices = np.argsort(mean_shap)[::-1]
            
            print(f"  Ranking SHAP (importância média absoluta):")
            for i, idx in enumerate(indices, 1):
                print(f"    {i}. {feature_names[idx]:20} : {mean_shap[idx]:.4f}")
            
            # Salvar gráfico SHAP
            print(f"  Gerando gráfico SHAP summary...")
            plt.figure(figsize=(8, 5))
            
            if isinstance(shap_values, list):
                # Multiclass: plotar para classe 0 (Vitória Casa)
                shap.summary_plot(shap_values[0], X_sample, 
                                 feature_names=feature_names,
                                 show=False, plot_type='bar')
                plt.title(f'{name} - SHAP Importância (Classe: Vitória Casa)')
            else:
                shap.summary_plot(shap_values, X_sample,
                                 feature_names=feature_names, 
                                 show=False, plot_type='bar')
                plt.title(f'{name} - SHAP Importância')
            
            plt.tight_layout()
            plt.savefig(f'models/shap_{name.lower()}.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Gráfico salvo: models/shap_{name.lower()}.png")
            
        except Exception as e:
            print(f"  ✗ Erro ao calcular SHAP: {e}")

print("\n" + "="*60)
print("ANÁLISE CONCLUÍDA!")
print("="*60)

if SHAP_AVAILABLE:
    print("\n✓ Gráficos SHAP salvos na pasta models/")
    print("  Estes gráficos mostram quais features têm maior impacto nas previsões.")
else:
    print("\n💡 Instale 'shap' para análise avançada de explicabilidade:")
    print("   pip install shap")
