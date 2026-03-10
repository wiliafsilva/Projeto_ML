"""
GridSearch Focado para 43 Features (DIA 10-Fase1-B)

Objetivo: Re-otimizar hiperparâmetros após adicionar Form + μₖ features.
- DIA 5: GridSearch com 36 features
- AGORA: GridSearch com 43 features (13% mais features!)

Foco:
- RandomForest: max_features (mais features agora)
- XGBoost: colsample_bytree (mais features para amostrar)
- NaiveBayes: var_smoothing (27 features Class A agora)

Features críticas adicionadas:
- #6 form_diff (74.8% importância do top 1)
- #8 shots_diff (72.6%)
- #13 shotsontarget_diff (68.7%)
- #16 corners_diff (65.5%)
- #20 goals_avg_diff (56.3%)

Tempo estimado: 45-90 minutos
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
import warnings
warnings.filterwarnings('ignore')

from preprocessing import load_all_data
from feature_engineering import calculate_team_stats
from train_models import prepare_features_by_model
import joblib

def rps(y_true, y_prob):
    """Ranked Probability Score - métrica principal do artigo"""
    y_true = y_true.astype(int)
    y_true_onehot = np.eye(3)[y_true]
    y_true_cum = np.cumsum(y_true_onehot, axis=1)
    y_prob_cum = np.cumsum(y_prob, axis=1)
    return np.mean(np.sum((y_true_cum - y_prob_cum)**2, axis=1))

def rps_scorer(y_true, y_pred_proba):
    """Scorer para GridSearchCV (menor é melhor)"""
    return -rps(y_true, y_pred_proba)

def main():
    print("="*80)
    print("GRIDSEARCH FOCADO - 43 FEATURES (DIA 10-Fase1-B)")
    print("="*80)
    print("\nMotivação: Form + μₖ features são excelentes (#6, #8, #13, #16, #20)")
    print("           mas hiperparâmetros foram otimizados para 36 features.")
    print("\nObjetivo: Re-otimizar para 43 features (+19% features)")
    print("="*80)
    
    # ========================================================================
    # 1. CARREGAR E PREPARAR DADOS
    # ========================================================================
    print("\n[1/5] CARREGANDO DADOS...")
    df_all = load_all_data()
    df_train = df_all[df_all['Season'] <= 2014].copy().reset_index(drop=True)
    df_test = df_all[df_all['Season'] > 2014].copy().reset_index(drop=True)
    
    print("\n[2/5] CALCULANDO FEATURES...")
    df_train = calculate_team_stats(df_train)
    df_test = calculate_team_stats(df_test)
    
    # ========================================================================
    # 2. PREPARAR DADOS POR MODELO
    # ========================================================================
    print("\n[3/5] PREPARANDO DADOS POR MODELO...")
    
    models_to_tune = {
        'RandomForest': 'RandomForest',
        'XGBoost': 'XGBoost',
        'NaiveBayes': 'NaiveBayes'
    }
    
    results = {}
    
    for model_name, model_key in models_to_tune.items():
        print(f"\n{'='*80}")
        print(f"MODELO: {model_name}")
        print(f"{'='*80}")
        
        # Preparar features específicas do modelo
        df_train_model = prepare_features_by_model(df_train, model_key)
        df_test_model = prepare_features_by_model(df_test, model_key)
        
        X_train = df_train_model.drop(['Result', 'Season'], axis=1)
        y_train = df_train_model['Result']
        X_test = df_test_model.drop(['Result', 'Season'], axis=1)
        y_test = df_test_model['Result']
        
        # Verificar NaN (NÃO deve haver após fix de índices)
        nan_train = X_train.isna().sum().sum()
        nan_test = X_test.isna().sum().sum()
        if nan_train > 0 or nan_test > 0:
            print(f"\n⚠️ AVISO: {nan_train} NaN em treino, {nan_test} NaN em teste")
            print("   Isso NÃO deveria acontecer após fix de índices!")
        else:
            print(f"\n✓ Dados limpos: 0 NaN encontrados")
        
        print(f"Dados: {X_train.shape[0]} treino, {X_test.shape[0]} teste")
        print(f"Features: {X_train.shape[1]} (Class {'A' if model_key == 'NaiveBayes' else 'B'})")
        
        # ========================================================================
        # 3. DEFINIR GRIDS DE HIPERPARÂMETROS
        # ========================================================================
        
        if model_name == 'RandomForest':
            print("\n[4/5] CONFIGURANDO GRID - RandomForest")
            print("Hiperparâmetros focados:")
            print("  - max_features: Mais features agora (sqrt(43) ≈ 6.5)")
            print("  - max_depth: Profundidade para capturar interações")
            print("  - min_samples_split: Controle de overfitting")
            
            base_model = RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            
            param_grid = {
                'max_features': ['sqrt', 'log2', 8, 12, 16],  # sqrt(43)≈6.5, expandir
                'max_depth': [15, 20, 25, None],  # Mais features = mais profundidade
                'min_samples_split': [5, 10, 20]
            }
            
            print(f"\nCombinações: {len(param_grid['max_features']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split'])} = 60")
            
        elif model_name == 'XGBoost':
            print("\n[4/5] CONFIGURANDO GRID - XGBoost")
            print("Hiperparâmetros focados:")
            print("  - colsample_bytree: Amostragem de features (43 agora)")
            print("  - subsample: Amostragem de samples")
            print("  - max_depth: Profundidade das árvores")
            
            sample_weights = compute_sample_weight('balanced', y_train)
            
            base_model = XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
            
            param_grid = {
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9],  # Mais features = mais sampling
                'subsample': [0.7, 0.8, 0.9],
                'max_depth': [4, 5, 6]
            }
            
            print(f"\nCombinações: {len(param_grid['colsample_bytree']) * len(param_grid['subsample']) * len(param_grid['max_depth'])} = 36")
            
        elif model_name == 'NaiveBayes':
            print("\n[4/5] CONFIGURANDO GRID - NaiveBayes")
            print("Hiperparâmetros focados:")
            print("  - var_smoothing: Suavização de variância (27 features Class A)")
            
            sample_weights = compute_sample_weight('balanced', y_train)
            
            base_model = GaussianNB()
            
            param_grid = {
                'var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
            }
            
            print(f"\nCombinações: {len(param_grid['var_smoothing'])} = 6")
        
        # ========================================================================
        # 4. EXECUTAR GRIDSEARCH
        # ========================================================================
        print(f"\n[5/5] EXECUTANDO GRIDSEARCH ({model_name})...")
        print("Validação: 5-fold cross-validation")
        print("Métrica: RPS (Ranked Probability Score)")
        print("-"*80)
        
        rps_score = make_scorer(rps_scorer, needs_proba=True, greater_is_better=True)
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=rps_score,
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        
        # Fit
        if model_name == 'XGBoost' or model_name == 'NaiveBayes':
            grid_search.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            grid_search.fit(X_train, y_train)
        
        # Melhores parâmetros
        print("\n" + "="*80)
        print(f"RESULTADOS - {model_name}")
        print("="*80)
        print(f"\nMelhores hiperparâmetros:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        
        print(f"\nMelhor RPS (CV): {-grid_search.best_score_:.4f}")
        
        # Calibrar e avaliar no teste
        best_model = grid_search.best_estimator_
        
        if model_name != 'NaiveBayes':
            print("\nCalibrando modelo (CalibratedClassifierCV)...")
            best_model = CalibratedClassifierCV(best_model, cv=5, method='isotonic')
            if model_name == 'XGBoost':
                best_model.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                best_model.fit(X_train, y_train)
        
        # Avaliar no teste
        y_pred_proba = best_model.predict_proba(X_test)
        test_rps = rps(y_test.values, y_pred_proba)
        
        print(f"\n✓ RPS no teste: {test_rps:.4f}")
        
        # Salvar resultados
        results[model_name] = {
            'best_params': grid_search.best_params_,
            'cv_rps': -grid_search.best_score_,
            'test_rps': test_rps,
            'model': best_model
        }
        
        print(f"\n{'='*80}\n")
    
    # ========================================================================
    # 5. RESUMO FINAL
    # ========================================================================
    print("\n" + "="*80)
    print("RESUMO FINAL - GRIDSEARCH 43 FEATURES")
    print("="*80)
    
    print(f"\n{'Modelo':<20} {'RPS CV':<12} {'RPS Teste':<12} {'Melhoria'}")
    print("-"*80)
    
    # RPS baselines (DIA 10-Fase1 antes do GridSearch)
    baselines = {
        'RandomForest': 0.4127,
        'XGBoost': 0.4141,
        'NaiveBayes': 0.4169
    }
    
    for model_name, result in results.items():
        baseline = baselines[model_name]
        improvement = (baseline - result['test_rps']) / baseline * 100
        status = "✓" if improvement > 0 else "✗"
        
        print(f"{model_name:<20} {result['cv_rps']:<12.4f} {result['test_rps']:<12.4f} {improvement:+6.2f}% {status}")
    
    # Melhor modelo
    best_model_name = min(results.keys(), key=lambda x: results[x]['test_rps'])
    best_rps = results[best_model_name]['test_rps']
    
    print("\n" + "="*80)
    print(f"🏆 MELHOR MODELO: {best_model_name}")
    print(f"   RPS: {best_rps:.4f}")
    print(f"   Baseline DIA 9: 0.4127")
    print(f"   Melhoria total: {(0.4127 - best_rps) / 0.4127 * 100:+.2f}%")
    print("="*80)
    
    # Salvar resultados
    print("\nSalvando resultados...")
    output = {
        'results': results,
        'baselines': baselines,
        'best_model': best_model_name,
        'best_rps': best_rps,
        'num_features': 43
    }
    
    joblib.dump(output, 'models/gridsearch_43features_results.pkl')
    print("✓ Resultados salvos em: models/gridsearch_43features_results.pkl")
    
    # Salvar CSV com detalhes
    summary_data = []
    for model_name, result in results.items():
        summary_data.append({
            'Model': model_name,
            'RPS_CV': result['cv_rps'],
            'RPS_Test': result['test_rps'],
            'RPS_Baseline': baselines[model_name],
            'Improvement_%': (baselines[model_name] - result['test_rps']) / baselines[model_name] * 100,
            'Best_Params': str(result['best_params'])
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv('models/gridsearch_43features_summary.csv', index=False)
    print("✓ Resumo salvo em: models/gridsearch_43features_summary.csv")
    
    print("\n" + "="*80)
    print("GRIDSEARCH CONCLUÍDO!")
    print("="*80)

if __name__ == "__main__":
    main()
