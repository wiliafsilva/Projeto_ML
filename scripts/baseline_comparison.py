"""
Baseline Model Comparison
=========================

Compara modelos ML com modelo trivial (baseline) que sempre prevê a classe majoritária.

Este script é ESSENCIAL para artigos científicos - valida que os modelos realmente
aprendem algo além de sempre prever "Vitória Casa".

Autor: Projeto_ML
Data: 10 de março de 2026
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib

from src.preprocessing import load_all_data
from src.feature_engineering import calculate_team_stats
from src.train_models import prepare_features_by_model


def calculate_baseline_metrics():
    """
    Calcula métricas do baseline (sempre prever classe majoritária).
    
    Returns:
        dict: Métricas do baseline
    """
    print("="*80)
    print("BASELINE MODEL COMPARISON")
    print("="*80)
    print()
    
    # Carregar dados
    print("Carregando dados...")
    df_all = load_all_data()
    
    # Calcular features
    print("Calculando features...")
    df_features = calculate_team_stats(df_all)
    
    # Train/Test split
    df_train = df_all[df_all['Season'] <= 2014].copy().reset_index(drop=True)
    df_test = df_all[df_all['Season'] > 2014].copy().reset_index(drop=True)
    
    df_features_train = df_features[df_all['Season'] <= 2014].reset_index(drop=True)
    df_features_test = df_features[df_all['Season'] > 2014].reset_index(drop=True)
    
    # Remover colunas não-feature (Result e Season) para X
    X_train = df_features_train.drop(['Result', 'Season'], axis=1)
    X_test = df_features_test.drop(['Result', 'Season'], axis=1)
    y_train = df_train['Result']
    y_test = df_test['Result']
    
    print(f"   Treino: {len(X_train)} amostras")
    print(f"   Teste: {len(X_test)} amostras")
    print()
    
    # Distribuição de classes
    print("Distribuicao de Classes (Conjunto de Teste):")
    print("-" * 60)
    class_counts = y_test.value_counts().sort_index()
    class_names = {0: 'Home Win (H)', 1: 'Draw (D)', 2: 'Away Win (A)'}
    
    total = len(y_test)
    for cls, count in class_counts.items():
        pct = count / total * 100
        print(f"   {class_names[cls]}: {count:3d} jogos ({pct:5.1f}%)")
    
    print()
    print(f"   Classe majoritária: {class_names[class_counts.idxmax()]} "
          f"({class_counts.max() / total * 100:.1f}%)")
    print()
    
    # ============================================================================
    # BASELINE 1: Sempre prever classe majoritária (most_frequent)
    # ============================================================================
    print("BASELINE 1: Sempre Prever Classe Majoritaria")
    print("-" * 60)
    
    baseline_most_freq = DummyClassifier(strategy='most_frequent', random_state=42)
    baseline_most_freq.fit(X_train, y_train)
    y_pred_baseline1 = baseline_most_freq.predict(X_test)
    
    # Métricas
    acc_baseline1 = accuracy_score(y_test, y_pred_baseline1)
    f1_baseline1 = f1_score(y_test, y_pred_baseline1, average='macro', zero_division=0)
    prec_baseline1 = precision_score(y_test, y_pred_baseline1, average='macro', zero_division=0)
    rec_baseline1 = recall_score(y_test, y_pred_baseline1, average='macro', zero_division=0)
    
    print(f"   Predição: Sempre '{class_names[y_pred_baseline1[0]]}'")
    print(f"   Accuracy:  {acc_baseline1:.4f} ({acc_baseline1*100:.2f}%)")
    print(f"   F1 (macro): {f1_baseline1:.4f}")
    print(f"   Precision: {prec_baseline1:.4f}")
    print(f"   Recall:    {rec_baseline1:.4f}")
    print()
    
    # ============================================================================
    # BASELINE 2: Prever proporcionalmente à distribuição (stratified)
    # ============================================================================
    print("BASELINE 2: Predicao Estratificada (Proporcional)")
    print("-" * 60)
    print("   (Prevê aleatoriamente respeitando distribuição de classes)")
    print()
    
    baseline_stratified = DummyClassifier(strategy='stratified', random_state=42)
    baseline_stratified.fit(X_train, y_train)
    y_pred_baseline2 = baseline_stratified.predict(X_test)
    
    # Métricas
    acc_baseline2 = accuracy_score(y_test, y_pred_baseline2)
    f1_baseline2 = f1_score(y_test, y_pred_baseline2, average='macro', zero_division=0)
    prec_baseline2 = precision_score(y_test, y_pred_baseline2, average='macro', zero_division=0)
    rec_baseline2 = recall_score(y_test, y_pred_baseline2, average='macro', zero_division=0)
    
    print(f"   Accuracy:  {acc_baseline2:.4f} ({acc_baseline2*100:.2f}%)")
    print(f"   F1 (macro): {f1_baseline2:.4f}")
    print(f"   Precision: {prec_baseline2:.4f}")
    print(f"   Recall:    {rec_baseline2:.4f}")
    print()
    
    # ============================================================================
    # BASELINE 3: Sempre prever empate (worst case)
    # ============================================================================
    print("BASELINE 3: Sempre Prever Empate (Pior Caso)")
    print("-" * 60)
    
    baseline_uniform = DummyClassifier(strategy='constant', constant=1, random_state=42)
    baseline_uniform.fit(X_train, y_train)
    y_pred_baseline3 = baseline_uniform.predict(X_test)
    
    # Métricas
    acc_baseline3 = accuracy_score(y_test, y_pred_baseline3)
    f1_baseline3 = f1_score(y_test, y_pred_baseline3, average='macro', zero_division=0)
    prec_baseline3 = precision_score(y_test, y_pred_baseline3, average='macro', zero_division=0)
    rec_baseline3 = recall_score(y_test, y_pred_baseline3, average='macro', zero_division=0)
    
    print(f"   Predição: Sempre 'Draw (D)'")
    print(f"   Accuracy:  {acc_baseline3:.4f} ({acc_baseline3*100:.2f}%)")
    print(f"   F1 (macro): {f1_baseline3:.4f}")
    print(f"   Precision: {prec_baseline3:.4f}")
    print(f"   Recall:    {rec_baseline3:.4f}")
    print()
    
    # ============================================================================
    # COMPARAÇÃO COM MODELOS ML (POR TEMPORADA)
    # ============================================================================
    print("="*80)
    print("COMPARAÇÃO: BASELINE vs MODELOS ML (POR TEMPORADA)")
    print("="*80)
    print()
    
    # Carregar modelos treinados
    try:
        models_data = joblib.load('models/trained_models.pkl')
        
        models = models_data['models']
        
        # Temporadas de teste
        seasons_test = [
            ('2014-2015', df_test['Season'] == 2015),
            ('2015-2016', df_test['Season'] == 2016),
            ('All', df_test['Season'] > 2014)
        ]
        
        all_results = []
        
        for season_name, season_mask in seasons_test:
            print(f"\n{'='*80}")
            print(f"TEMPORADA: {season_name}")
            print(f"{'='*80}\n")
            
            # Filtrar dados da temporada
            y_season = df_test[season_mask]['Result']
            df_features_season = df_features_test[season_mask].reset_index(drop=True)
            
            print(f"Total de jogos: {len(y_season)}")
            print()
            
            # Recalcular baselines para esta temporada
            baseline_most_freq_season = DummyClassifier(strategy='most_frequent', random_state=42)
            X_season_dummy = df_features_season.drop(['Result', 'Season'], axis=1)
            baseline_most_freq_season.fit(X_train, y_train)
            y_pred_b1 = baseline_most_freq_season.predict(X_season_dummy)
            
            baseline_stratified_season = DummyClassifier(strategy='stratified', random_state=42)
            baseline_stratified_season.fit(X_train, y_train)
            y_pred_b2 = baseline_stratified_season.predict(X_season_dummy)
            
            baseline_draw_season = DummyClassifier(strategy='constant', constant=1, random_state=42)
            baseline_draw_season.fit(X_train, y_train)
            y_pred_b3 = baseline_draw_season.predict(X_season_dummy)
            
            # Métricas baselines
            acc_b1 = accuracy_score(y_season, y_pred_b1)
            f1_b1 = f1_score(y_season, y_pred_b1, average='macro', zero_division=0)
            prec_b1 = precision_score(y_season, y_pred_b1, average='macro', zero_division=0)
            rec_b1 = recall_score(y_season, y_pred_b1, average='macro', zero_division=0)
            
            acc_b2 = accuracy_score(y_season, y_pred_b2)
            f1_b2 = f1_score(y_season, y_pred_b2, average='macro', zero_division=0)
            prec_b2 = precision_score(y_season, y_pred_b2, average='macro', zero_division=0)
            rec_b2 = recall_score(y_season, y_pred_b2, average='macro', zero_division=0)
            
            acc_b3 = accuracy_score(y_season, y_pred_b3)
            f1_b3 = f1_score(y_season, y_pred_b3, average='macro', zero_division=0)
            prec_b3 = precision_score(y_season, y_pred_b3, average='macro', zero_division=0)
            rec_b3 = recall_score(y_season, y_pred_b3, average='macro', zero_division=0)
            
            print("TABELA COMPARATIVA")
            print("-" * 80)
            print(f"{'Modelo':<25} {'Accuracy':>10} {'F1 (macro)':>12} {'Precision':>11} {'Recall':>10}")
            print("-" * 80)
            
            # Baselines
            print(f"{'Baseline (Most Freq)':<25} {acc_b1:>10.4f} {f1_b1:>12.4f} "
                  f"{prec_b1:>11.4f} {rec_b1:>10.4f}")
            print(f"{'Baseline (Stratified)':<25} {acc_b2:>10.4f} {f1_b2:>12.4f} "
                  f"{prec_b2:>11.4f} {rec_b2:>10.4f}")
            print(f"{'Baseline (Always Draw)':<25} {acc_b3:>10.4f} {f1_b3:>12.4f} "
                  f"{prec_b3:>11.4f} {rec_b3:>10.4f}")
            print("-" * 80)
            
            # Modelos ML
            for name in ['RandomForest', 'XGBoost', 'NaiveBayes', 'SVM']:
                if name in models:
                    model = models[name]['model']
                    
                    # Preparar features específicas
                    df_season_model = prepare_features_by_model(df_features_season, name)
                    X_season_model = df_season_model.drop(['Result', 'Season'], axis=1)
                    
                    y_pred_ml = model.predict(X_season_model)
                    
                    acc_ml = accuracy_score(y_season, y_pred_ml)
                    f1_ml = f1_score(y_season, y_pred_ml, average='macro')
                    prec_ml = precision_score(y_season, y_pred_ml, average='macro')
                    rec_ml = recall_score(y_season, y_pred_ml, average='macro')
                    
                    print(f"{name:<25} {acc_ml:>10.4f} {f1_ml:>12.4f} {prec_ml:>11.4f} {rec_ml:>10.4f}")
                    
                    all_results.append({
                        'Temporada': season_name,
                        'Modelo': name,
                        'Accuracy': acc_ml,
                        'F1': f1_ml,
                        'Precision': prec_ml,
                        'Recall': rec_ml,
                        'Tipo': 'ML'
                    })
            
            # Adicionar baselines aos resultados
            all_results.extend([
                {'Temporada': season_name, 'Modelo': 'Baseline (Most Freq)', 'Accuracy': acc_b1, 'F1': f1_b1, 'Precision': prec_b1, 'Recall': rec_b1, 'Tipo': 'Baseline'},
                {'Temporada': season_name, 'Modelo': 'Baseline (Stratified)', 'Accuracy': acc_b2, 'F1': f1_b2, 'Precision': prec_b2, 'Recall': rec_b2, 'Tipo': 'Baseline'},
                {'Temporada': season_name, 'Modelo': 'Baseline (Always Draw)', 'Accuracy': acc_b3, 'F1': f1_b3, 'Precision': prec_b3, 'Recall': rec_b3, 'Tipo': 'Baseline'}
            ])
            
            print("-" * 80)
            print()
        
        # Salvar resultados detalhados
        ml_results = [r for r in all_results if r['Tipo'] == 'ML' and r['Temporada'] == 'All']
        
        # ========================================================================
        # GANHOS RELATIVOS (vs Baseline Most Frequent - All)
        # ========================================================================
        print("\n" + "="*80)
        print("📈 GANHOS RELATIVOS (vs Baseline Most Frequent)")
        print("="*80)
        
        baseline_all = [r for r in all_results if r['Modelo'] == 'Baseline (Most Freq)' and r['Temporada'] == 'All'][0]
        
        print(f"\n{'Modelo':<25} {'Δ Accuracy':>12} {'Δ F1':>12} {'× Better':>12}")
        print("-" * 80)
        
        for result in ml_results:
            delta_acc = (result['Accuracy'] - baseline_all['Accuracy']) / baseline_all['Accuracy'] * 100
            delta_f1 = result['F1'] - baseline_all['F1']
            improvement = result['Accuracy'] / baseline_all['Accuracy']
            
            print(f"{result['Modelo']:<25} {delta_acc:>+11.2f}% {delta_f1:>+11.4f} "
                  f"{improvement:>11.2f}×")
        
        print("-" * 80)
        print()
        
        # ========================================================================
        # INTERPRETAÇÃO
        # ========================================================================
        print("💡 INTERPRETAÇÃO DOS RESULTADOS (All)")
        print("-" * 80)
        
        best_ml = max(ml_results, key=lambda x: x['Accuracy'])
        
        print(f"✅ VALIDAÇÃO:")
        print(f"   - Baseline (Most Freq) Accuracy: {baseline_all['Accuracy']:.1%}")
        print(f"   - Melhor ML ({best_ml['Modelo']}) Accuracy: {best_ml['Accuracy']:.1%}")
        print(f"   - Ganho: {(best_ml['Accuracy'] - baseline_all['Accuracy']) / baseline_all['Accuracy'] * 100:+.1f}%")
        print()
        
        if best_ml['Accuracy'] > baseline_all['Accuracy']:
            print("   ✅ Modelos ML são SIGNIFICATIVAMENTE melhores que baseline!")
            print("   ✅ Isso prova que as features são informativas.")
        else:
            print("   ⚠️ PROBLEMA: ML não está melhor que baseline trivial!")
            print("   ⚠️ Revisar features, hiperparâmetros ou dados.")
        
        print()
        print(f"📌 F1-Score:")
        print(f"   - Baseline F1: {baseline_all['F1']:.4f}")
        print(f"   - ML médio F1: {np.mean([r['F1'] for r in ml_results]):.4f}")
        if baseline_all['F1'] > 0:
            print(f"   - ML é {np.mean([r['F1'] for r in ml_results])/baseline_all['F1']:.1f}× melhor em F1!")
        print()
        
        # ========================================================================
        # SALVAR RESULTADOS
        # ========================================================================
        results_df = pd.DataFrame(all_results)
        results_df.to_csv('models/baseline_comparison.csv', index=False)
        print("💾 Resultados salvos em: models/baseline_comparison.csv")
        print()
        
        print("="*80)
        print("✅ BASELINE COMPARISON CONCLUÍDO!")
        print("="*80)
        
        return {
            'baseline_results': [r for r in all_results if r['Tipo'] == 'Baseline'],
            'ml_results': [r for r in all_results if r['Tipo'] == 'ML'],
            'all_results': all_results
        }
        
    except FileNotFoundError:
        print("⚠️ Arquivo models/trained_models.pkl não encontrado!")
        print("   Execute 'python main.py' primeiro para treinar os modelos.")
        return None


if __name__ == '__main__':
    results = calculate_baseline_metrics()
