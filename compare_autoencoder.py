"""
Script para comparar desempenho: Features Originais vs Autoencoder Latentes
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from src.preprocessing import load_multiple_seasons
from src.feature_engineering import calculate_team_stats
import time


def main():
    print("\n" + "="*80)
    print("AUTOENCODER VALIDATION: FEATURES ORIGINAIS vs LATENTES")
    print("="*80)
    
    # ========================================================================
    # CARREGAR DADOS
    # ========================================================================
    print("\n[Loading] Dados originais...")
    df_train = load_multiple_seasons('data/data_2005_2014')
    df_test = load_multiple_seasons('data/data_2014_2016')
    
    df_train_feat = calculate_team_stats(df_train)
    df_test_feat = calculate_team_stats(df_test)
    
    X_train_orig = df_train_feat.drop(['Result', 'Season'], axis=1, errors='ignore').values
    X_test_orig = df_test_feat.drop(['Result', 'Season'], axis=1, errors='ignore').values
    y_train = df_train_feat['Result'].values
    y_test = df_test_feat['Result'].values
    
    print(f"  ✓ Train: {X_train_orig.shape[0]} samples, {X_train_orig.shape[1]} features")
    print(f"  ✓ Test: {X_test_orig.shape[0]} samples")
    
    # Carregar features latentes
    print("\n[Loading] Features latentes do autoencoder...")
    with open('models/autoencoder/latent_features/X_train_latent.pkl', 'rb') as f:
        X_train_latent = pickle.load(f)
    with open('models/autoencoder/latent_features/X_test_latent.pkl', 'rb') as f:
        X_test_latent = pickle.load(f)
    
    print(f"  ✓ Train latent: {X_train_latent.shape}")
    print(f"  ✓ Test latent: {X_test_latent.shape}")
    
    # ========================================================================
    # TREINAR E COMPARAR MODELOS
    # ========================================================================
    models_config = [
        ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ('GradientBoosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('NaiveBayes', GaussianNB()),
    ]
    
    results = []
    
    for model_name, model_class in models_config:
        print(f"\n[Training] {model_name}...")
        
        # Treinar com features originais
        print(f"  • Treinando com FEATURES ORIGINAIS ({X_train_orig.shape[1]} dim)...")
        start = time.time()
        model_orig = model_class.__class__(**model_class.get_params())
        model_orig.fit(X_train_orig, y_train)
        time_orig = time.time() - start
        
        y_pred_orig = model_orig.predict(X_test_orig)
        y_pred_proba_orig = model_orig.predict_proba(X_test_orig)
        
        acc_orig = accuracy_score(y_test, y_pred_orig)
        f1_orig = f1_score(y_test, y_pred_orig, average='weighted')
        auc_orig = roc_auc_score(y_test, y_pred_proba_orig, multi_class='ovr')
        
        # Treinar com features latentes
        print(f"  • Treinando com FEATURES LATENTES ({X_train_latent.shape[1]} dim)...")
        start = time.time()
        model_latent = model_class.__class__(**model_class.get_params())
        model_latent.fit(X_train_latent, y_train)
        time_latent = time.time() - start
        
        y_pred_latent = model_latent.predict(X_test_latent)
        y_pred_proba_latent = model_latent.predict_proba(X_test_latent)
        
        acc_latent = accuracy_score(y_test, y_pred_latent)
        f1_latent = f1_score(y_test, y_pred_latent, average='weighted')
        auc_latent = roc_auc_score(y_test, y_pred_proba_latent, multi_class='ovr')
        
        # Calcular diferenças
        acc_diff = acc_latent - acc_orig
        f1_diff = f1_latent - f1_orig
        auc_diff = auc_latent - auc_orig
        # Proteção contra divisão por zero (treinos muito rápidos)
        time_speedup = (1 - time_latent / time_orig) * 100 if time_orig > 0 else 0
        
        results.append({
            'Model': model_name,
            'Acc_Orig': acc_orig,
            'Acc_Latent': acc_latent,
            'Acc_Diff': acc_diff,
            'F1_Orig': f1_orig,
            'F1_Latent': f1_latent,
            'F1_Diff': f1_diff,
            'AUC_Orig': auc_orig,
            'AUC_Latent': auc_latent,
            'AUC_Diff': auc_diff,
            'Time_Orig': time_orig,
            'Time_Latent': time_latent,
            'Speedup%': time_speedup
        })
        
        print(f"\n  📊 {model_name} Results:")
        print(f"     Original:  Acc={acc_orig:.4f}, F1={f1_orig:.4f}, AUC={auc_orig:.4f}, Time={time_orig:.2f}s")
        print(f"     Latent:    Acc={acc_latent:.4f}, F1={f1_latent:.4f}, AUC={auc_latent:.4f}, Time={time_latent:.2f}s")
        print(f"     Diff:      Acc={acc_diff:+.4f}, F1={f1_diff:+.4f}, AUC={auc_diff:+.4f}, Speedup={time_speedup:+.1f}%")
    
    # ========================================================================
    # SALVAR RESULTADOS
    # ========================================================================
    print("\n" + "="*80)
    print("RESUMO FINAL")
    print("="*80)
    
    df_results = pd.DataFrame(results)
    print("\n" + df_results.to_string(index=False))
    
    # Salvar como CSV
    df_results.to_csv('models/autoencoder_comparison.csv', index=False)
    print(f"\n✓ Resultados salvos em: models/autoencoder_comparison.csv")
    
    # ========================================================================
    # RECOMENDAÇÃO
    # ========================================================================
    print("\n" + "="*80)
    print("RECOMENDAÇÃO")
    print("="*80)
    
    avg_acc_diff = df_results['Acc_Diff'].mean()
    avg_speedup = df_results['Speedup%'].mean()
    
    print(f"\nMédia de Melhoria em Accuracy: {avg_acc_diff:+.4f}")
    print(f"Média de Speedup: {avg_speedup:+.1f}%")
    print(f"Redução de Features: 61% (41 → 16)")
    
    if avg_acc_diff > 0.01:
        print("\n✅ USE AUTOENCODER!")
        print("   → Melhor desempenho + features mais compactas")
    elif avg_acc_diff > -0.01:
        print("\n➡️  NEUTRAL (similar accuracy)")
        print("   → Use para ganho em velocidade/regularização")
    else:
        print("\n⚠️  CONSIDERE AJUSTES")
        print("   → Aumentar latent_dim ou usar arquitetura diferente")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
