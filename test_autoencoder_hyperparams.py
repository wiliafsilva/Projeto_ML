"""
Teste automatizado de hiperparâmetros do autoencoder
Testa diferentes latent_dim, learning_rate, batch_size
Treina modelos com cada uma e retorna a melhor configuração
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
import json
import itertools
import time
import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf
from src.preprocessing import load_multiple_seasons
from src.feature_engineering import calculate_team_stats
from src.autoencoder import build_autoencoder, train_autoencoder, encode
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def main():
    print("\n" + "="*80)
    print("TESTE AUTOMATIZADO: MELHOR CONFIGURAÇÃO DO AUTOENCODER")
    print("="*80)
    
    # ========================================================================
    # CARREGAR DADOS (UMA VEZ)
    # ========================================================================
    print("\n[1/5] Carregando dados (uma vez)...")
    df_train = load_multiple_seasons('data/data_2005_2014')
    df_test = load_multiple_seasons('data/data_2014_2016')
    
    df_train_feat = calculate_team_stats(df_train)
    df_test_feat = calculate_team_stats(df_test)
    
    X_train_orig = df_train_feat.drop(['Result', 'Season'], axis=1, errors='ignore').values
    X_test_orig = df_test_feat.drop(['Result', 'Season'], axis=1, errors='ignore').values
    y_train = df_train_feat['Result'].values
    y_test = df_test_feat['Result'].values
    
    print(f"  ✓ Train: {X_train_orig.shape}")
    print(f"  ✓ Test: {X_test_orig.shape}")
    
    # ========================================================================
    # DEFINIR GRID DE HIPERPARÂMETROS A TESTAR
    # ========================================================================
    print("\n[2/5] Definindo grid de hiperparâmetros...")
    
    latent_dims = [12, 16, 20, 24]
    learning_rates = [1e-3, 5e-3]
    batch_sizes = [32, 64]
    # Manter epochs e dropout fixo para teste rápido
    
    total_configs = len(latent_dims) * len(learning_rates) * len(batch_sizes)
    print(f"  Total de configurações a testar: {total_configs}")
    print(f"  Tempo estimado: ~60 minutos\n")
    
    results = []
    config_num = 0
    
    # ========================================================================
    # TESTAR CADA COMBINAÇÃO
    # ========================================================================
    print("[3/5] Testando combinações de hiperparâmetros...\n")
    
    start_time = time.time()
    
    for latent_dim, lr, batch_size in itertools.product(
        latent_dims, learning_rates, batch_sizes
    ):
        config_num += 1
        print(f"┌─ Config {config_num}/{total_configs}")
        print(f"│  latent_dim={latent_dim}, lr={lr}, batch_size={batch_size}")
        
        encoder_dir = f'models/autoencoder_test_latent{latent_dim}_lr{str(lr).replace(".", "_")}_bs{batch_size}'
        
        try:
            # Dividir treino/validação
            X_tr, X_val = train_test_split(
                X_train_orig,
                test_size=0.2,
                random_state=42
            )
            
            # Treinar autoencoder
            encoder, scaler, history, metadata = train_autoencoder(
                X_train=X_tr,
                X_val=X_val,
                encoder_dir=encoder_dir,
                latent_dim=latent_dim,
                epochs=80,  # Reduzido para teste rápido
                batch_size=batch_size,
                learning_rate=lr,
                patience=8,
                verbose=0  # Sem output detalhado
            )
            
            # Gerar features latentes
            X_train_latent = encode(X_train_orig, encoder, scaler)
            X_test_latent = encode(X_test_orig, encoder, scaler)
            
            # Treinar modelo de comparação (RandomForest rápido)
            start_rf = time.time()
            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X_train_latent, y_train)
            time_train = time.time() - start_rf
            
            y_pred = model.predict(X_test_latent)
            y_pred_proba = model.predict_proba(X_test_latent)
            
            # Calcular métricas
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            
            result = {
                'Config': config_num,
                'latent_dim': latent_dim,
                'learning_rate': lr,
                'batch_size': batch_size,
                'Final_Val_Loss': metadata['final_val_loss'],
                'Accuracy': acc,
                'F1_Score': f1,
                'AUC_ROC': auc,
                'RF_Train_Time': time_train,
                'Rank_Score': (acc + f1 + auc) / 3  # Score composto
            }
            
            results.append(result)
            
            print(f"│  ✓ Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
            print(f"│  ✓ Val Loss={metadata['final_val_loss']:.6f}")
            print(f"└─ OK\n")
            
        except Exception as e:
            print(f"│  ✗ ERRO: {str(e)[:50]}")
            print(f"└─ SKIP\n")
            continue
    
    # ========================================================================
    # ANALISAR RESULTADOS
    # ========================================================================
    print("\n" + "="*80)
    print("[4/5] Analisando resultados...")
    print("="*80)
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Tempo total: {elapsed/60:.1f} minutos")
    print(f"✓ Testes completados: {len(results)}/{total_configs}")
    
    if not results:
        print("\n❌ Nenhum resultado válido!")
        return
    
    df_results = pd.DataFrame(results)
    
    # Salvar resultados
    csv_path = 'models/autoencoder_hyperparameter_search.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"\n✓ Resultados salvos em: {csv_path}")
    
    # Encontrar melhor
    best_idx = df_results['Rank_Score'].idxmax()
    best = df_results.loc[best_idx]
    
    print("\n" + "="*80)
    print("MELHOR CONFIGURAÇÃO")
    print("="*80)
    print(f"\n🏆 Config #{best['Config']}")
    print(f"   latent_dim: {int(best['latent_dim'])}")
    print(f"   learning_rate: {best['learning_rate']}")
    print(f"   batch_size: {int(best['batch_size'])}")
    print(f"\n📊 Métricas:")
    print(f"   Accuracy: {best['Accuracy']:.4f}")
    print(f"   F1-Score: {best['F1_Score']:.4f}")
    print(f"   AUC-ROC: {best['AUC_ROC']:.4f}")
    print(f"   Rank Score (média): {best['Rank_Score']:.4f}")
    print(f"   Val Loss: {best['Final_Val_Loss']:.6f}")
    
    # Top 5
    print("\n" + "="*80)
    print("TOP 5 MELHORES CONFIGURAÇÕES")
    print("="*80)
    top5 = df_results.nlargest(5, 'Rank_Score')[
        ['Config', 'latent_dim', 'learning_rate', 'batch_size', 'Accuracy', 'F1_Score', 'AUC_ROC', 'Rank_Score']
    ]
    print("\n" + top5.to_string(index=False))
    
    # Análise por hiperparâmetro
    print("\n" + "="*80)
    print("ANÁLISE POR HIPERPARÂMETRO")
    print("="*80)
    
    print("\n📊 Accuracy médio por LATENT_DIM:")
    for ld in sorted(df_results['latent_dim'].unique()):
        avg_acc = df_results[df_results['latent_dim'] == ld]['Accuracy'].mean()
        print(f"   latent_dim={int(ld)}: {avg_acc:.4f}")
    
    print("\n📊 Accuracy médio por LEARNING_RATE:")
    for lr_val in sorted(df_results['learning_rate'].unique()):
        avg_acc = df_results[df_results['learning_rate'] == lr_val]['Accuracy'].mean()
        print(f"   lr={lr_val}: {avg_acc:.4f}")
    
    print("\n📊 Accuracy médio por BATCH_SIZE:")
    for bs in sorted(df_results['batch_size'].unique()):
        avg_acc = df_results[df_results['batch_size'] == bs]['Accuracy'].mean()
        print(f"   batch_size={int(bs)}: {avg_acc:.4f}")
    
    # Recomendação
    print("\n" + "="*80)
    print("RECOMENDAÇÃO FINAL")
    print("="*80)
    
    best_latent_dim = int(best['latent_dim'])
    best_lr = best['learning_rate']
    best_bs = int(best['batch_size'])
    
    print(f"\n✅ USE ESSA CONFIGURAÇÃO:")
    print(f"\n   python scripts/run_autoencoder_pipeline.py \\")
    print(f"     --latent-dim {best_latent_dim} \\")
    print(f"     --learning-rate {best_lr} \\")
    print(f"     --batch-size {best_bs} \\")
    print(f"     --epochs 150")
    
    print(f"\n📈 Melhoria esperada: +{(best['Accuracy']-0.60)*100:.2f}% em accuracy")
    print(f"   (vs baseline de ~60%)")
    
    # Salvar recomendação
    recommendation = {
        'best_config': {
            'latent_dim': best_latent_dim,
            'learning_rate': best_lr,
            'batch_size': best_bs,
        },
        'metrics': {
            'accuracy': float(best['Accuracy']),
            'f1_score': float(best['F1_Score']),
            'auc_roc': float(best['AUC_ROC']),
        },
        'top5': top5.to_dict('records'),
        'hyperparameter_analysis': {
            'latent_dim': {str(ld): float(df_results[df_results['latent_dim'] == ld]['Accuracy'].mean()) 
                          for ld in sorted(df_results['latent_dim'].unique())},
            'learning_rate': {str(lr_val): float(df_results[df_results['learning_rate'] == lr_val]['Accuracy'].mean()) 
                             for lr_val in sorted(df_results['learning_rate'].unique())},
            'batch_size': {str(bs): float(df_results[df_results['batch_size'] == bs]['Accuracy'].mean()) 
                          for bs in sorted(df_results['batch_size'].unique())},
        }
    }
    
    json_path = 'models/autoencoder_best_config.json'
    with open(json_path, 'w') as f:
        json.dump(recommendation, f, indent=2)
    print(f"\n✓ Recomendação salva em: {json_path}")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
