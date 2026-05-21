"""
Teste RÁPIDO de hiperparâmetros (versão reduzida para feedback imediato)
Testa apenas as 4 configurações mais promissoras
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.preprocessing import load_multiple_seasons
from src.feature_engineering import calculate_team_stats
from src.autoencoder import train_autoencoder, encode


def main():
    print("\n" + "="*80)
    print("TESTE RÁPIDO: 4 CONFIGURAÇÕES MAIS PROMISSORAS")
    print("="*80 + "\n")
    
    # Carregar dados
    print("Carregando dados...")
    df_train = load_multiple_seasons('data/data_2005_2014')
    df_test = load_multiple_seasons('data/data_2014_2016')
    
    df_train_feat = calculate_team_stats(df_train)
    df_test_feat = calculate_team_stats(df_test)
    
    X_train_orig = df_train_feat.drop(['Result', 'Season'], axis=1, errors='ignore').values
    X_test_orig = df_test_feat.drop(['Result', 'Season'], axis=1, errors='ignore').values
    y_train = df_train_feat['Result'].values
    y_test = df_test_feat['Result'].values
    
    print(f"✓ Dados carregados\n")
    
    # 4 configs mais promissoras (baseado em literatura)
    configs = [
        {'latent_dim': 16, 'lr': 1e-3, 'batch_size': 32, 'name': 'Padrão'},
        {'latent_dim': 20, 'lr': 1e-3, 'batch_size': 32, 'name': 'Latent+'},
        {'latent_dim': 16, 'lr': 5e-3, 'batch_size': 32, 'name': 'LR Alto'},
        {'latent_dim': 20, 'lr': 5e-3, 'batch_size': 64, 'name': 'Otimizado'},
    ]
    
    results = []
    start_total = time.time()
    
    for i, config in enumerate(configs, 1):
        print(f"[{i}/4] Testando {config['name']}...")
        print(f"      latent_dim={config['latent_dim']}, lr={config['lr']}, bs={config['batch_size']}")
        
        encoder_dir = f"models/quick_test_{config['name'].replace(' ', '_')}"
        
        try:
            X_tr, X_val = train_test_split(X_train_orig, test_size=0.2, random_state=42)
            
            start = time.time()
            encoder, scaler, history, metadata = train_autoencoder(
                X_train=X_tr,
                X_val=X_val,
                encoder_dir=encoder_dir,
                latent_dim=config['latent_dim'],
                epochs=60,  # Rápido
                batch_size=config['batch_size'],
                learning_rate=config['lr'],
                patience=6,
                verbose=0
            )
            train_time = time.time() - start
            
            X_train_latent = encode(X_train_orig, encoder, scaler)
            X_test_latent = encode(X_test_orig, encoder, scaler)
            
            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X_train_latent, y_train)
            
            y_pred = model.predict(X_test_latent)
            y_pred_proba = model.predict_proba(X_test_latent)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            
            results.append({
                'Config': config['name'],
                'latent_dim': config['latent_dim'],
                'learning_rate': config['lr'],
                'batch_size': config['batch_size'],
                'Accuracy': acc,
                'F1_Score': f1,
                'AUC': auc,
                'Time': train_time,
                'Score': (acc + f1 + auc) / 3
            })
            
            print(f"      ✓ Accuracy={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
            print(f"      ✓ Tempo: {train_time:.1f}s\n")
            
        except Exception as e:
            print(f"      ✗ Erro: {str(e)[:50]}\n")
            continue
    
    # Resultados
    print("="*80)
    print("RESULTADOS")
    print("="*80)
    
    df = pd.DataFrame(results)
    print("\n" + df[['Config', 'Accuracy', 'F1_Score', 'AUC', 'Score']].to_string(index=False))
    
    best = df.loc[df['Score'].idxmax()]
    
    print("\n" + "="*80)
    print("🏆 MELHOR CONFIGURAÇÃO")
    print("="*80)
    print(f"\n{best['Config']}")
    print(f"  latent_dim: {best['latent_dim']}")
    print(f"  learning_rate: {best['learning_rate']}")
    print(f"  batch_size: {int(best['batch_size'])}")
    print(f"  Accuracy: {best['Accuracy']:.4f}")
    print(f"  F1-Score: {best['F1_Score']:.4f}")
    print(f"  AUC: {best['AUC']:.4f}")
    
    print(f"\n✅ Próximo passo:")
    print(f"\npython scripts/run_autoencoder_pipeline.py \\")
    print(f"  --latent-dim {int(best['latent_dim'])} \\")
    print(f"  --learning-rate {best['learning_rate']} \\")
    print(f"  --batch-size {int(best['batch_size'])} \\")
    print(f"  --epochs 150")
    
    # Salvar resultado
    df.to_csv('models/quick_test_results.csv', index=False)
    print(f"\n✓ Resultados salvos em: models/quick_test_results.csv")
    
    elapsed = time.time() - start_total
    print(f"⏱️  Total: {elapsed/60:.1f} minutos\n")


if __name__ == '__main__':
    main()
