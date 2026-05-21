"""
Visualizar resultados do teste de hiperparâmetros
Execute isto após test_autoencoder_hyperparams.py terminar
"""

import os
import json
import pandas as pd
from pathlib import Path

def show_results():
    csv_file = 'models/autoencoder_hyperparameter_search.csv'
    json_file = 'models/autoencoder_best_config.json'
    
    if not os.path.exists(csv_file):
        print("❌ Arquivo de resultados não encontrado!")
        print(f"   Execute primeiro: python test_autoencoder_hyperparams.py")
        return
    
    print("\n" + "="*80)
    print("RESULTADOS DO TESTE DE HIPERPARÂMETROS")
    print("="*80)
    
    # Carregar CSV
    df = pd.read_csv(csv_file)
    
    print(f"\n✓ Testes completados: {len(df)}")
    print("\nTodos os resultados:")
    print(df[['latent_dim', 'learning_rate', 'batch_size', 'Accuracy', 'F1_Score', 'AUC_ROC']].to_string(index=False))
    
    # Carregar recomendação
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            rec = json.load(f)
        
        print("\n" + "="*80)
        print("🏆 MELHOR CONFIGURAÇÃO")
        print("="*80)
        config = rec['best_config']
        metrics = rec['metrics']
        
        print(f"\nlatent_dim: {config['latent_dim']}")
        print(f"learning_rate: {config['learning_rate']}")
        print(f"batch_size: {config['batch_size']}")
        
        print(f"\nMétricas:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        
        print(f"\n✅ Próximo comando para treinar com melhor config:")
        print(f"\npython scripts/run_autoencoder_pipeline.py \\")
        print(f"  --latent-dim {config['latent_dim']} \\")
        print(f"  --learning-rate {config['learning_rate']} \\")
        print(f"  --batch-size {config['batch_size']} \\")
        print(f"  --epochs 150")

if __name__ == '__main__':
    show_results()
