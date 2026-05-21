"""
Teste rápido: treina autoencoder em um subset pequeno de dados.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from src.preprocessing import load_multiple_seasons
from src.feature_engineering import calculate_team_stats
from src.autoencoder import build_autoencoder, train_autoencoder
from sklearn.model_selection import train_test_split

def test_autoencoder():
    print("\n" + "="*70)
    print("TESTE RÁPIDO: AUTOENCODER")
    print("="*70)
    
    # Carregar dados
    print("\n[1/4] Carregando dados...")
    df = load_multiple_seasons('data/data_2005_2014')
    df = df.head(500)  # Usar apenas primeiros 500 para teste rápido
    print(f"  [OK] {len(df)} partidas carregadas")
    
    # Feature engineering
    print("\n[2/4] Feature engineering...")
    df_features = calculate_team_stats(df)
    X = df_features.drop(['Result', 'Season'], axis=1, errors='ignore').values
    y = df_features['Result'].values if 'Result' in df_features.columns else None
    print(f"  [OK] {X.shape[0]} amostras, {X.shape[1]} features")
    
    # Dividir
    print("\n[3/4] Dividindo treino/validação...")
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
    print(f"  [OK] Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
    
    # Treinar
    print("\n[4/4] Treinando autoencoder (epochs=20, quick test)...")
    try:
        encoder, scaler, history, metadata = train_autoencoder(
            X_train=X_train,
            X_val=X_val,
            encoder_dir='models/autoencoder',
            latent_dim=16,
            epochs=20,  # Teste rápido
            batch_size=32,
            learning_rate=1e-3,
            patience=5,
            verbose=0  # Sem verbose para teste rápido
        )
        
        print(f"\n" + "="*70)
        print("[OK] TESTE PASSOU COM SUCESSO!")
        print("="*70)
        print(f"\nResumo:")
        print(f"  Input dim: {metadata['input_dim']}")
        print(f"  Latent dim: {metadata['latent_dim']}")
        print(f"  Epochs treinados: {metadata['epochs_trained']}")
        print(f"  MSE treino final: {metadata['final_train_loss']:.6f}")
        print(f"  MSE val final: {metadata['final_val_loss']:.6f}")
        print(f"\nArquivos salvos em: models/autoencoder/")
        print(f"  - encoder.keras")
        print(f"  - scaler.pkl")
        print(f"  - metadata.json")
        print(f"  - training_history.pkl")
        
        return True
        
    except Exception as e:
        print(f"\n✗ TESTE FALHOU: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_autoencoder()
    sys.exit(0 if success else 1)
