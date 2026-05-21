"""
CLI para treinar um autoencoder a partir de dados de entrada.

Uso:
    python scripts/train_autoencoder.py --data-path data/ --output-dir models/autoencoder/

Argumentos opcionais:
    --data-path: Caminho dos dados CSV (default: data/)
    --output-dir: Diretório para salvar artefatos (default: models/autoencoder/)
    --latent-dim: Dimensão do espaço latente (default: 16)
    --epochs: Número máximo de epochs (default: 100)
    --batch-size: Tamanho do batch (default: 32)
    --learning-rate: Taxa de aprendizado (default: 0.001)
    --patience: Patience do EarlyStopping (default: 10)
    --test-size: Proporção de validação (default: 0.2)
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import load_multiple_seasons
from src.feature_engineering import calculate_team_stats
from src.autoencoder import train_autoencoder
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(
        description='Treina um autoencoder para redução de dimensionalidade'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/data_2005_2014',
        help='Caminho do diretório com dados (default: data/data_2005_2014)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/autoencoder',
        help='Diretório para salvar encoder/scaler (default: models/autoencoder)'
    )
    
    parser.add_argument(
        '--latent-dim',
        type=int,
        default=16,
        help='Dimensão do espaço latente (default: 16)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Número máximo de epochs (default: 100)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Tamanho do batch (default: 32)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Taxa de aprendizado (default: 0.001)'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Patience do EarlyStopping (default: 10)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proporção de validação (default: 0.2)'
    )
    
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help='Nível de verbosidade (0=silent, 1=progress, 2=detailed)'
    )
    
    args = parser.parse_args()
    
    # Validar paths
    if not os.path.exists(args.data_path):
        print(f"\n[ERRO] Diretório não encontrado: {args.data_path}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("TREINAMENTO DE AUTOENCODER PARA REDUÇÃO DE DIMENSIONALIDADE")
    print("="*70)
    
    # Carregar dados
    print(f"\n[1/4] Carregando dados de: {args.data_path}")
    try:
        df = load_multiple_seasons(args.data_path)
        print(f"  ✓ {len(df)} partidas carregadas")
    except Exception as e:
        print(f"\n[ERRO] Falha ao carregar dados: {e}")
        sys.exit(1)
    
    # Feature engineering
    print(f"\n[2/4] Aplicando feature engineering...")
    try:
        df_features = calculate_team_stats(df)
        X = df_features.drop(['Result', 'Season'], axis=1, errors='ignore').values
        y = df_features['Result'].values if 'Result' in df_features.columns else None
        print(f"  ✓ {X.shape[0]} amostras, {X.shape[1]} features")
    except Exception as e:
        print(f"\n[ERRO] Falha no feature engineering: {e}")
        sys.exit(1)
    
    # Dividir treino/validação
    print(f"\n[3/4] Dividindo dados (treino={1-args.test_size:.0%}, val={args.test_size:.0%})...")
    X_train, X_val = train_test_split(
        X,
        test_size=args.test_size,
        random_state=42
    )
    print(f"  ✓ Train: {X_train.shape[0]} amostras")
    print(f"  ✓ Val: {X_val.shape[0]} amostras")
    
    # Treinar autoencoder
    print(f"\n[4/4] Treinando autoencoder...")
    try:
        encoder, scaler, history, metadata = train_autoencoder(
            X_train=X_train,
            X_val=X_val,
            encoder_dir=args.output_dir,
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            verbose=args.verbose
        )
        
        print(f"\n" + "="*70)
        print("SUCESSO: Autoencoder treinado e salvo!")
        print("="*70)
        print(f"\nArquivos gerados em: {os.path.abspath(args.output_dir)}")
        print(f"  - encoder.keras (modelo)")
        print(f"  - scaler.pkl (normalização)")
        print(f"  - metadata.json (configuração)")
        print(f"  - training_history.pkl (histórico de treino)")
        
        print(f"\nPróximos passos:")
        print(f"  1. Usar encoder para gerar features latentes:")
        print(f"     python scripts/run_autoencoder_pipeline.py --encoder-dir {args.output_dir}")
        print(f"  2. Treinar modelos com features latentes")
        
    except Exception as e:
        print(f"\n[ERRO] Falha durante treinamento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
