"""
Pipeline completa: treina autoencoder + gera features latentes para múltiplos splits.

Uso:
    python scripts/run_autoencoder_pipeline.py --train-dir data/data_2005_2014 --test-dir data/data_2014_2016

Opções:
    --train-dir: Diretório com dados de treino (default: data/data_2005_2014)
    --test-dir: Diretório com dados de teste (default: data/data_2014_2016)
    --encoder-dir: Diretório para salvar encoder (default: models/autoencoder)
    --output-dir: Diretório para salvar features latentes (default: models/autoencoder/latent_features)
    --latent-dim: Dimensão do espaço latente (default: 16)
    --epochs: Máximo de epochs de treino (default: 100)
    --batch-size: Tamanho do batch (default: 32)
    --learning-rate: Taxa de aprendizado (default: 1e-3)
    --patience: Patience para EarlyStopping (default: 10)
    --val-size: Proporção de validação (default: 0.2)
    --verbose: Nível de verbosidade (default: 1)
"""

import argparse
import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import load_multiple_seasons
from src.feature_engineering import calculate_team_stats
from src.autoencoder import train_autoencoder, load_encoder_and_scaler, encode
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(
        description='Pipeline completo: treina autoencoder e gera features latentes'
    )
    
    parser.add_argument(
        '--train-dir',
        type=str,
        default='data/data_2005_2014',
        help='Diretório com dados de treino (default: data/data_2005_2014)'
    )
    
    parser.add_argument(
        '--test-dir',
        type=str,
        default='data/data_2014_2016',
        help='Diretório com dados de teste (default: data/data_2014_2016)'
    )
    
    parser.add_argument(
        '--encoder-dir',
        type=str,
        default='models/autoencoder',
        help='Diretório para salvar encoder (default: models/autoencoder)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/autoencoder/latent_features',
        help='Diretório para salvar features latentes (default: models/autoencoder/latent_features)'
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
        help='Máximo de epochs (default: 100)'
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
        help='Taxa de aprendizado (default: 1e-3)'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Patience para EarlyStopping (default: 10)'
    )
    
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.2,
        help='Proporção de validação (default: 0.2)'
    )
    
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help='Nível de verbosidade (default: 1)'
    )
    
    args = parser.parse_args()
    
    # Validar paths
    if not os.path.exists(args.train_dir):
        print(f"\n[ERRO] Diretório de treino não encontrado: {args.train_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.test_dir):
        print(f"\n[ERRO] Diretório de teste não encontrado: {args.test_dir}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("PIPELINE: AUTOENCODER + GERAÇÃO DE FEATURES LATENTES")
    print("="*70)
    
    # ETAPA 1: Carregar dados de treino
    print(f"\n[ETAPA 1/5] Carregando dados de TREINO...")
    try:
        df_train = load_multiple_seasons(args.train_dir)
        print(f"  ✓ {len(df_train)} partidas carregadas")
    except Exception as e:
        print(f"\n[ERRO] Falha ao carregar dados de treino: {e}")
        sys.exit(1)
    
    # ETAPA 2: Feature engineering (treino)
    print(f"\n[ETAPA 2/5] Aplicando feature engineering (TREINO)...")
    try:
        df_features_train = calculate_team_stats(df_train)
        X_train = df_features_train.drop(['Result', 'Season'], axis=1, errors='ignore').values
        y_train = df_features_train['Result'].values if 'Result' in df_features_train.columns else None
        print(f"  ✓ {X_train.shape[0]} amostras, {X_train.shape[1]} features")
    except Exception as e:
        print(f"\n[ERRO] Falha no feature engineering: {e}")
        sys.exit(1)
    
    # ETAPA 3: Dividir treino/validação e treinar autoencoder
    print(f"\n[ETAPA 3/5] Dividindo treino/validação e treinando AUTOENCODER...")
    try:
        X_tr, X_val = train_test_split(
            X_train,
            test_size=args.val_size,
            random_state=42
        )
        
        encoder, scaler, history, metadata = train_autoencoder(
            X_train=X_tr,
            X_val=X_val,
            encoder_dir=args.encoder_dir,
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            verbose=args.verbose
        )
        print(f"  ✓ Autoencoder treinado com sucesso!")
        
    except Exception as e:
        print(f"\n[ERRO] Falha durante treinamento do autoencoder: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ETAPA 4: Carregar dados de teste e gerar features latentes
    print(f"\n[ETAPA 4/5] Carregando dados de TESTE e gerando features latentes...")
    try:
        df_test = load_multiple_seasons(args.test_dir)
        print(f"  ✓ {len(df_test)} partidas de teste carregadas")
        
        df_features_test = calculate_team_stats(df_test)
        X_test = df_features_test.drop(['Result', 'Season'], axis=1, errors='ignore').values
        y_test = df_features_test['Result'].values if 'Result' in df_features_test.columns else None
        print(f"  ✓ Features de teste: {X_test.shape[0]} amostras, {X_test.shape[1]} features")
        
    except Exception as e:
        print(f"\n[ERRO] Falha ao carregar dados de teste: {e}")
        sys.exit(1)
    
    # ETAPA 5: Gerar e salvar features latentes
    print(f"\n[ETAPA 5/5] Gerando features latentes para todos os splits...")
    try:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Gerar latent features
        X_train_latent = encode(X_train, encoder, scaler)
        X_test_latent = encode(X_test, encoder, scaler)
        
        # Salvar como pickle
        train_latent_path = os.path.join(args.output_dir, 'X_train_latent.pkl')
        test_latent_path = os.path.join(args.output_dir, 'X_test_latent.pkl')
        
        with open(train_latent_path, 'wb') as f:
            pickle.dump(X_train_latent, f)
        print(f"  ✓ X_train_latent: {X_train_latent.shape} → {train_latent_path}")
        
        with open(test_latent_path, 'wb') as f:
            pickle.dump(X_test_latent, f)
        print(f"  ✓ X_test_latent: {X_test_latent.shape} → {test_latent_path}")
        
        # Salvar também como CSV (para análise manual)
        train_latent_csv = os.path.join(args.output_dir, 'X_train_latent.csv')
        test_latent_csv = os.path.join(args.output_dir, 'X_test_latent.csv')
        
        pd.DataFrame(X_train_latent).to_csv(train_latent_csv, index=False)
        print(f"  ✓ X_train_latent.csv: {train_latent_csv}")
        
        pd.DataFrame(X_test_latent).to_csv(test_latent_csv, index=False)
        print(f"  ✓ X_test_latent.csv: {test_latent_csv}")
        
        # Salvar targets também
        y_train_path = os.path.join(args.output_dir, 'y_train.pkl')
        y_test_path = os.path.join(args.output_dir, 'y_test.pkl')
        
        with open(y_train_path, 'wb') as f:
            pickle.dump(y_train, f)
        print(f"  ✓ y_train: {y_train.shape} → {y_train_path}")
        
        with open(y_test_path, 'wb') as f:
            pickle.dump(y_test, f)
        print(f"  ✓ y_test: {y_test.shape} → {y_test_path}")
        
        # Resumo
        print(f"\n" + "="*70)
        print("PIPELINE COMPLETA COM SUCESSO!")
        print("="*70)
        
        print(f"\nArquivos gerados:")
        print(f"  Encoder: {os.path.abspath(args.encoder_dir)}")
        print(f"  Features latentes: {os.path.abspath(args.output_dir)}")
        
        print(f"\nResumo de dimensionalidade:")
        print(f"  Original: {X_train.shape[1]} features")
        print(f"  Latent: {X_train_latent.shape[1]} features (redução {100*(1-X_train_latent.shape[1]/X_train.shape[1]):.1f}%)")
        print(f"  Treino: {X_train_latent.shape[0]} amostras")
        print(f"  Teste: {X_test_latent.shape[0]} amostras")
        
        print(f"\nPróximos passos:")
        print(f"  1. Usar features latentes para treinar modelos:")
        print(f"     python src/train_models.py --features-path {train_latent_path}")
        print(f"  2. Comparar com desempenho original para validar ganho")
        
    except Exception as e:
        print(f"\n[ERRO] Falha ao gerar features latentes: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
