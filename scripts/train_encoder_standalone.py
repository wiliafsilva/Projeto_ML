"""
scripts/train_encoder_standalone.py
====================================
Script standalone para treinar o autoencoder (redução: ~43 dims → 16 dims latentes).

Pipeline:
  1. Carrega dados 2005-2014 (treino)
  2. Calcula 43 features originais (Class A + Class B)
  3. Treina autoencoder com arquitetura: 43 → 32 → 16 (bottleneck)
  4. Salva encoder (apenas a parte de compressão, sem decoder)
  5. Salva StandardScaler para normalizar dados novos

As 16 dimensões latentes são depois combinadas com as 43 originais
para criar datasets com 59 features totais.

Uso:
    python scripts/train_encoder_standalone.py

Saídas:
    models/encoder_16dims.keras   — Encoder treinado (29.8 KB)
    models/scaler_43features.pkl  — StandardScaler fitado (1.2 KB)

Nota: Este script é executado automaticamente como parte de main.py
"""

import sys
import os

# Garante que o diretório raiz do projeto está no path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.preprocessing import load_multiple_seasons
from src.feature_engineering import calculate_team_stats
from src.encoder import train_encoder, save_encoder_models, extract_latent_features

# ── Configurações ──────────────────────────────────────────────────────────────
TRAIN_DIR   = "data/data_2005_2014"
MODELS_DIR  = "models"
LATENT_DIM  = 16
EPOCHS      = 100
BATCH_SIZE  = 32
RANDOM_SEED = 42

# Features Class B (usadas como entrada do encoder)
CLASS_B_FEATURES = [
    'gd_diff', 'streak_diff', 'weighted_diff',
    'form_diff',
    'corners_diff', 'shotsontarget_diff', 'shots_diff', 'goals_avg_diff',
    'overall_diff', 'attack_diff', 'midfield_diff', 'defense_diff',
    'position_diff', 'points_diff',
    'h2h_confidence', 'away_advantage', 'season_trend',
    'position_form_home', 'position_form_away', 'strength_balance',
    'B365H', 'B365D', 'B365A',
    'prob_home', 'prob_draw', 'prob_away',
    'prob_home_norm', 'prob_draw_norm', 'prob_away_norm',
]


def get_feature_matrix(df, feature_list):
    """Filtra apenas as colunas disponíveis no DataFrame."""
    available = [f for f in feature_list if f in df.columns]
    missing   = [f for f in feature_list if f not in df.columns]
    if missing:
        print(f"[Setup] ⚠ Features ausentes (não usadas): {missing}")
        print(f"[Setup] ! Features ausentes (nao usadas): {missing}")
    print(f"[Setup] Features disponiveis para encoder: {len(available)}")
    return df[available].fillna(0).values, available


def main():
    print("=" * 70)
    print("TREINAMENTO DO ENCODER STANDALONE (43 -> 16 DIMS)")
    print("=" * 70)

    # -- 1. Carregar dados de treino --------------------------------------------
    print(f"\n[1/4] Carregando dados de treino: {TRAIN_DIR}")
    df_train_raw = load_multiple_seasons(TRAIN_DIR)

    # -- 2. Engenharia de features ----------------------------------------------
    print("\n[2/4] Calculando features...")
    df_train = calculate_team_stats(df_train_raw, add_latent=False)
    print(f"      Shape do DataFrame de features: {df_train.shape}")

    # -- 3. Montar matriz X -----------------------------------------------------
    X_train, used_features = get_feature_matrix(df_train, CLASS_B_FEATURES)
    input_dim = X_train.shape[1]
    print(f"[3/4] Matriz X_train montada: {X_train.shape}")
    print(f"      Features usadas ({input_dim}): {used_features}")

    # -- 4. Treinar encoder -----------------------------------------------------
    print(f"\n[4/4] Treinando encoder ({input_dim} -> {LATENT_DIM} dims)...")
    encoder, scaler = train_encoder(
        X_train,
        input_dim=input_dim,
        latent_dim=LATENT_DIM,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    # -- Salvar modelos ---------------------------------------------------------
    print("\n[Salvando] Encoder e Scaler...")
    save_encoder_models(encoder, scaler, path=MODELS_DIR)

    # -- Verificacao rapida -----------------------------------------------------
    print("\n[Verificacao] Extraindo features latentes de amostra...")
    X_latent = extract_latent_features(X_train[:5], encoder, scaler)
    print(f"  Shape latente (5 amostras): {X_latent.shape}")
    print(f"  Valores exemplo (1a amostra): {X_latent[0].round(4)}")

    print("\n" + "=" * 70)
    print("ENCODER TREINADO E SALVO COM SUCESSO!")
    print(f"  - models/encoder_16dims.keras")
    print(f"  - models/scaler_43features.pkl")
    print("=" * 70)


if __name__ == "__main__":
    main()
