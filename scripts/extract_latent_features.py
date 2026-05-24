"""
scripts/extract_latent_features.py
====================================
Extrai as 16 dimensões latentes para dados de treino e teste usando
o autoencoder já treinado.

Pipeline de 59 Features:
  - Dados (2005-2014): load → 43 features originais → Encoder → 16 latentes
  - Resultado final: 43 + 16 = 59 features por match
  - Dados de teste (2014-2016): mesmo processo para validação

Uso:
    python scripts/extract_latent_features.py

Pré-requisitos:
    - Encoder treinado: models/encoder_16dims.keras
    - Scaler treinado:  models/scaler_43features.pkl (ajustado nos dados 2005-2014)

Saídas:
    data/latent_features_train.csv  — 16 dims latentes para 2005-2014 (3.420 matches)
    data/latent_features_test.csv   — 16 dims latentes para 2014-2016 (760 matches)

Nota: Estes arquivos são depois combinados com as 43 features originais
no script combine_and_save_59features.py para gerar datasets com 59 features.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.preprocessing import load_multiple_seasons
from src.feature_engineering import calculate_team_stats
from src.encoder import load_encoder_models, extract_latent_features

# ── Configurações ──────────────────────────────────────────────────────────────
TRAIN_DIR  = "data/data_2005_2014"
TEST_DIR   = "data/data_2014_2016"
MODELS_DIR = "models"
DATA_DIR   = "data"
LATENT_DIM = 16

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
    available = [f for f in feature_list if f in df.columns]
    return df[available].fillna(0).values, available


def save_latent_csv(X_latent, df_original, filename, latent_dim=16):
    """Salva features latentes + Result + Season em CSV."""
    cols = [f"latent_{i}" for i in range(latent_dim)]
    df_latent = pd.DataFrame(X_latent, columns=cols, index=df_original.index)

    for col in ["Result", "Season"]:
        if col in df_original.columns:
            df_latent[col] = df_original[col].values

    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
    df_latent.to_csv(filename, index=False)
    print(f"  ✓ Salvo: {filename} — Shape: {df_latent.shape}")


def main():
    print("=" * 70)
    print("EXTRAÇÃO DE FEATURES LATENTES (16 DIMS)")
    print("=" * 70)

    # ── 1. Carregar encoder ────────────────────────────────────────────────────
    print(f"\n[1/5] Carregando encoder de {MODELS_DIR}...")
    encoder, scaler = load_encoder_models(path=MODELS_DIR)

    # ── 2. Carregar e processar dados de TREINO ────────────────────────────────
    print(f"\n[2/5] Carregando dados de treino: {TRAIN_DIR}")
    df_train_raw = load_multiple_seasons(TRAIN_DIR)
    df_train     = calculate_team_stats(df_train_raw, add_latent=False)
    X_train, used_feats = get_feature_matrix(df_train, CLASS_B_FEATURES)
    print(f"      X_train shape: {X_train.shape}")

    # ── 3. Carregar e processar dados de TESTE ─────────────────────────────────
    print(f"\n[3/5] Carregando dados de teste: {TEST_DIR}")
    df_test_raw = load_multiple_seasons(TEST_DIR)
    df_test     = calculate_team_stats(df_test_raw, add_latent=False)
    X_test, _   = get_feature_matrix(df_test, used_feats)  # Mesmas features
    print(f"      X_test  shape: {X_test.shape}")

    # ── 4. Extrair features latentes ───────────────────────────────────────────
    print(f"\n[4/5] Extraindo features latentes...")
    X_latent_train = extract_latent_features(X_train, encoder, scaler)
    X_latent_test  = extract_latent_features(X_test,  encoder, scaler)

    print(f"\n  Latentes treino shape: {X_latent_train.shape}")
    print(f"  Latentes teste  shape: {X_latent_test.shape}")

    # ── 5. Salvar CSVs ─────────────────────────────────────────────────────────
    print(f"\n[5/5] Salvando CSVs...")
    save_latent_csv(X_latent_train, df_train, f"{DATA_DIR}/latent_features_train.csv")
    save_latent_csv(X_latent_test,  df_test,  f"{DATA_DIR}/latent_features_test.csv")

    print("\n" + "=" * 70)
    print("✓ FEATURES LATENTES EXTRAÍDAS COM SUCESSO!")
    print(f"  → data/latent_features_train.csv")
    print(f"  → data/latent_features_test.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
