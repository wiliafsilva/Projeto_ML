"""
scripts/combine_and_save_59features.py
=======================================
Combina as 43 features originais com as 16 latentes do autoencoder,
gerando arquivos CSV com 59 features totais (43 + 16 latentes, + Result + Season).

Nota: Este script é parte do pipeline de preparação de dados com autoencoder.
As 59 features são então usadas para treinamento de modelos ML.

Uso:
    python scripts/combine_and_save_59features.py

Pré-requisitos:
    - data/latent_features_train.csv
    - data/latent_features_test.csv

Saídas:
    data/features_59combined_train.csv
    data/features_59combined_test.csv
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.preprocessing import load_multiple_seasons
from src.feature_engineering import calculate_team_stats
from src.latent_features import combine_original_and_latent, save_combined_features

# ── Configurações ──────────────────────────────────────────────────────────────
TRAIN_DIR       = "data/data_2005_2014"
TEST_DIR        = "data/data_2014_2016"
LATENT_TRAIN    = "data/latent_features_train.csv"
LATENT_TEST     = "data/latent_features_test.csv"
OUT_TRAIN       = "data/features_59combined_train.csv"
OUT_TEST        = "data/features_59combined_test.csv"
LATENT_DIM      = 16


def load_latent_array(csv_path: str, latent_dim: int = 16) -> np.ndarray:
    """Lê CSV de features latentes e retorna apenas as colunas latent_*."""
    df = pd.read_csv(csv_path)
    cols = [f"latent_{i}" for i in range(latent_dim)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes em {csv_path}: {missing}")
    return df[cols].values


def main():
    print("=" * 80)
    print("COMBINAÇÃO DE FEATURES: 43 ORIGINAIS + 16 LATENTES (AUTOENCODER) = 59 FEATURES")
    print("=" * 80)

    # ── 1. Verificar pré-requisitos ────────────────────────────────────────────
    for path in [LATENT_TRAIN, LATENT_TEST]:
        if not os.path.exists(path):
            print(f"\n❌ Arquivo não encontrado: {path}")
            print("   Execute primeiro: python scripts/extract_latent_features.py")
            sys.exit(1)

    # ── 2. Carregar features originais ─────────────────────────────────────────
    print(f"\n[1/4] Carregando features originais de treino ({TRAIN_DIR})...")
    df_train_raw = load_multiple_seasons(TRAIN_DIR)
    df_train     = calculate_team_stats(df_train_raw)
    print(f"      df_train shape: {df_train.shape}")

    print(f"\n[2/4] Carregando features originais de teste ({TEST_DIR})...")
    df_test_raw = load_multiple_seasons(TEST_DIR)
    df_test     = calculate_team_stats(df_test_raw)
    print(f"      df_test  shape: {df_test.shape}")

    # ── 3. Carregar arrays latentes ────────────────────────────────────────────
    print(f"\n[3/4] Carregando features latentes...")
    X_latent_train = load_latent_array(LATENT_TRAIN, LATENT_DIM)
    X_latent_test  = load_latent_array(LATENT_TEST,  LATENT_DIM)
    print(f"      Latentes treino: {X_latent_train.shape}")
    print(f"      Latentes teste:  {X_latent_test.shape}")

    # ── Verificar alinhamento de registros ────────────────────────────────────
    if len(df_train) != len(X_latent_train):
        raise ValueError(
            f"Número de registros de treino diverge: "
            f"df_train={len(df_train)} vs latent={len(X_latent_train)}"
        )
    if len(df_test) != len(X_latent_test):
        raise ValueError(
            f"Número de registros de teste diverge: "
            f"df_test={len(df_test)} vs latent={len(X_latent_test)}"
        )

    # ── 4. Combinar e salvar ───────────────────────────────────────────────────
    print(f"\n[4/4] Combinando e salvando...")

    df_combined_train = combine_original_and_latent(df_train, X_latent_train, LATENT_DIM)
    save_combined_features(df_combined_train, OUT_TRAIN)

    df_combined_test = combine_original_and_latent(df_test, X_latent_test, LATENT_DIM)
    save_combined_features(df_combined_test, OUT_TEST)

    # ── Relatório final ────────────────────────────────────────────────────────
    feature_cols = [c for c in df_combined_train.columns if c not in ["Result", "Season"]]
    latent_cols  = [c for c in feature_cols if c.startswith("latent_")]
    orig_cols    = [c for c in feature_cols if not c.startswith("latent_")]

    print("\n" + "=" * 70)
    print("✓ COMBINAÇÃO CONCLUÍDA COM SUCESSO!")
    print(f"  Features originais : {len(orig_cols)}")
    print(f"  Features latentes  : {len(latent_cols)}")
    print(f"  Total de features  : {len(feature_cols)}")
    print(f"  → {OUT_TRAIN}")
    print(f"  → {OUT_TEST}")
    print("=" * 70)


if __name__ == "__main__":
    main()
