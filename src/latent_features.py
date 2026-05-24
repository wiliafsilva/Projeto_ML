"""
src/latent_features.py
======================
Combina as 43 features originais com as 16 dimensões latentes do encoder,
gerando um DataFrame de 59 features totais.

Estrutura das colunas resultantes:
    - Colunas 0..42  : features originais (nomes mantidos)
    - Colunas 43..58 : latent_0, latent_1, ..., latent_15
    - Coluna Result  : label alvo (H=0, D=1, A=2)
    - Coluna Season  : temporada
"""

import numpy as np
import pandas as pd
import os


def combine_original_and_latent(
    df_original: pd.DataFrame,
    X_latent: np.ndarray,
    latent_dim: int = 16,
) -> pd.DataFrame:
    """
    Combina um DataFrame de features originais com as dimensões latentes do encoder.

    Args:
        df_original: DataFrame com features originais + colunas 'Result' e 'Season'
        X_latent: Array (N, latent_dim) com as features latentes
        latent_dim: Número de dimensões latentes (padrão: 16)

    Returns:
        DataFrame (N, 43 + latent_dim + 2) com colunas originais + latent_0..latent_{n-1}
        + Result + Season
    """
    if len(df_original) != len(X_latent):
        raise ValueError(
            f"Tamanhos incompatíveis: df_original={len(df_original)}, "
            f"X_latent={len(X_latent)}"
        )

    # Criar DataFrame com features latentes
    latent_cols = [f"latent_{i}" for i in range(latent_dim)]
    df_latent = pd.DataFrame(X_latent, columns=latent_cols, index=df_original.index)

    # Separar features originais das colunas de controle
    control_cols = [c for c in ["Result", "Season"] if c in df_original.columns]
    feature_cols = [c for c in df_original.columns if c not in control_cols]

    # Concatenar: features originais | latentes | controle
    df_combined = pd.concat(
        [df_original[feature_cols], df_latent, df_original[control_cols]],
        axis=1,
    )

    n_orig = len(feature_cols)
    n_lat = latent_dim
    print(
        f"[LatentFeatures] ✓ Features combinadas: "
        f"{n_orig} originais + {n_lat} latentes = {n_orig + n_lat} features totais"
    )
    print(f"[LatentFeatures]   Shape final: {df_combined.shape}")

    # Verificar nomes únicos
    feature_names = [c for c in df_combined.columns if c not in control_cols]
    duplicates = [c for c in feature_names if feature_names.count(c) > 1]
    if duplicates:
        raise ValueError(f"[LatentFeatures] Nomes de colunas duplicados: {set(duplicates)}")

    return df_combined


def save_combined_features(
    df_combined: pd.DataFrame,
    filename: str = "data/features_59combined.csv",
) -> None:
    """
    Salva o DataFrame combinado em CSV.

    Args:
        df_combined: DataFrame com 59 features + Result + Season
        filename: Caminho do arquivo de saída
    """
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
    df_combined.to_csv(filename, index=False)
    print(f"[LatentFeatures] ✓ Salvo em: {filename} ({len(df_combined)} registros)")


def load_combined_features(
    filename: str = "data/features_59combined.csv",
) -> pd.DataFrame:
    """
    Carrega o DataFrame combinado de um CSV.

    Args:
        filename: Caminho do arquivo

    Returns:
        DataFrame com 59 features
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Arquivo não encontrado: {filename}")

    df = pd.read_csv(filename)
    print(f"[LatentFeatures] ✓ Carregado: {filename} — Shape: {df.shape}")
    return df


def get_latent_feature_names(latent_dim: int = 16) -> list[str]:
    """
    Retorna os nomes das features latentes.

    Args:
        latent_dim: Número de dimensões latentes (padrão: 16)

    Returns:
        Lista com nomes ['latent_0', 'latent_1', ..., 'latent_{n-1}']
    """
    return [f"latent_{i}" for i in range(latent_dim)]
