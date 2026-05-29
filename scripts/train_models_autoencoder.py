import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import load_multiple_seasons
from src.feature_engineering import calculate_team_stats
from src.train_models import train_models_autoencoder


def main():
    train_dir = "data/data_2005_2014"
    test_dir = "data/data_2014_2016"

    if not os.path.exists(train_dir):
        raise ValueError(f"Diretorio de treinamento nao encontrado: {train_dir}")
    if not os.path.exists(test_dir):
        raise ValueError(f"Diretorio de teste nao encontrado: {test_dir}")

    df_train = load_multiple_seasons(train_dir)
    df_test = load_multiple_seasons(test_dir)

    features_train = calculate_team_stats(df_train)
    features_test = calculate_team_stats(df_test)

    train_models_autoencoder(features_train, features_test, latent_dim=8)


if __name__ == "__main__":
    main()
