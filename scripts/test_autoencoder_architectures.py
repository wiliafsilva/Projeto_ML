import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_multiple_seasons
from src.feature_engineering import calculate_team_stats
from scripts.autoencoder_latent_models import evaluate_models


SEED = 21

np.random.seed(SEED)
tf.random.set_seed(SEED)

TRAIN_DIR = "data/data_2005_2014"
TEST_DIR = "data/data_2014_2016"

OUTPUT = "models/original_vs_latent.json"


def load_features(directory_path):

    df = load_multiple_seasons(directory_path)
    df = df.reset_index(drop=True)

    features = calculate_team_stats(df)
    features = features.reset_index(drop=True)

    if "Result" not in features.columns:
        raise ValueError("Result column not found")

    y = features["Result"].astype(int).values

    seasons = (
        df["Season"].values
        if "Season" in df.columns
        else None
    )

    X = features.drop(
        columns=[
            c
            for c in ["Result", "Season"]
            if c in features.columns
        ]
    )

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.select_dtypes(
        include=[np.number]
    )

    if X.isna().any().any():
        X = X.fillna(
            X.median(numeric_only=True)
        )

    return (
        X.values.astype(np.float32),
        y,
        seasons
    )


def build_autoencoder(
    input_dim,
    latent_dim
):

    encoder = tf.keras.Sequential([

        layers.Input(
            shape=(input_dim,)
        ),

        layers.Dense(
            64,
            activation="relu"
        ),

        layers.Dense(
            32,
            activation="relu"
        ),

        layers.Dense(
            latent_dim,
            activation="linear",
            name="latent"
        )
    ])

    decoder = tf.keras.Sequential([

        layers.Dense(
            32,
            activation="relu"
        ),

        layers.Dense(
            64,
            activation="relu"
        ),

        layers.Dense(
            input_dim,
            activation="sigmoid"
        )
    ])

    class AutoEncoder(Model):

        def __init__(
            self,
            encoder,
            decoder
        ):
            super().__init__()

            self.encoder = encoder
            self.decoder = decoder

        def call(self, x):

            latent = self.encoder(x)

            reconstruction = self.decoder(
                latent
            )

            return reconstruction

    ae = AutoEncoder(
        encoder,
        decoder
    )

    ae.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="mae"
    )

    return ae


def reconstruction_metrics(
    ae,
    X_train_scaled,
    X_test_scaled
):

    X_train_rec = ae.predict(
        X_train_scaled,
        verbose=0
    )

    X_test_rec = ae.predict(
        X_test_scaled,
        verbose=0
    )

    return {

        "mae_train": float(
            np.mean(
                np.abs(
                    X_train_scaled
                    - X_train_rec
                )
            )
        ),

        "mse_train": float(
            np.mean(
                (
                    X_train_scaled
                    - X_train_rec
                ) ** 2
            )
        ),

        "mae_test": float(
            np.mean(
                np.abs(
                    X_test_scaled
                    - X_test_rec
                )
            )
        ),

        "mse_test": float(
            np.mean(
                (
                    X_test_scaled
                    - X_test_rec
                ) ** 2
            )
        )
    }


def main():

    print(
        "\n[INFO] Loading data..."
    )

    X_train, y_train, _ = load_features(
        TRAIN_DIR
    )

    X_test, y_test, seasons_test = load_features(
        TEST_DIR
    )

    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    results = {}

    print(
        "\n==========================="
    )

    print(
        "ORIGINAL FEATURES (43F)"
    )

    print(
        "==========================="
    )

    df_results, df_season = evaluate_models(
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        seasons=seasons_test
    )

    results["original_43_features"] = {

        "n_features":
            X_train_scaled.shape[1],

        "downstream_results":
            df_results.to_dict(
                orient="records"
            ),

        "downstream_by_season":
            df_season.to_dict(
                orient="records"
            )
    }

    latent_dims = [
        8,
        16,
        32
    ]

    for latent_dim in latent_dims:

        print(
            f"\n==========================="
        )

        print(
            f"LATENT {latent_dim}"
        )

        print(
            "==========================="
        )

        X_train_ae, X_val_ae = train_test_split(

            X_train_scaled,

            test_size=0.20,

            random_state=SEED,

            shuffle=True
        )

        ae = build_autoencoder(
            input_dim=X_train_scaled.shape[1],
            latent_dim=latent_dim
        )

        callbacks = [

            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                min_delta=1e-5
            ),

            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        history = ae.fit(

            X_train_ae,
            X_train_ae,

            validation_data=(
                X_val_ae,
                X_val_ae
            ),

            epochs=100,

            batch_size=128,

            callbacks=callbacks,

            verbose=1
        )

        metrics = reconstruction_metrics(
            ae,
            X_train_scaled,
            X_test_scaled
        )

        X_train_latent = ae.encoder.predict(
            X_train_scaled,
            verbose=0
        )

        X_test_latent = ae.encoder.predict(
            X_test_scaled,
            verbose=0
        )

        df_results, df_season = evaluate_models(

            X_train_latent,
            y_train,

            X_test_latent,
            y_test,

            seasons=seasons_test
        )

        results[f"latent_{latent_dim}"] = {

            "latent_dim":
                latent_dim,

            "best_val_loss":
                float(
                    min(
                        history.history[
                            "val_loss"
                        ]
                    )
                ),

            **metrics,

            "downstream_results":
                df_results.to_dict(
                    orient="records"
                ),

            "downstream_by_season":
                df_season.to_dict(
                    orient="records"
                )
        }

    os.makedirs(
        os.path.dirname(OUTPUT),
        exist_ok=True
    )

    with open(
        OUTPUT,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print(
        f"\n[INFO] Results saved to {OUTPUT}"
    )


if __name__ == "__main__":
    main()