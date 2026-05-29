import json
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_multiple_seasons
from src.feature_engineering import calculate_team_stats


SEED = 21
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Maximize CPU utilization for TensorFlow
CPU_COUNT = os.cpu_count() or 1
tf.config.threading.set_intra_op_parallelism_threads(CPU_COUNT)
tf.config.threading.set_inter_op_parallelism_threads(CPU_COUNT)


class Autoencoder(Model):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(latent_dim, activation="relu"),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(32, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(input_dim, activation="sigmoid"),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_features(directory_path):
    df = load_multiple_seasons(directory_path)
    df = df.reset_index(drop=True)
    features = calculate_team_stats(df)
    features = features.reset_index(drop=True)

    if "Result" not in features.columns:
        raise ValueError("Result column not found in features")

    y = features["Result"].astype(int).values
    seasons = df["Season"].values if "Season" in df.columns else None
    X = features.drop(columns=[c for c in ["Result", "Season"] if c in features.columns])
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.select_dtypes(include=[np.number]).copy()
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))

    return X, y, seasons


def train_autoencoder(X_train, X_val, latent_dim=8):
    autoencoder = Autoencoder(X_train.shape[1], latent_dim=latent_dim)
    autoencoder.compile(optimizer="adam", loss="mae")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        min_delta=1e-4,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
    )

    autoencoder.fit(
        X_train,
        X_train,
        epochs=200,
        batch_size=min(2048, len(X_train)),
        validation_data=(X_val, X_val),
        shuffle=True,
        verbose=1,
        callbacks=[early_stop, reduce_lr],
    )

    return autoencoder


def evaluate_models(X_train_latent, y_train, X_test_latent, y_test, seasons=None):
    results = []
    season_results = []

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=SEED),
        "SVM": SVC(kernel="rbf", C=5.0, gamma="scale", probability=True, random_state=SEED),
        "NaiveBayes": GaussianNB(),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=SEED,
            eval_metric="mlogloss",
        )
    except Exception:
        pass

    for name, model in models.items():
        model.fit(X_train_latent, y_train)
        preds = model.predict(X_test_latent)
        results.append({
            "model": name,
            "accuracy": accuracy_score(y_test, preds),
            "f1_macro": f1_score(y_test, preds, average="macro"),
        })

        if seasons is not None:
            for season in sorted(np.unique(seasons)):
                mask = seasons == season
                if not np.any(mask):
                    continue
                season_results.append({
                    "season": int(season),
                    "model": name,
                    "accuracy": accuracy_score(y_test[mask], preds[mask]),
                    "f1_macro": f1_score(y_test[mask], preds[mask], average="macro"),
                })

    season_results_df = pd.DataFrame(season_results)

    return pd.DataFrame(results), season_results_df


def main():
    train_dir = "data/data_2005_2014"
    test_dir = "data/data_2014_2016"
    output_dir = "models/autoencoder_latent"
    ensure_dir(output_dir)

    print("[Latent] Loading features...")
    X_train, y_train, _ = load_features(train_dir)
    X_test, y_test, seasons_test = load_features(test_dir)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.values.astype(np.float32))
    X_test_scaled = scaler.transform(X_test.values.astype(np.float32))

    print("[Latent] Training autoencoder...")
    autoencoder = train_autoencoder(X_train_scaled, X_test_scaled, latent_dim=8)

    print("[Latent] Encoding features...")
    X_train_latent = autoencoder.encoder(X_train_scaled).numpy()
    X_test_latent = autoencoder.encoder(X_test_scaled).numpy()

    print("[Latent] Training classifiers on latent space...")
    results_df, season_results_df = evaluate_models(
        X_train_latent,
        y_train,
        X_test_latent,
        y_test,
        seasons=seasons_test,
    )

    results_df.to_csv(os.path.join(output_dir, "latent_model_results.csv"), index=False)
    if not season_results_df.empty:
        season_results_df.to_csv(
            os.path.join(output_dir, "latent_model_results_by_season.csv"),
            index=False,
        )

    summary = {
        "latent_dim": 8,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": int(X_train.shape[1]),
        "models": results_df.to_dict(orient="records"),
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[Latent] Done.")
    print(f"[Latent] Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
