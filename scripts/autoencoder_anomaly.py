import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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


class AnomalyDetector(Model):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu"),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(16, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(input_dim, activation="sigmoid"),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    train_dir = "data/data_2005_2014"
    output_dir = "models/autoencoder_anomaly"
    figures_dir = os.path.join(output_dir, "figures")
    top_n = 50
    threshold_percentiles = [95]

    ensure_dir(output_dir)
    ensure_dir(figures_dir)

    print("[Autoencoder] Loading data...")
    csv_files = sorted(
        [f for f in os.listdir(train_dir) if f.lower().endswith(".csv")]
    )
    print(f"[Autoencoder] CSV files found: {len(csv_files)}")
    for name in csv_files:
        print(f"  - {name}")
    df_train = load_multiple_seasons(train_dir)
    df_train = df_train.reset_index(drop=True)
    match_cols = [
        col for col in ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "Season"]
        if col in df_train.columns
    ]
    df_matches = df_train[match_cols].copy() if match_cols else None
    seasons = sorted(df_train["Season"].dropna().unique().tolist())
    print(f"[Autoencoder] Seasons in data: {len(seasons)} -> {seasons}")

    print("[Autoencoder] Building features...")
    features = calculate_team_stats(df_train)
    features = features.reset_index(drop=True)
    if df_matches is not None and len(df_matches) != len(features):
        min_len = min(len(df_matches), len(features))
        df_matches = df_matches.iloc[:min_len].reset_index(drop=True)
        features = features.iloc[:min_len].reset_index(drop=True)

    drop_cols = [c for c in ["Result", "Season"] if c in features.columns]
    features = features.drop(columns=drop_cols)

    # Force numeric to keep all engineered features consistent.
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.select_dtypes(include=[np.number]).copy()
    if features.isna().any().any():
        features = features.fillna(features.median(numeric_only=True))

    feature_columns = features.columns.tolist()
    print(f"[Autoencoder] Feature count: {len(feature_columns)}")
    print(f"[Autoencoder] Feature columns: {feature_columns}")

    data = features.values.astype(np.float32)
    indices = np.arange(len(data))

    train_data, test_data, train_idx, test_idx = train_test_split(
        data,
        indices,
        test_size=0.2,
        random_state=SEED,
    )

    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    input_dim = train_data.shape[1]
    autoencoder = AnomalyDetector(input_dim)
    autoencoder.compile(optimizer="adam", loss="mae")

    print("[Autoencoder] Training model...")
    max_batch = min(2048, len(train_data))
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
    history = autoencoder.fit(
        train_data,
        train_data,
        epochs=200,
        batch_size=max_batch,
        validation_data=(test_data, test_data),
        shuffle=True,
        verbose=1,
        callbacks=[early_stop, reduce_lr],
    )

    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    save_plot(os.path.join(figures_dir, "loss_curve.png"))

    reconstructions = autoencoder.predict(train_data, verbose=0)
    train_loss = tf.keras.losses.mae(reconstructions, train_data).numpy()

    plt.figure(figsize=(6, 4))
    plt.hist(train_loss, bins=50)
    plt.xlabel("Train loss")
    plt.ylabel("No of examples")
    plt.title("Reconstruction Error (Train)")
    save_plot(os.path.join(figures_dir, "train_loss_hist.png"))

    threshold = float(np.percentile(train_loss, threshold_percentiles[0]))
    print(f"[Autoencoder] Threshold (p{threshold_percentiles[0]}): {threshold:.6f}")

    reconstructions = autoencoder.predict(test_data, verbose=0)
    test_loss = tf.keras.losses.mae(reconstructions, test_data).numpy()

    plt.figure(figsize=(6, 4))
    plt.hist(test_loss, bins=50)
    plt.xlabel("Test loss")
    plt.ylabel("No of examples")
    plt.title("Reconstruction Error (Test)")
    save_plot(os.path.join(figures_dir, "test_loss_hist.png"))

    test_anomaly = test_loss > threshold

    percentile_rows = []
    for perc in threshold_percentiles:
        perc_threshold = float(np.percentile(train_loss, perc))
        perc_anomaly = test_loss > perc_threshold
        anomaly_rate = float(perc_anomaly.mean()) if len(test_loss) else 0.0
        percentile_rows.append({
            "percentile": perc,
            "threshold": perc_threshold,
            "test_anomalies": int(perc_anomaly.sum()),
            "test_anomaly_rate": anomaly_rate,
        })

        perc_top_n = min(top_n, len(test_loss))
        perc_order = np.argsort(test_loss)[-perc_top_n:][::-1]
        perc_rows = features.iloc[test_idx[perc_order]].copy()
        if df_matches is not None:
            match_rows = df_matches.iloc[test_idx[perc_order]].reset_index(drop=True)
            perc_rows = pd.concat([match_rows, perc_rows.reset_index(drop=True)], axis=1)
        perc_rows.insert(0, "row_index", test_idx[perc_order])
        perc_rows.insert(1, "reconstruction_error", test_loss[perc_order])
        perc_rows.insert(2, "is_anomaly", perc_anomaly[perc_order])
        perc_rows.to_csv(
            os.path.join(output_dir, f"top_anomalies_p{perc}.csv"),
            index=False,
        )
        if df_matches is not None:
            perc_rows.to_csv(
                os.path.join(output_dir, f"top_anomalies_p{perc}_matches.csv"),
                index=False,
            )

    pd.DataFrame(percentile_rows).to_csv(
        os.path.join(output_dir, "percentile_results.csv"),
        index=False,
    )

    results_df = pd.DataFrame({
        "split": ["train"] * len(train_loss) + ["test"] * len(test_loss),
        "loss": np.concatenate([train_loss, test_loss]),
        "is_anomaly": np.concatenate([
            np.zeros_like(train_loss, dtype=bool),
            test_anomaly,
        ]),
    })
    results_df.to_csv(os.path.join(output_dir, "reconstruction_errors.csv"), index=False)

    summary = {
        "threshold": threshold,
        "threshold_percentile": threshold_percentiles[0],
        "train_count": int(len(train_loss)),
        "test_count": int(len(test_loss)),
        "test_anomalies": int(test_anomaly.sum()),
        "test_anomaly_rate": float(test_anomaly.mean()) if len(test_loss) else 0.0,
        "train_loss_mean": float(np.mean(train_loss)),
        "train_loss_std": float(np.std(train_loss)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Pick a low-loss and high-loss sample for plots
    test_sorted_idx = np.argsort(test_loss)
    normal_idx = test_sorted_idx[int(0.1 * len(test_sorted_idx))]
    anomalous_idx = test_sorted_idx[int(0.99 * len(test_sorted_idx))]

    normal_input = test_data[normal_idx]
    anomalous_input = test_data[anomalous_idx]

    normal_decoded = autoencoder.decoder(autoencoder.encoder(normal_input[None, :])).numpy()[0]
    anomalous_decoded = autoencoder.decoder(autoencoder.encoder(anomalous_input[None, :])).numpy()[0]

    x_axis = np.arange(input_dim)

    plt.figure(figsize=(6, 4))
    plt.grid(True)
    plt.plot(x_axis, normal_input)
    plt.title("A Normal Sample")
    save_plot(os.path.join(figures_dir, "normal_sample.png"))

    plt.figure(figsize=(6, 4))
    plt.grid(True)
    plt.plot(x_axis, anomalous_input)
    plt.title("An Anomalous Sample")
    save_plot(os.path.join(figures_dir, "anomalous_sample.png"))

    plt.figure(figsize=(6, 4))
    plt.plot(normal_input, "b")
    plt.plot(normal_decoded, "r")
    plt.fill_between(x_axis, normal_decoded, normal_input, color="lightcoral")
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    plt.title("Normal Sample Reconstruction")
    save_plot(os.path.join(figures_dir, "normal_reconstruction.png"))

    plt.figure(figsize=(6, 4))
    plt.plot(anomalous_input, "b")
    plt.plot(anomalous_decoded, "r")
    plt.fill_between(x_axis, anomalous_decoded, anomalous_input, color="lightcoral")
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    plt.title("Anomalous Sample Reconstruction")
    save_plot(os.path.join(figures_dir, "anomalous_reconstruction.png"))

    print("[Autoencoder] Done.")
    print(f"[Autoencoder] Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
