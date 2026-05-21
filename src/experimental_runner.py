import hashlib
import os
import subprocess
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.feature_engineering import calculate_team_stats
from src.models.autoencoder import KerasAutoencoder, get_package_versions
from src.preprocessing import load_multiple_seasons
from src.train_models import prepare_features_by_model, rps

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AUTOENCODER_DIR = os.path.join(ROOT, "models", "autoencoders")
os.makedirs(AUTOENCODER_DIR, exist_ok=True)
EXPERIMENTS_CSV = os.path.join(ROOT, "experiments_registry.csv")
ARTIFACTS_DIR = os.path.join(ROOT, "models", "experiments")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

MODELS = {
    "SVM": SVC(probability=True, kernel="rbf", C=0.1, gamma=0.001, random_state=42, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=2, min_samples_leaf=1, random_state=42, class_weight="balanced"),
    "XGBoost": XGBClassifier(eval_metric="mlogloss", n_estimators=200, max_depth=3, learning_rate=0.01, subsample=0.8, colsample_bytree=1.0, random_state=42),
    "NaiveBayes": GaussianNB(var_smoothing=1e-05),
}


def hash_dataframe(df):
    h = hashlib.sha256()
    # stable serialization
    cols = sorted(df.columns.tolist())
    h.update(",".join(cols).encode())
    # sample rows
    sample = df[cols].head(50).to_csv(index=False).encode()
    h.update(sample)
    return h.hexdigest()


def git_commit_hash():
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
        return out
    except Exception:
        return ""


def append_registry(row):
    cols = [
        "experiment_id",
        "experimental_round",
        "pipeline_type",
        "model_name",
        "season_range",
        "random_state",
        "latent_dim",
        "input_dim",
        "compression_ratio",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "rps",
        "train_time_seconds",
        "inference_time_seconds",
        "artifact_size_mb",
        "git_commit",
        "timestamp",
        "dataset_hash",
        "feature_hash",
        "scaler_hash",
        "encoder_hash",
        "success",
        "failure_reason",
    ]
    new = pd.DataFrame([row], columns=cols)
    header = not os.path.exists(EXPERIMENTS_CSV)
    new.to_csv(EXPERIMENTS_CSV, mode="a", index=False, header=header)


def save_artifact_row(csv_path, row):
    df = pd.DataFrame([row])
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", index=False, header=header)


def run_pipeline(pipeline_type, df_train, df_test, experimental_round=1):
    timestamp = datetime.utcnow().isoformat()
    git_commit = git_commit_hash()
    season_range = "2005-2016"

    # For each model, train according to pipeline
    for model_name, model in MODELS.items():
        try:
            print(f"\n[RUN] Pipeline={pipeline_type} Model={model_name}")
            df_train_model = prepare_features_by_model(df_train, model_name)
            df_test_model = prepare_features_by_model(df_test, model_name)

            X_train = df_train_model.drop(["Result", "Season"], axis=1)
            y_train = df_train_model["Result"]
            X_test = df_test_model.drop(["Result", "Season"], axis=1)
            y_test = df_test_model["Result"]

            input_dim = X_train.shape[1]
            latent_dim = None
            compression_ratio = None
            scaler_hash = ""
            encoder_hash = ""

            # Copy raw features for original pipeline
            if pipeline_type == "original":
                X_train_proc = X_train.copy()
                X_test_proc = X_test.copy()
            else:
                # baseline_scaled and autoencoder_latent both fit scaler on train
                scaler = StandardScaler()
                scaler.fit(X_train)
                scaler_path = os.path.join(AUTOENCODER_DIR, f"{model_name}_{pipeline_type}_scaler.pkl")
                joblib.dump(scaler, scaler_path)
                scaler_hash = hashlib.sha256(pd.util.hash_pandas_object(pd.Series(list(scaler.mean_))).values.tobytes()).hexdigest()
                X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
                X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
                if pipeline_type == "baseline_scaled":
                    X_train_proc = X_train_scaled
                    X_test_proc = X_test_scaled
                elif pipeline_type == "autoencoder_latent":
                    latent_dim = min(16, X_train_scaled.shape[1] // 2)
                    ae = KerasAutoencoder(input_dim=X_train_scaled.shape[1], latent_dim=latent_dim, random_state=42)
                    start = time.time()
                    ae.fit(X_train_scaled)
                    train_time_ae = time.time() - start
                    encoder_path, config_path, summary_path = ae.save(AUTOENCODER_DIR, f"{model_name}_auto_{latent_dim}_seed42_exp")
                    # encoder_hash: hash of config
                    try:
                        cfg = joblib.load(config_path)
                        encoder_hash = hashlib.sha256(str(cfg).encode()).hexdigest()
                    except Exception:
                        encoder_hash = ""
                    X_train_proc = pd.DataFrame(ae.transform(X_train_scaled), columns=[f"ae_{i}" for i in range(latent_dim)])
                    X_test_proc = pd.DataFrame(ae.transform(X_test_scaled), columns=[f"ae_{i}" for i in range(latent_dim)])
                    compression_ratio = latent_dim / X_train_scaled.shape[1]

            # Train
            t0 = time.time()
            model.fit(X_train_proc, y_train)
            train_time = time.time() - t0

            # Inference time (per sample average)
            t0 = time.time()
            probs = model.predict_proba(X_test_proc)
            preds = model.predict(X_test_proc)
            infer_time = time.time() - t0
            infer_time_per_sample = infer_time / max(1, len(X_test_proc))

            # Metrics
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="macro", zero_division=0)
            prec = precision_score(y_test, preds, average="macro", zero_division=0)
            rec = recall_score(y_test, preds, average="macro", zero_division=0)
            rps_score = rps(y_test.values, probs)

            # Save artifact model
            artifact_name = f"{pipeline_type}_{model_name}_round{experimental_round}.pkl"
            artifact_path = os.path.join(ARTIFACTS_DIR, artifact_name)
            joblib.dump(model, artifact_path)
            artifact_size_mb = os.path.getsize(artifact_path) / (1024 * 1024)

            # Feature hash
            feature_hash = hashlib.sha256(",".join(X_train.columns).encode()).hexdigest()

            # Dataset hash (train+test sample)
            dataset_hash = hashlib.sha256((str(len(df_train)) + str(len(df_test))).encode()).hexdigest()

            experiment_id = f"{model_name}_{pipeline_type}_{experimental_round}_{feature_hash[:8]}"

            row = {
                "experiment_id": experiment_id,
                "experimental_round": experimental_round,
                "pipeline_type": pipeline_type,
                "model_name": model_name,
                "season_range": season_range,
                "random_state": 42,
                "latent_dim": latent_dim if latent_dim is not None else "",
                "input_dim": input_dim,
                "compression_ratio": compression_ratio if compression_ratio is not None else "",
                "accuracy": acc,
                "f1": f1,
                "precision": prec,
                "recall": rec,
                "rps": rps_score,
                "train_time_seconds": train_time,
                "inference_time_seconds": infer_time_per_sample,
                "artifact_size_mb": artifact_size_mb,
                "git_commit": git_commit,
                "timestamp": timestamp,
                "dataset_hash": dataset_hash,
                "feature_hash": feature_hash,
                "scaler_hash": scaler_hash,
                "encoder_hash": encoder_hash,
                "success": True,
                "failure_reason": "",
            }

            append_registry(row)

            # Save per-pipeline artifact CSV line
            csv_map = {
                "original": os.path.join(ROOT, "models", "baseline_comparison_original.csv"),
                "baseline_scaled": os.path.join(ROOT, "models", "baseline_comparison_scaled.csv"),
                "autoencoder_latent": os.path.join(ROOT, "models", "baseline_comparison_autoencoder.csv"),
            }
            csv_path = csv_map.get(pipeline_type)
            artifact_row = {
                "pipeline_type": pipeline_type,
                "experimental_round": experimental_round,
                "git_commit": git_commit,
                "timestamp": timestamp,
                "model": model_name,
                "accuracy": acc,
                "f1": f1,
                "rps": rps_score,
            }
            save_artifact_row(csv_path, artifact_row)

            print(f"[OK] {pipeline_type} {model_name} acc={acc:.4f} f1={f1:.4f} rps={rps_score:.4f}")

        except Exception as e:
            ts = datetime.utcnow().isoformat()
            row = {
                "experiment_id": f"{model_name}_{pipeline_type}_{experimental_round}_error",
                "experimental_round": experimental_round,
                "pipeline_type": pipeline_type,
                "model_name": model_name,
                "season_range": season_range,
                "random_state": 42,
                "latent_dim": "",
                "input_dim": "",
                "compression_ratio": "",
                "accuracy": "",
                "f1": "",
                "precision": "",
                "recall": "",
                "rps": "",
                "train_time_seconds": "",
                "inference_time_seconds": "",
                "artifact_size_mb": "",
                "git_commit": git_commit,
                "timestamp": ts,
                "dataset_hash": "",
                "feature_hash": "",
                "scaler_hash": "",
                "encoder_hash": "",
                "success": False,
                "failure_reason": str(e)[:500],
            }
            append_registry(row)
            print(f"[ERROR] {pipeline_type} {model_name} -> {e}")


def main():
    # Load data and compute features
    train_dir = os.path.join(ROOT, "data", "data_2005_2014")
    test_dir = os.path.join(ROOT, "data", "data_2014_2016")
    df_train = load_multiple_seasons(train_dir)
    df_test = load_multiple_seasons(test_dir)
    features_train = calculate_team_stats(df_train)
    features_test = calculate_team_stats(df_test)

    # Pipelines to run
    pipelines = ["original", "baseline_scaled", "autoencoder_latent"]
    for p in pipelines:
        run_pipeline(p, features_train, features_test, experimental_round=1)


if __name__ == "__main__":
    main()
