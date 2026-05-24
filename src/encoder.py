"""
src/encoder.py
==============
Encoder para compressão de features: 43 → 32 → 16 dimensões latentes.

IMPORTANTE: Este módulo implementa APENAS o encoder (parte de compressão),
sem o decoder. Isso é diferente de um autoencoder completo.

Arquitetura:
    Input (43) → Dense(32, relu) → BatchNorm → Dropout → Dense(16, relu) → Latent (16)

Validação temporal:
    - Scaler e encoder são FITADOS apenas nos dados 2005-2014
    - Aplicados (transform) nos dados 2014-2016 (sem vazamento)
"""

import numpy as np
import os
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler


# Reproducibilidade
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


def build_encoder(input_dim: int = 43, latent_dim: int = 16) -> keras.Model:
    """
    Constrói o modelo encoder (somente a parte de compressão).

    Arquitetura:
        Input(input_dim) → Dense(32, relu) → BatchNorm → Dropout(0.2)
                        → Dense(latent_dim, relu) → Output

    Args:
        input_dim: Número de features de entrada (padrão: 43)
        latent_dim: Número de dimensões latentes (padrão: 16)

    Returns:
        keras.Model: Modelo encoder compilado
    """
    inputs = keras.Input(shape=(input_dim,), name="encoder_input")

    # Camada oculta: 43 → 32
    x = layers.Dense(32, activation="relu", name="hidden_32")(inputs)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(0.2, seed=SEED, name="dropout")(x)

    # Camada latente: 32 → 16
    latent = layers.Dense(latent_dim, activation="relu", name="latent_16")(x)

    encoder = keras.Model(inputs=inputs, outputs=latent, name="encoder_43_to_16")

    encoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",  # Treinamento sem decoder: loss auxiliar de reconstrução interna
    )

    return encoder


def _build_autoencoder_for_training(
    input_dim: int = 43, latent_dim: int = 16
) -> tuple[keras.Model, keras.Model]:
    """
    Constrói um autoencoder temporário para treinar o encoder via reconstrução.
    O decoder é descartado após o treinamento.

    Returns:
        (autoencoder, encoder): Autoencoder completo e apenas o encoder
    """
    # ---- Encoder ----
    inputs = keras.Input(shape=(input_dim,), name="ae_input")
    x = layers.Dense(32, activation="relu", name="enc_hidden_32")(inputs)
    x = layers.BatchNormalization(name="enc_batch_norm")(x)
    x = layers.Dropout(0.2, seed=SEED, name="enc_dropout")(x)
    latent = layers.Dense(latent_dim, activation="relu", name="enc_latent_16")(x)

    # ---- Decoder (apenas para treinamento) ----
    x = layers.Dense(32, activation="relu", name="dec_hidden_32")(latent)
    reconstruction = layers.Dense(input_dim, activation="linear", name="dec_output")(x)

    # ---- Modelos ----
    autoencoder = keras.Model(inputs=inputs, outputs=reconstruction, name="autoencoder")
    encoder = keras.Model(inputs=inputs, outputs=latent, name="encoder_43_to_16")

    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )

    return autoencoder, encoder


def train_encoder(
    X_train: np.ndarray,
    input_dim: int = 43,
    latent_dim: int = 16,
    epochs: int = 100,
    batch_size: int = 32,
    validation_split: float = 0.2,
    verbose: int = 1,
) -> tuple:
    """
    Treina o encoder usando a estratégia de autoencoder (43 → 16 → 43).
    Após o treinamento, somente a parte do encoder é retornada.

    Passos:
        1. Fit do StandardScaler nos dados de treino
        2. Normalização dos dados
        3. Treinamento do autoencoder completo (com early stopping)
        4. Extração do encoder treinado
        5. Descarte do decoder

    Args:
        X_train: Array de features (N, 43) — dados crus NÃO normalizados
        input_dim: Dimensão de entrada (padrão: 43)
        latent_dim: Dimensão latente (padrão: 16)
        epochs: Número de épocas máximo (padrão: 100)
        batch_size: Tamanho do batch (padrão: 32)
        validation_split: Fração para validação interna (padrão: 0.2)
        verbose: Verbosidade do Keras (0=silencioso, 1=progresso)

    Returns:
        (encoder_model, scaler): Modelo encoder treinado e StandardScaler fitado
    """
    if isinstance(X_train, type(None)):
        raise ValueError("X_train não pode ser None")

    X_arr = np.array(X_train, dtype=np.float32)

    print(f"\n[Encoder] Shape de entrada: {X_arr.shape}")
    print(f"[Encoder] Arquitetura: {input_dim} → 32 → {latent_dim} → 32 → {input_dim}")

    # 1. Normalização (fit apenas no treino)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr).astype(np.float32)
    print(f"[Encoder] Normalização concluída (StandardScaler)")

    # 2. Construir autoencoder para treino
    autoencoder, encoder = _build_autoencoder_for_training(input_dim, latent_dim)

    print(f"\n[Encoder] Resumo da arquitetura:")
    encoder.summary(print_fn=lambda x: print(f"  {x}"))

    # 3. Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # 4. Treinar
    print(f"\n[Encoder] Iniciando treinamento ({epochs} épocas máx, batch={batch_size})...")
    history = autoencoder.fit(
        X_scaled,
        X_scaled,  # Target = input (reconstrução)
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=verbose,
    )

    best_val_loss = min(history.history["val_loss"])
    epochs_trained = len(history.history["loss"])
    print(f"\n[Encoder] ✓ Treinamento concluído!")
    print(f"[Encoder]   Épocas treinadas: {epochs_trained}")
    print(f"[Encoder]   Melhor val_loss:  {best_val_loss:.6f}")

    return encoder, scaler


def extract_latent_features(
    X: np.ndarray,
    encoder_model: keras.Model,
    scaler: StandardScaler,
) -> np.ndarray:
    """
    Extrai 16 dimensões latentes para um conjunto de dados.

    Args:
        X: Array de features originais (N, 43) — NÃO normalizadas
        encoder_model: Modelo encoder já treinado
        scaler: StandardScaler já fitado nos dados de treino

    Returns:
        X_latent: Array (N, 16) com as features latentes
    """
    X_arr = np.array(X, dtype=np.float32)
    X_scaled = scaler.transform(X_arr).astype(np.float32)
    X_latent = encoder_model.predict(X_scaled, verbose=0)

    print(f"[Encoder] Features latentes extraídas: {X_latent.shape}")
    return X_latent


def save_encoder_models(
    encoder: keras.Model,
    scaler: StandardScaler,
    path: str = "models",
    encoder_filename: str = "encoder_16dims.keras",
    scaler_filename: str = "scaler_43features.pkl",
) -> None:
    """
    Salva o encoder treinado e o scaler em disco.

    Args:
        encoder: Modelo encoder treinado
        scaler: StandardScaler fitado
        path: Diretório de destino (padrão: 'models')
        encoder_filename: Nome do arquivo do encoder
        scaler_filename: Nome do arquivo do scaler
    """
    os.makedirs(path, exist_ok=True)

    encoder_path = os.path.join(path, encoder_filename)
    scaler_path = os.path.join(path, scaler_filename)

    encoder.save(encoder_path)
    joblib.dump(scaler, scaler_path)

    print(f"[Encoder] ✓ Encoder salvo em: {encoder_path}")
    print(f"[Encoder] ✓ Scaler  salvo em: {scaler_path}")


def load_encoder_models(
    path: str = "models",
    encoder_filename: str = "encoder_16dims.keras",
    scaler_filename: str = "scaler_43features.pkl",
) -> tuple:
    """
    Carrega o encoder e o scaler salvos em disco.

    Args:
        path: Diretório onde estão os arquivos
        encoder_filename: Nome do arquivo do encoder
        scaler_filename: Nome do arquivo do scaler

    Returns:
        (encoder_model, scaler): Modelos carregados
    """
    encoder_path = os.path.join(path, encoder_filename)
    scaler_path = os.path.join(path, scaler_filename)

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder não encontrado em: {encoder_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler não encontrado em: {scaler_path}")

    encoder = keras.models.load_model(encoder_path)
    scaler = joblib.load(scaler_path)

    print(f"[Encoder] ✓ Encoder carregado de: {encoder_path}")
    print(f"[Encoder] ✓ Scaler  carregado de: {scaler_path}")

    return encoder, scaler
