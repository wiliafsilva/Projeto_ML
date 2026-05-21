"""
Módulo para construir, treinar e usar Autoencoders.
Implementação usando TensorFlow/Keras para redução de dimensionalidade.
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def build_autoencoder(input_dim, latent_dim=16, hidden_layers=[64, 32], dropout_rate=0.2):
    """
    Constrói um modelo autoencoder simples (simétrico).
    
    Arquitetura:
    Input -> Dense(hidden_layers[0]) -> Dense(hidden_layers[1]) -> Dense(latent_dim)
    -> Dense(hidden_layers[1]) -> Dense(hidden_layers[0]) -> Dense(input_dim)
    
    Args:
        input_dim (int): Dimensão de entrada (número de features)
        latent_dim (int): Dimensão do espaço latente (default: 16)
        hidden_layers (list): Dimensões das camadas ocultas (default: [64, 32])
        dropout_rate (float): Taxa de dropout (default: 0.2)
    
    Returns:
        tuple: (model_autoencoder, encoder_model)
    """
    # Entrada
    input_layer = layers.Input(shape=(input_dim,), name='input')
    
    # Encoder
    x = input_layer
    for i, units in enumerate(hidden_layers):
        x = layers.Dense(units, activation='relu', name=f'encoder_dense_{i}')(x)
        x = layers.Dropout(dropout_rate, name=f'encoder_dropout_{i}')(x)
    
    # Espaço latente
    latent = layers.Dense(latent_dim, activation='linear', name='latent')(x)
    
    # Decoder
    x = latent
    for i, units in enumerate(reversed(hidden_layers)):
        x = layers.Dense(units, activation='relu', name=f'decoder_dense_{i}')(x)
        x = layers.Dropout(dropout_rate, name=f'decoder_dropout_{i}')(x)
    
    # Saída (reconstrução)
    output = layers.Dense(input_dim, activation='linear', name='output')(x)
    
    # Modelo completo (autoencoder)
    autoencoder = Model(input_layer, output, name='autoencoder')
    
    # Modelo apenas do encoder (para gerar features latentes)
    encoder = Model(input_layer, latent, name='encoder')
    
    return autoencoder, encoder


def train_autoencoder(X_train, X_val, encoder_dir, latent_dim=16, 
                     epochs=100, batch_size=32, learning_rate=1e-3, 
                     patience=10, verbose=1):
    """
    Treina um autoencoder e salva encoder + scaler.
    
    Args:
        X_train (np.ndarray): Features de treino (shape: (n_samples, n_features))
        X_val (np.ndarray): Features de validação (shape: (n_samples, n_features))
        encoder_dir (str): Diretório para salvar artefatos
        latent_dim (int): Dimensão do espaço latente (default: 16)
        epochs (int): Número máximo de epochs (default: 100)
        batch_size (int): Tamanho do batch (default: 32)
        learning_rate (float): Taxa de aprendizado (default: 1e-3)
        patience (int): Patience para EarlyStopping (default: 10)
        verbose (int): Nível de verbosidade (default: 1)
    
    Returns:
        tuple: (encoder_model, scaler, history_dict, metadata_dict)
    """
    # Criar diretório se não existir
    Path(encoder_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Autoencoder] Iniciando treinamento...")
    print(f"  Input dim: {X_train.shape[1]}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")
    
    # Normalizar dados
    print(f"\n[Preprocessing] Normalizando features com StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    print(f"  Mean: {scaler.mean_[:3]}... (primeiras 3 features)")
    print(f"  Scale: {scaler.scale_[:3]}... (primeiras 3 features)")
    
    # Build model
    input_dim = X_train_scaled.shape[1]
    autoencoder, encoder = build_autoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_layers=[64, 32],
        dropout_rate=0.2
    )
    
    # Compilar
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    print(f"\n[Model] Arquitetura do autoencoder:")
    autoencoder.summary()
    
    # EarlyStopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    # Treinar
    print(f"\n[Training] Iniciando treinamento (máx {epochs} epochs)...")
    history = autoencoder.fit(
        X_train_scaled, X_train_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_scaled, X_val_scaled),
        callbacks=[early_stop],
        verbose=verbose
    )
    
    # Salvar artefatos
    print(f"\n[Saving] Salvando encoder e metadados...")
    
    # Encoder
    encoder_path = os.path.join(encoder_dir, 'encoder.keras')
    encoder.save(encoder_path)
    print(f"  ✓ Encoder salvo: {encoder_path}")
    
    # Scaler
    scaler_path = os.path.join(encoder_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  ✓ Scaler salvo: {scaler_path}")
    
    # Metadata
    metadata = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'hidden_layers': [64, 32],
        'dropout_rate': 0.2,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epochs_trained': len(history.history['loss']),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
    }
    
    metadata_path = os.path.join(encoder_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata salvo: {metadata_path}")
    
    # History
    history_dict = {
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
        'mae': history.history.get('mae', []),
        'val_mae': history.history.get('val_mae', [])
    }
    
    history_path = os.path.join(encoder_dir, 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history_dict, f)
    print(f"  ✓ History salvo: {history_path}")
    
    print(f"\n[Training] Completo!")
    print(f"  Final train MSE: {metadata['final_train_loss']:.6f}")
    print(f"  Final val MSE: {metadata['final_val_loss']:.6f}")
    
    return encoder, scaler, history_dict, metadata


def load_encoder_and_scaler(encoder_dir):
    """
    Carrega encoder e scaler salvos.
    
    Args:
        encoder_dir (str): Diretório contendo encoder.keras e scaler.pkl
    
    Returns:
        tuple: (encoder_model, scaler_object, metadata_dict)
    """
    # Carregar encoder
    encoder_path = os.path.join(encoder_dir, 'encoder.keras')
    encoder = keras.models.load_model(encoder_path)
    
    # Carregar scaler
    scaler_path = os.path.join(encoder_dir, 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Carregar metadata
    metadata_path = os.path.join(encoder_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return encoder, scaler, metadata


def encode(X, encoder, scaler):
    """
    Normaliza features e passa pelo encoder para gerar representação latente.
    
    Args:
        X (np.ndarray ou pd.DataFrame): Features originais
        encoder (keras.Model): Modelo encoder treinado
        scaler (StandardScaler): Scaler normalizado em treino
    
    Returns:
        np.ndarray: Features latentes (shape: (n_samples, latent_dim))
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    X_scaled = scaler.transform(X)
    X_latent = encoder.predict(X_scaled, verbose=0)
    
    return X_latent


def encode_batch(X_list, encoder, scaler, batch_size=1000):
    """
    Codifica múltiplos batches (util para datasets grandes).
    
    Args:
        X_list (list): Lista de arrays de features
        encoder (keras.Model): Modelo encoder
        scaler (StandardScaler): Scaler
        batch_size (int): Tamanho do batch para predição
    
    Returns:
        list: Lista de features latentes
    """
    latent_list = []
    
    for X in X_list:
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = scaler.transform(X)
        X_latent = encoder.predict(X_scaled, batch_size=batch_size, verbose=0)
        latent_list.append(X_latent)
    
    return latent_list
