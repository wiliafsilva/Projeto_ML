"""
Variational Autoencoder (VAE) - versão mais sofisticada
Melhor regularização, features latentes mais estruturadas
"""

import numpy as np
import pickle
import json
import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class VAE(keras.Model):
    """Variational Autoencoder"""
    
    def __init__(self, input_dim, latent_dim=20):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
        ])
        
        # Latent: mean e log_var (padrão VAE)
        self.mean_layer = layers.Dense(latent_dim, name='z_mean')
        self.logvar_layer = layers.Dense(latent_dim, name='z_log_var')
        
        # Decoder
        self.decoder = keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(input_dim, activation='linear'),
        ])
    
    def reparameterize(self, mean, logvar):
        """Reparameterization trick"""
        eps = tf.random.normal(shape=mean.shape)
        return mean + tf.exp(0.5 * logvar) * eps
    
    def encode(self, x):
        """Codificar para espaço latente"""
        h = self.encoder(x)
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        z = self.reparameterize(mean, logvar)
        return z, mean, logvar
    
    def decode(self, z):
        """Decodificar do espaço latente"""
        return self.decoder(z)
    
    def call(self, x):
        z, mean, logvar = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, mean, logvar


def vae_loss(x, reconstruction, mean, logvar):
    """VAE Loss = Reconstruction Loss + KL Divergence"""
    
    # Reconstruction loss (MSE)
    reconstruction_loss = tf.reduce_mean(
        tf.square(x - reconstruction)
    )
    
    # KL Divergence (regularização)
    kl_loss = -0.5 * tf.reduce_mean(
        1 + logvar - tf.square(mean) - tf.exp(logvar)
    )
    
    total_loss = reconstruction_loss + kl_loss
    return total_loss, reconstruction_loss, kl_loss


def train_vae(X_train, X_val, encoder_dir, latent_dim=20, epochs=100, batch_size=32):
    """Treina um VAE"""
    
    Path(encoder_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n[VAE] Iniciando treinamento...")
    print(f"  Input dim: {X_train.shape[1]}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")
    
    # Normalizar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Build VAE
    input_dim = X_train_scaled.shape[1]
    vae = VAE(input_dim=input_dim, latent_dim=latent_dim)
    
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    
    # Treinar
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(epochs):
        # Train
        train_loss = 0
        n_batches = 0
        
        for i in range(0, len(X_train_scaled), batch_size):
            x_batch = X_train_scaled[i:i+batch_size]
            
            with tf.GradientTape() as tape:
                reconstruction, mean, logvar = vae(x_batch)
                loss, rec_loss, kl_loss = vae_loss(x_batch, reconstruction, mean, logvar)
            
            grads = tape.gradient(loss, vae.trainable_weights)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))
            
            train_loss += loss.numpy()
            n_batches += 1
        
        train_loss /= n_batches
        
        # Validation
        reconstruction_val, mean_val, logvar_val = vae(X_val_scaled)
        val_loss, _, _ = vae_loss(X_val_scaled, reconstruction_val, mean_val, logvar_val)
        val_loss = val_loss.numpy()
        
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
    
    # Salvar encoder (apenas a parte que codifica)
    encoder_only = keras.Sequential([
        vae.encoder,
        vae.mean_layer,
    ])
    encoder_only.save(os.path.join(encoder_dir, 'vae_encoder.keras'))
    
    # Salvar scaler
    with open(os.path.join(encoder_dir, 'scaler_vae.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Salvar metadata
    metadata = {
        'type': 'VAE',
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'epochs_trained': epochs,
        'final_train_loss': float(train_loss_history[-1]),
        'final_val_loss': float(val_loss_history[-1]),
    }
    
    with open(os.path.join(encoder_dir, 'metadata_vae.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n[VAE] Treinamento completo!")
    print(f"  Final train loss: {train_loss_history[-1]:.6f}")
    print(f"  Final val loss: {val_loss_history[-1]:.6f}")
    
    return vae, scaler, {'train': train_loss_history, 'val': val_loss_history}


def encode_vae(X, vae, scaler):
    """Codificar com VAE"""
    X_scaled = scaler.transform(X)
    z, _, _ = vae.encode(X_scaled)
    return z.numpy()
