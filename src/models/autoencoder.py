import numpy as np
import time
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
import sklearn

__PIPELINE_VERSION__ = 'v1.0.0-initial'

def get_package_versions():
    return {
        'tensorflow': tf.__version__,
        'keras': keras.__version__,
        'scikit-learn': sklearn.__version__,
        'pipeline': __PIPELINE_VERSION__,
    }

class KerasAutoencoder(BaseEstimator, TransformerMixin):
    def __init__(self, input_dim=None, latent_dim=16, hidden_dim=32, epochs=30, batch_size=32, learning_rate=1e-3,
                 random_state=42, validation_split=0.1, early_stopping=True, patience=5, verbose=0):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.validation_split = validation_split
        self.early_stopping = early_stopping
        self.patience = patience
        self.verbose = verbose
        self.model_ = None
        self.encoder_ = None
        self.history_ = None
        self._fitted = False
        self.best_val_loss_ = None
        self.best_epoch_ = None
        self.train_time_ = None
        self.version_info_ = get_package_versions()

    def build_model(self, input_dim):
        inputs = keras.Input(shape=(input_dim,))
        x = layers.Dense(self.hidden_dim, activation='relu')(inputs)
        latent = layers.Dense(self.latent_dim, activation='relu', name='latent')(x)
        x = layers.Dense(self.hidden_dim, activation='relu')(latent)
        outputs = layers.Dense(input_dim, activation=None)(x)
        autoencoder = keras.Model(inputs, outputs, name='autoencoder')
        encoder = keras.Model(inputs, latent, name='encoder')
        autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return autoencoder, encoder

    def fit(self, X, y=None):
        import random
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
        random.seed(self.random_state)

        X = np.array(X).astype(np.float32)
        n_samples, n_features = X.shape
        if self.input_dim is None:
            self.input_dim = n_features

        if not (0 < self.latent_dim < self.input_dim):
            raise ValueError(f"latent_dim ({self.latent_dim}) deve ser maior que 0 e menor que input_dim ({self.input_dim})")

        self.model_, self.encoder_ = self.build_model(self.input_dim)
        callbacks = []
        if self.early_stopping:
            callbacks.append(EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True, verbose=self.verbose))
        start_time = time.time()
        self.history_ = self.model_.fit(
            X, X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=self.verbose
        )
        self.train_time_ = time.time() - start_time
        # Extração dos melhores resultados
        val_losses = self.history_.history.get('val_loss', [])
        self.best_val_loss_ = min(val_losses) if val_losses else None
        self.best_epoch_ = (np.argmin(val_losses)+1) if val_losses else None
        self._fitted = True
        return self

    def transform(self, X):
        if not self._fitted:
            raise RuntimeError("Autoencoder not fitted. Call fit(X) first.")
        X = np.array(X).astype(np.float32)
        latent = self.encoder_.predict(X, batch_size=self.batch_size, verbose=0)
        # Safeguard: check nan/inf
        if np.any(np.isnan(latent)) or np.any(np.isinf(latent)):
            raise RuntimeError("Latent features contain NaN or Inf. Aborting.")
        return latent

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_config(self):
        return {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'hidden_dim': self.hidden_dim,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'random_state': self.random_state,
            'validation_split': self.validation_split,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'version_info': self.version_info_,
        }
    def get_training_summary(self):
        return {
            'best_val_loss': self.best_val_loss_,
            'best_epoch': self.best_epoch_,
            'train_time': self.train_time_,
        }

    def save(self, encoder_dir, model_name):
        import os
        import joblib
        os.makedirs(encoder_dir, exist_ok=True)
        encoder_path = os.path.join(encoder_dir, f"{model_name}_encoder.h5")
        config_path = os.path.join(encoder_dir, f"{model_name}_config.pkl")
        self.encoder_.save(encoder_path)
        joblib.dump(self.get_config(), config_path)
        # Também salva resumo de treino
        train_summary = self.get_training_summary()
        train_summary_path = os.path.join(encoder_dir, f"{model_name}_train_summary.pkl")
        joblib.dump(train_summary, train_summary_path)
        return encoder_path, config_path, train_summary_path

    @classmethod
    def load(cls, encoder_dir, model_name):
        import os
        import joblib
        encoder_path = os.path.join(encoder_dir, f"{model_name}_encoder.h5")
        config_path = os.path.join(encoder_dir, f"{model_name}_config.pkl")
        train_summary_path = os.path.join(encoder_dir, f"{model_name}_train_summary.pkl")
        config = joblib.load(config_path)
        obj = cls(**config)
        obj.encoder_ = keras.models.load_model(encoder_path)
        if os.path.exists(train_summary_path):
            obj.best_val_loss_, obj.best_epoch_, obj.train_time_ = joblib.load(train_summary_path).values()
        obj._fitted = True
        return obj
