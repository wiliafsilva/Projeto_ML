import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.base import BaseEstimator, TransformerMixin

class KerasAutoencoder(BaseEstimator, TransformerMixin):
    def __init__(self, input_dim=None, latent_dim=16, hidden_dim=32, epochs=50, batch_size=32, learning_rate=1e-3, random_state=42, verbose=0):
        self.input_dim = input_dim  # Número de features
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.verbose = verbose
        self.model_ = None
        self.encoder_ = None
        self.history_ = None
        self._fitted = False

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
        X = np.array(X).astype(np.float32)
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        self.model_, self.encoder_ = self.build_model(self.input_dim)
        self.history_ = self.model_.fit(
            X, X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            verbose=self.verbose
        )
        self._fitted = True
        return self

    def transform(self, X):
        if not self._fitted:
            raise RuntimeError("Autoencoder not fitted; call fit(X) first.")
        X = np.array(X).astype(np.float32)
        return self.encoder_.predict(X, batch_size=self.batch_size, verbose=0)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def get_config(self):
        return {
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'hidden_dim': self.hidden_dim,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'random_state': self.random_state,
        }

    def save(self, model_path_prefix):
        if not self._fitted:
            raise RuntimeError("Autoencoder not fitted -- nothing to save.")
        self.model_.save(f"{model_path_prefix}_autoencoder.h5")
        self.encoder_.save(f"{model_path_prefix}_encoder.h5")
        # Save params (except model weights)
        import joblib
        joblib.dump(self.get_config(), f"{model_path_prefix}_config.pkl")

    @classmethod
    def load(cls, model_path_prefix):
        import joblib
        config = joblib.load(f"{model_path_prefix}_config.pkl")
        obj = cls(**config)
        obj.model_ = keras.models.load_model(f"{model_path_prefix}_autoencoder.h5")
        obj.encoder_ = keras.models.load_model(f"{model_path_prefix}_encoder.h5")
        obj._fitted = True
        return obj
