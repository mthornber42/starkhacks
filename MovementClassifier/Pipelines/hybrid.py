from .pipeline import Pipeline, build_autoencoder
import os
import json
import tensorflow as tf

class HybridPipeline(Pipeline):
    def __init__(self, *, n_components: int=3, embedding_dim: int=16,
                 epochs: int=10, batch_size: int=32):
        super().__init__(n_components=n_components)

        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size

        # Encoders
        self.autoencoder_ = None
        self.encoder_ = None

    def fit(self, dataset):
        # Preprocess + wavelets
        X_wave, y = dataset.get_wavelet_features()

        X_wave = X_wave.astype("float32")

        # Train autoencoder on wavelet features
        self.autoencoder_, self.encoder_ = build_autoencoder(
            X_wave.shape[1], embedding_dim=self.embedding_dim
        )
        self.autoencoder_.compile(optimizer='adam', loss='mse')

        self.autoencoder_.fit(
            X_wave, X_wave,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=0
        )

        # Get embeddings
        X_emb = self.encoder_.predict(X_wave, verbose=0)

        # Scale + PCA
        X_scaled = self.scaler.fit_transform(X_emb)
        self.pca.fit(X_scaled)

        # Save input dim
        self.input_dim_ = X_wave.shape[1]
        return self

    def transform(self, dataset):
        # Preprocess + wavelets
        X_wave, y = dataset.get_wavelet_features()

        X_wave = X_wave.astype("float32")

        # Check input dim
        if X_wave.shape[1] != self.input_dim_:
            raise ValueError("Feature dimension mismatch.")

        # Get embeddings
        X_emb = self.encoder_.predict(X_wave, verbose=0)

        # Scale + PCA
        X_scaled = self.scaler.transform(X_emb)
        return self.pca.transform(X_scaled)

    def _save_extra(self, path: str):
        # --- Save Keras models ---
        if self.autoencoder_ is not None:
            self.autoencoder_.save(os.path.join(path, "autoencoder"))

        if self.encoder_ is not None:
            self.encoder_.save(os.path.join(path, "encoder"))

        # --- Save metadata ---
        metadata = {
            "embedding_dim": self.embedding_dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "input_dim_": getattr(self, "input_dim_", None),
        }

        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    def _load_extra(self, path: str):
        # --- Load Keras models ---
        self.autoencoder_ = tf.keras.models.load_model(
            os.path.join(path, "autoencoder")
        )

        self.encoder_ = tf.keras.models.load_model(
            os.path.join(path, "encoder")
        )

        # --- Load metadata ---
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)

        self.embedding_dim = metadata["embedding_dim"]
        self.epochs = metadata["epochs"]
        self.batch_size = metadata["batch_size"]
        self.input_dim_ = metadata["input_dim_"]

