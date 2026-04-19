from .pipeline import Pipeline
import os
import json


class WaveletsPipeline(Pipeline):

    def fit(self, dataset):
        # Preprocessing + wavelet
        X_wave, y = dataset.get_wavelet_features()

        # Scale + PCA
        X_scaled = self.scaler.fit_transform(X_wave)
        self.pca.fit(X_scaled)

        # Save input dim
        self.input_dim_ = X_wave.shape[1]
        return self

    def transform(self, dataset):
        # Preprocessing + wavelet
        X_wave, _ = dataset.get_wavelet_features()

        # Check input dim
        if X_wave.shape[1] != self.input_dim_:
            raise ValueError("Feature dimension mismatch.")

        # Scale + PCA
        X_scaled = self.scaler.transform(X_wave)
        return self.pca.transform(X_scaled)

    def _save_extra(self, path: str):
        metadata = {
            "input_dim_": getattr(self, "input_dim_", None),
        }

        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    def _load_extra(self, path: str):
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)

        self.input_dim_ = metadata["input_dim_"]