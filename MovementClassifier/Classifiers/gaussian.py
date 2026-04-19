import numpy as np
from sklearn.covariance import LedoitWolf
from .classifier import GenericClassifier
import json
import os

class GaussianClassifier(GenericClassifier):
    def __init__(self, threshold_scale: float=1.2):
        self.threshold_scale = threshold_scale

        self.models_ = {}
        self.thresholds_ = {}

    def fit(self, Z, y):
        # Iterate through labels
        for label in np.unique(y):
            # Subset with same label
            Zc = Z[y == label]

            # Covariance & mean
            mean = Zc.mean(axis=0)
            cov = LedoitWolf().fit(Zc).covariance_
            inv_cov = np.linalg.inv(cov)

            # Compute distances
            dists = []
            for z in Zc:
                d = z - mean
                dist = np.sqrt(d @ inv_cov @ d)
                dists.append(dist)

            self.models_[label] = (mean, inv_cov)
            self.thresholds_[label] = max(dists) * self.threshold_scale

    def predict(self, Z):
        preds = []

        for z in Z:
            best_label = None
            best_dist = float('inf')

            # Find closest match
            for label, (mean, inv_cov) in self.models_.items():
                d = z - mean
                dist = np.sqrt(d @ inv_cov @ d)

                if dist < best_dist:
                    best_dist = dist
                    best_label = label

            # Reject if above threshold
            if best_dist > self.thresholds_[best_label]:
                preds.append(None)
            else:
                preds.append(best_label)

        return np.array(preds)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)

        metadata = {
            "threshold_scale": getattr(self, "threshold_scale", None),
            "models_": getattr(self, "models_", None),
            "thresholds_": getattr(self, "thresholds_", None),
        }

        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, path: str):
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)

        obj = cls(threshold_scale=metadata["threshold_scale"])

        obj.models_ = metadata["models_"]
        obj.thresholds_ = metadata["thresholds_"]

        return obj

