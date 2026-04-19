import numpy as np
from .classifier import GenericClassifier
import json
import os

class CentroidClassifier(GenericClassifier):

    def __init__(self, threshold_scale: float=1.3):
        self.threshold_scale = threshold_scale

        self.centroids_ = {}
        self.thresholds_ = {}

    def fit(self, Z, y):
        self.centroids_ = {}
        self.thresholds_ = {}

        # Iterate through labels
        for label in np.unique(y):
            # Subset with same class
            Zc = Z[y == label]

            # Calculate centroid of class
            centroid = Zc.mean(axis=0)

            # Calculate distances to the centroid
            dists = np.linalg.norm(Zc - centroid, axis=1)

            self.centroids_[label] = centroid
            self.thresholds_[label] = dists.max() * self.threshold_scale

    def predict(self, Z):
        preds = []

        for z in Z:
            best_label = None
            best_dist = float('inf')

            # Find closest match
            for label, centroid in self.centroids_.items():
                d = np.linalg.norm(z - centroid)
                if d < best_dist:
                    best_dist = d
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
            "centroids_": list(getattr(self, "centroids_", None)),
            "thresholds_": list(getattr(self, "thresholds_", None)),
        }

        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, path: str):
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)

        obj = cls(threshold_scale=metadata["threshold_scale"])

        obj.centroids_ = metadata["centroids_"]
        obj.thresholds_ = metadata["thresholds_"]

        return obj
