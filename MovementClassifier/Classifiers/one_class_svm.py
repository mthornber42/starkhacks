from sklearn.svm import OneClassSVM
import numpy as np
from .classifier import GenericClassifier
import os
import json

class OneClassSVMClassifier(GenericClassifier):
    def __init__(self, nu: float=0.1, gamma: str='scale'):
        self.nu = nu
        self.gamma = gamma

        self.models_ = {}

    def fit(self, Z, y):
        self.models_ = {}

        # Iterate through labels
        for label in np.unique(y):
            # Subset with same label
            Zc = Z[y == label]

            # One class SVM fit to label
            model = OneClassSVM(kernel='rbf', nu=self.nu, gamma=self.gamma)
            model.fit(Zc)

            self.models_[label] = model

    def predict(self, Z):
        preds = []

        for z in Z:
            scores = {}

            # Find best-scoring model
            for label, model in self.models_.items():
                score = model.decision_function([z])[0]
                if score > 0:
                    scores[label] = score

            if len(scores) == 0:
                preds.append(None)
            elif len(scores) == 1:
                preds.append(next(iter(scores)))
            else:
                # multiple matches → choose best
                preds.append(max(scores, key=scores.get))

        return np.array(preds)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)

        metadata = {
            "nu": getattr(self, "nu", None),
            "gamma": getattr(self, "gamma", None),
            "models_": getattr(self, "models_", None),
        }

        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, path: str):
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)

        obj = cls(nu=metadata["nu"], gamma=metadata["gamma"])

        obj.models_ = metadata["models_"]

        return obj
