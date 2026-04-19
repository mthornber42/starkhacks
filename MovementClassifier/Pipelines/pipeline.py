import numpy as np
import os
import joblib
from abc import ABC, abstractmethod
from MovementClassifier.dataset import Dataset
from MovementClassifier.Classifiers.classifier import GenericClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

from keras.models import Model
from keras.layers import Input, Dense


def build_autoencoder(input_dim, embedding_dim=32):
    inputs = Input(shape=(input_dim,))

    # Encoder
    x = Dense(128, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    embedding = Dense(embedding_dim, activation='linear', name="embedding")(x)

    # Decoder
    x = Dense(64, activation='relu')(embedding)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(input_dim, activation='linear')(x)

    autoencoder = Model(inputs, outputs)
    encoder = Model(inputs, embedding)

    return autoencoder, encoder

class Pipeline(ABC):

    def __init__(self, *, n_components=32):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)

    @abstractmethod
    def fit(self, dataset: Dataset):
        raise NotImplementedError

    @abstractmethod
    def transform(self, dataset: Dataset) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, dataset: Dataset) -> np.ndarray:
        return self.fit(dataset).transform(dataset)

    @classmethod
    def evaluate(cls, *, dataset_labeled: Dataset, dataset_unlabeled: Dataset,
                 classifier: GenericClassifier,
                 n_splits: int=5, random_state: int=42, clf_kwargs: dict={},
                 **kwargs) -> tuple[list, list]:
        X_lab = dataset_labeled.X_raw
        y = np.array(dataset_labeled.y)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=random_state)

        all_true = []
        all_pred = []

        for train_idx, val_idx in skf.split(X_lab, y):
            train_data = dataset_labeled.subset(train_idx, augment=True)
            val_data = dataset_labeled.subset(val_idx)

            # ---- BUILD PIPELINE FRESH ----
            pipe = cls(**kwargs)

            # ---- CREATE DATA FOR UNSUPERVISED FIT ----
            if dataset_unlabeled is not None:
                # combine unlabeled + train fold
                unsup_data = dataset_unlabeled
            else:
                unsup_data = train_data  # fallback

            # ---- FIT REPRESENTATION (UNSUPERVISED) ----
            pipe.fit(unsup_data)

            # ---- TRANSFORM LABELED DATA ----
            X_train_feat = pipe.transform(train_data)
            y_train = np.array(train_data._X_cached[1])

            X_val_feat = pipe.transform(val_data)
            y_val = np.array(val_data.y)

            # ---- TRAIN CLASSIFIER ----
            clf = classifier(**clf_kwargs)  # fresh model each fold
            clf.fit(X_train_feat, y_train)

            # ---- EVALUATE ----
            y_pred = list(clf.predict(X_val_feat))
            y_true = list(y_val)

            all_true.extend(y_true)
            all_pred.extend(y_pred)

        return all_true, all_pred

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)

        # Save sklearn parts
        joblib.dump({
            "scaler": self.scaler,
            "pca": self.pca,
        }, os.path.join(path, "preprocessing.joblib"))

        # Save subclass-specific things
        self._save_extra(path)

    @classmethod
    def load(cls, path: str):
        obj = cls()

        data = joblib.load(os.path.join(path, "preprocessing.joblib"))
        obj.scaler = data["scaler"]
        obj.pca = data["pca"]

        obj._load_extra(path)
        return obj

    def _save_extra(self, path: str):
        pass

    def _load_extra(self, path: str):
        pass

