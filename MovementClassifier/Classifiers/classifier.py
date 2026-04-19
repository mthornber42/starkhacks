from abc import ABC, abstractmethod
import numpy as np
import json
import os

class GenericClassifier(ABC):

    @abstractmethod
    def fit(self, Z: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str):
        raise NotImplementedError
