import pywt
import numpy as np

from typing import Optional, Any
from scipy.signal import savgol_filter, resample

from .handle_files import read_csv


def segment(mag):
    threshold = np.mean(mag) + 0.5 * np.std(mag)
    mask = mag > threshold

    segments = []
    current_segment = np.array([])
    for i in range(len(mag)):
        if mask[i]:
            current_segment = np.append(current_segment, i)
        else:
            if current_segment.size > 0:
                segments.append(current_segment)
                current_segment = np.array([])

    if current_segment.size > 0:
        segments.append(current_segment)

    lengths = np.array([len(seg) for seg in segments])
    max_index = np.argmax(lengths)

    seg_mask = np.array(segments[max_index], dtype=int)

    return seg_mask


def augment_within_class(X, y, noise_std=0.01, augment_factor=1):
    X_aug = [X]
    y_aug = [y]

    unique_labels = np.unique(y)

    for label in unique_labels:
        idx = np.where(y == label)[0]
        X_class = X[idx]

        for _ in range(augment_factor):
            noise = np.random.normal(0, noise_std, X_class.shape)
            X_new = X_class + noise

            X_aug.append(X_new)
            y_aug.append(np.full(len(idx), label))

    return np.vstack(X_aug), np.concatenate(y_aug)


class Dataset:
    def __init__(self, target_length: int=100, augment: bool=False,
                 noise_std: float=0.01, augment_factor: int=1):
        # [(?, 6)]
        self.X_raw: list[np.ndarray] = []
        self.y: list = []
        self.target_length = target_length

        # [(tl, 100)], []
        self._X_cached: Optional[tuple[list[np.ndarray], list]] = None

        # augmentation settings
        self.augment = augment
        self.noise_std = noise_std
        self.augment_factor = augment_factor

    def add_file(self, file_name: str, file_label: str) -> None:
        x = read_csv(file_name)
        self.add_sample(x, file_label)

    def add_sample(self, x: np.ndarray, label: Any) -> None:
        self.X_raw.append(x)
        self.y.append(label)
        self._X_cached = None

    def preprocess(self) -> Optional[tuple[list[np.ndarray], list]]:
        if self._X_cached is None:
            X_pre = []

            for X in self.X_raw:
                # X: [(?, 6)]
                mag = np.linalg.norm(X, axis=1)

                # Segment to get the most relevant part of the data
                seg_mask = segment(mag)
                X = X[seg_mask]
                mag = mag[seg_mask]

                # Smooth the data using Savitzky-Golay filter
                X = savgol_filter(X, window_length=min(7, X.shape[0]),
                                  polyorder=2, axis=0)

                # Add magnitude in for resampling  --  X: (?, 7)
                X = np.hstack((X, mag.reshape(-1, 1)))

                # Time normalization
                # X: (100, 7)
                X = resample(X, 100, axis=0)

                # Remove magnitude before normalization  --  X: (100, 6)
                mag = X[:, -1]
                X = X[:, :-1]

                # Normalize the data
                X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

                # Add magnitude  --  X: (100, 7)
                X = np.hstack((X, mag.reshape(-1, 1)))

                X_pre.append(X)

            X_pre = np.array(X_pre)
            y = np.array(self.y, dtype=object)

            if self.augment:
                mask = y != None
                X_labeled = X_pre[mask]
                y_labeled = y[mask]

                X_aug, y_aug = self._augment_within_class(X_labeled, y_labeled)

                # optionally combine with original unlabeled
                X_pre = np.vstack([X_aug, X_pre[~mask]])
                y = np.concatenate([y_aug, y[~mask]])

            self._X_cached = X_pre, y

        return self._X_cached

    def _augment_within_class(self, X: list[np.ndarray], y: list
                              ) -> tuple[np.ndarray, np.ndarray]:
        X_aug = [X]
        y_aug = [y]

        y = np.array(y, dtype=object)

        for label in np.unique(y):
            if label is None:
                continue

            idx = np.where(y == label)[0]
            X_class = X[idx]

            for _ in range(self.augment_factor):
                # sample within class
                ref_idx = np.random.choice(len(X_class), len(X_class))
                X_ref = X_class[ref_idx]

                noise = np.random.normal(0, self.noise_std, X_class.shape)

                # combine + noise
                X_new = 0.5 * X_class + 0.5 * X_ref + noise

                X_aug.append(X_new)
                y_aug.append(np.full(len(idx), label))

        return np.vstack(X_aug), np.concatenate(y_aug)

    @staticmethod
    def _get_wavelet_feats(X, wavelet: str='db4', level: int=3,
                           drop_cD1: bool=False) -> np.ndarray:
        w = pywt.Wavelet(wavelet)

        feats = []
        for ch in range(X.shape[1]):  # loop over 7 channels
            signal = X[:, ch]

            max_level = pywt.dwt_max_level(len(signal), w.dec_len)
            lvl = min(level if level is not None else max_level, max_level)

            coeffs = pywt.wavedec(signal, wavelet=wavelet, level=lvl)

            # coeffs = [cA_n, cD_n, ..., cD_1]
            for c in coeffs:
                feats.append(c)

            if drop_cD1:
                feats = feats[:-1]

        # flatten everything into 1 vector
        return np.concatenate(feats)

    def get_wavelet_features(self, wavelet: str='db4',
                             level: int=3, drop_cD1: bool=False
                             ) -> tuple[np.ndarray, list]:
        # X_proc: [N * (100, 7)]; y: [N]
        X_proc, y = self.preprocess()

        X_wave = np.array([
            self._get_wavelet_feats(X,
                                    wavelet=wavelet,
                                    level=level,
                                    drop_cD1=drop_cD1)
            for X in X_proc
        ])

        # X_wave: (N, d)
        return X_wave, y

    def subset(self, indices, augment: bool=False):
        new_dataset = Dataset(target_length=self.target_length,
                              augment=augment,
                              augment_factor=self.augment_factor,
                              noise_std=self.noise_std)
        for i in indices:
            X = self.X_raw[i]
            y = self.y[i]
            new_dataset.add_sample(X, y)

        return new_dataset


