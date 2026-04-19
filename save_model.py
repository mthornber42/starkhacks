from MovementClassifier.dataset import Dataset
from MovementClassifier.Pipelines import *
from MovementClassifier.Classifiers import *
import os
import numpy as np
from sklearn.metrics import classification_report

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

ROOT = "AllData"

def get_data():

    dataset_tr = Dataset(augment=True)  # training data
    dataset_all = Dataset()  # all (train + unlabeled)

    for pth in os.listdir(os.path.join(ROOT, "Train")):
        if not pth.endswith(".csv"):
            continue
        full_pth = os.path.join(ROOT, "Train", pth)
        label = pth.split("_")[0]

        dataset_tr.add_file(full_pth, label)
        dataset_all.add_file(full_pth, label)

    for pth in os.listdir(os.path.join(ROOT, "Test")):
        if not pth.endswith(".csv"):
            continue
        full_pth = os.path.join(ROOT, "Test", pth)
        label = pth.split("_")[0]

        dataset_tr.add_file(full_pth, label)
        dataset_all.add_file(full_pth, label)

    for pth in os.listdir(os.path.join(ROOT, "Nothing")):
        if not pth.endswith(".csv"):
            continue
        full_pth = os.path.join(ROOT, "Nothing", pth)
        label = "nothing"

        dataset_all.add_file(full_pth, label)

    return dataset_tr, dataset_all

pipelines = {
    "sklearn2": (WaveletsPipeline, {"n_components": 6, }),
}

classifiers = {
    "centroid": (CentroidClassifier, {"threshold_scale": 1.3,}),
}

def main():
    dataset_tr, dataset_all = get_data()

    pipeline = WaveletsPipeline(n_components=6)
    classifier = CentroidClassifier(threshold_scale=1.3)

    # UNSUPERVISED LEARNING
    pipeline.fit(dataset_all)

    # SUPERVISED LEARNING
    X_tr = pipeline.transform(dataset_tr)
    y_tr = np.array(dataset_tr._X_cached[1])

    # TRAIN CLASSIFIER
    classifier.fit(X_tr, y_tr)

    # SAVE MODELS
    pipeline.save("pipeline")
    classifier.save("classifier")


if __name__ == "__main__":
    main()
