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

    dataset_te = Dataset()  # testing data
    dataset_tr = Dataset()  # training data
    dataset_all = Dataset()  # all (train + test + unlabeled)

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

        dataset_te.add_file(full_pth, label)
        dataset_all.add_file(full_pth, label)

    for pth in os.listdir(os.path.join(ROOT, "Nothing")):
        if not pth.endswith(".csv"):
            continue
        full_pth = os.path.join(ROOT, "Nothing", pth)
        label = "nothing"

        dataset_all.add_file(full_pth, label)

    return dataset_tr, dataset_te, dataset_all

pipelines = {
    "sklearn": (WaveletsPipeline, {"n_components": 3, }),
    "keras": (EmbeddingsPipeline, {"n_components": 3,
                              "embedding_dim": 16,
                              "epochs": 10,
                              "batch_size": 32, }),
    "hybrid": (HybridPipeline, {"n_components": 3,
                                "embedding_dim": 16,
                                "epochs": 10,
                                "batch_size": 32,})

}

classifiers = {
    "centroid": (CentroidClassifier, {"threshold_scale": 1.3,}),
    "gaussian": (GaussianClassifier, {"threshold_scale": 1.2,}),
    "1cls_svm": (OneClassSVMClassifier, {"nu": 0.1,
                                         "gamma": "scale",}),
}

def main():
    results = {}

    dataset_tr, dataset_te, dataset_all = get_data()

    for p_name, (pipeline, p_kwargs) in pipelines.items():
        results[p_name] = {}

        for c_name, (clf, c_kwargs) in classifiers.items():
            y_tr, y_pr = pipeline.evaluate(
                dataset_labeled=dataset_tr,
                dataset_unlabeled=dataset_all,
                classifier=clf,
                n_splits=5,
                random_state=42,
                clf_kwargs=c_kwargs,
                **p_kwargs
            )

            results[p_name][c_name] = y_tr, y_pr

    for p_name in results:
        for c_name in results[p_name]:
            y_tr, y_pr = results[p_name][c_name]

            mask = np.array([y is not None for y in y_tr])

            y_tr_cls = np.array(y_tr)[mask]
            y_pr_cls = np.array(y_pr)[mask]
            y_pr_cls = np.array([y if y is not None else "None" for y in y_pr_cls])

            y_tr_none = [y if y is not None else "None" for y in y_tr]
            y_pr_none = [y if y is not None else "None" for y in y_pr]

            print(p_name, c_name)
            print("Classes only")
            print(classification_report(y_tr_cls, y_pr_cls, zero_division=0))
            print("\nAll inclusive")
            print(classification_report(y_tr_none, y_pr_none, zero_division=0))

if __name__ == "__main__":
    main()
