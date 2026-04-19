from MovementClassifier.dataset import Dataset
from MovementClassifier.Pipelines import WaveletsPipeline
from MovementClassifier.Classifiers import CentroidClassifier
import pandas as pd

import tensorflow as tf
tf.get_logger().setLevel("ERROR")


def predict(json_data):
    if "samples" in json_data:
        json_data = json_data["samples"]
    
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(json_data)

    # DataFrame -> numpy
    X = df.to_numpy()

    # Load pipeline & classifier
    pipeline = WaveletsPipeline.load("pipeline")
    classifier = CentroidClassifier.load("classifier")
    
    # Wrap data in dataset
    dataset = Dataset()
    dataset.add_sample(X, label="")

    X = pipeline.transform(dataset)
    y = classifier.predict(X)[0]
    
    return y
