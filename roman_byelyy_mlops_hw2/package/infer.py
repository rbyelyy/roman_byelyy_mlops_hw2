# infer.py
import pickle

import pandas as pd
from sklearn.datasets import load_iris


class InferModel:
    def __init__(self, model_file):
        with open(model_file, "rb") as f:
            self.model = pickle.load(f)
        iris = load_iris()
        self.data = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    def predict(self):
        predictions = self.model.predict(self.data)
        return predictions

    def save_predictions(self, predictions):
        pd.DataFrame(predictions).to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    infer = InferModel("model.pkl")
    predictions = infer.predict()
    infer.save_predictions(predictions)
