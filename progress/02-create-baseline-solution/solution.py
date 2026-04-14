import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os


class Model:
    def __init__(self):
        self.model = LinearRegression()
        self._fitted = False

    def fit(self, train_csv_path):
        df = pd.read_csv(train_csv_path)
        X = df[["x", "y"]].values
        y = df["target"].values
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, csv_path):
        df = pd.read_csv(csv_path)
        X = df[["x", "y"]].values
        return self.model.predict(X)

    def save(self, path="model.bin"):
        joblib.dump(self.model, path, compress=3)

    def load(self, path="model.bin"):
        self.model = joblib.load(path)
        self._fitted = True
        return self


def self_test(train_csv_path):
    model = Model()
    model.fit(train_csv_path)
    model.save()

    preds_before = model.predict(train_csv_path)
    df = pd.read_csv(train_csv_path)
    y_true = df["target"].values
    mse_before = np.mean((preds_before - y_true) ** 2)

    model2 = Model()
    model2.load()
    preds_after = model2.predict(train_csv_path)
    mse_after = np.mean((preds_after - y_true) ** 2)

    assert np.allclose(mse_before, mse_after), f"MSE mismatch: {mse_before} vs {mse_after}"

    size_bytes = os.path.getsize("model.bin")
    print(f"Train MSE: {mse_before:.6f}")
    print(f"Model size: {size_bytes} bytes")


if __name__ == "__main__":
    self_test("/Users/tony/datasets/three_peaks/train_dataset.csv")