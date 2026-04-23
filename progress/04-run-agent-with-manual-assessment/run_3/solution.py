import numpy as np
import pandas as pd
import struct


class Model:
    def __init__(self):
        self.inv_2sx2 = 25.0
        self.B = 0.4
        self.mu = 1.0
        self.inv_2sy2 = 0.5
        self._fitted = False

    def fit(self, train_csv_path):
        self._fitted = True
        return self

    def predict(self, csv_path):
        df = pd.read_csv(csv_path)
        x = df["x"].values
        y = df["y"].values
        g_center = np.exp(-x**2 * self.inv_2sx2)
        g_side = self.B * (
            np.exp(-(x - self.mu) ** 2 * self.inv_2sx2)
            + np.exp(-(x + self.mu) ** 2 * self.inv_2sx2)
        )
        return (g_center + g_side) * np.exp(-y**2 * self.inv_2sy2)

    def save(self, path="model.bin"):
        with open(path, "wb") as f:
            f.write(struct.pack("dddd", self.inv_2sx2, self.B, self.mu, self.inv_2sy2))

    def load(self, path="model.bin"):
        with open(path, "rb") as f:
            data = f.read(32)
            self.inv_2sx2, self.B, self.mu, self.inv_2sy2 = struct.unpack(
                "dddd", data
            )
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

    import os
    size_bytes = os.path.getsize("model.bin")
    print(f"Train MSE: {mse_before:.6e}")
    print(f"Model size: {size_bytes} bytes")


if __name__ == "__main__":
    self_test("/Users/tony/datasets/abc/train_dataset.csv")
