import numpy as np
import pandas as pd
import struct
import os


class Model:
    def __init__(self):
        self.params = None
        self._fitted = False

    def fit(self, train_csv_path):
        df = pd.read_csv(train_csv_path)
        X = df[["x", "y"]].values
        y = df["target"].values

        from scipy.optimize import curve_fit

        def model_fn(xy, a0, a1, mu, inv_sx_sq, inv_sy_sq):
            x = xy[:, 0]
            y_col = xy[:, 1]
            return np.exp(-inv_sy_sq * y_col**2) * (
                a0 * np.exp(-inv_sx_sq * x**2)
                + a1 * np.exp(-inv_sx_sq * (x - mu) ** 2)
                + a1 * np.exp(-inv_sx_sq * (x + mu) ** 2)
            )

        p0 = [1.0, 0.4, 1.0, 25.0, 0.5]
        popt, _ = curve_fit(model_fn, X, y, p0=p0, maxfev=50000)
        self.params = {
            "a0": popt[0],
            "a1": popt[1],
            "mu": popt[2],
            "inv_sx_sq": popt[3],
            "inv_sy_sq": popt[4],
        }
        self._fitted = True
        return self

    def predict(self, csv_path):
        df = pd.read_csv(csv_path)
        x = df["x"].values
        y_col = df["y"].values
        p = self.params
        return np.exp(-p["inv_sy_sq"] * y_col**2) * (
            p["a0"] * np.exp(-p["inv_sx_sq"] * x**2)
            + p["a1"] * np.exp(-p["inv_sx_sq"] * (x - p["mu"]) ** 2)
            + p["a1"] * np.exp(-p["inv_sx_sq"] * (x + p["mu"]) ** 2)
        )

    def save(self, path="model.bin"):
        p = self.params
        data = struct.pack(
            "ddddd", p["a0"], p["a1"], p["mu"], p["inv_sx_sq"], p["inv_sy_sq"]
        )
        with open(path, "wb") as f:
            f.write(data)

    def load(self, path="model.bin"):
        with open(path, "rb") as f:
            data = f.read()
        vals = struct.unpack("ddddd", data)
        self.params = {
            "a0": vals[0],
            "a1": vals[1],
            "mu": vals[2],
            "inv_sx_sq": vals[3],
            "inv_sy_sq": vals[4],
        }
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
