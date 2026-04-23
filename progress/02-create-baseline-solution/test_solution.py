import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from solution import Model


def test_solution(train_csv_path):
    model = Model()
    model.fit(train_csv_path)
    model.save()

    df = pd.read_csv(train_csv_path)
    y_true = df["target"].values

    preds_before = model.predict(train_csv_path)
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
    test_solution("/Users/tony/datasets/abc/train_dataset.csv")
