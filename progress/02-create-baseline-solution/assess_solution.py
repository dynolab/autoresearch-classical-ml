import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from solution import Model


def assess_solution(test_csv_path):
    model = Model()
    model.load()

    df = pd.read_csv(test_csv_path)
    y_true = df["target"].values

    y_pred = model.predict(test_csv_path)
    mse = np.mean((y_pred - y_true) ** 2)

    print(f"Test MSE: {mse:.6f}")


if __name__ == "__main__":
    assess_solution("/Users/tony/datasets/three_peaks/test_dataset.csv")
