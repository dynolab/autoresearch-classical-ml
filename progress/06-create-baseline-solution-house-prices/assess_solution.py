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
    y_true = df["SalePrice"].values

    y_pred = model.predict(test_csv_path)
    mse = np.mean((np.log(y_pred) - np.log(y_true)) ** 2)

    print(f"Test Log MSE: {mse:.2e}")


if __name__ == "__main__":
    assess_solution("/Users/tony/datasets/kaggle/house_prices/for_autoresearch/test.csv")
