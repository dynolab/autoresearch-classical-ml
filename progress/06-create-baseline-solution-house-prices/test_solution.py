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

    model2 = Model()
    model2.load()

    df = pd.read_csv(train_csv_path)
    y_true = df["SalePrice"].values

    preds_after = model2.predict(train_csv_path)
    mse_after = np.mean((np.log(preds_after) - np.log(y_true)) ** 2)

    print(f"Train Log MSE: {mse_after:.2e}")


if __name__ == "__main__":
    test_solution("/Users/tony/datasets/kaggle/house_prices/for_autoresearch/train.csv")
