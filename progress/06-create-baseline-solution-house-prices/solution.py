import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import joblib
import os


class Model:
    def __init__(self):
        self.model = XGBRegressor(random_state=42)
        self._fitted = False
        self._feature_columns = None
        self._cat_columns = None

    def fit(self, train_csv_path):
        df = pd.read_csv(train_csv_path)
        X = df.drop(columns=["SalePrice"])
        self._cat_columns = X.select_dtypes(include=["object"]).columns.tolist()
        X = pd.get_dummies(X, columns=self._cat_columns)
        self._feature_columns = X.columns.tolist()
        y = df["SalePrice"].values
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, csv_path):
        df = pd.read_csv(csv_path)
        if "SalePrice" in df.columns:
            X = df.drop(columns=["SalePrice"])
        else:
            X = df
        X = pd.get_dummies(X, columns=self._cat_columns)
        X = X.reindex(columns=self._feature_columns, fill_value=0)
        return self.model.predict(X)

    def save(self, path="model.bin"):
        joblib.dump({
            "model": self.model,
            "feature_columns": self._feature_columns,
            "cat_columns": self._cat_columns
        }, path, compress=3)

    def load(self, path="model.bin"):
        data = joblib.load(path)
        self.model = data["model"]
        self._feature_columns = data["feature_columns"]
        self._cat_columns = data["cat_columns"]
        self._fitted = True
        return self


def self_test(train_csv_path):
    model = Model()
    model.fit(train_csv_path)
    model.save()

    preds_before = model.predict(train_csv_path)
    df = pd.read_csv(train_csv_path)
    y_true = df["SalePrice"].values
    mse_before = np.mean((np.log(preds_before) - np.log(y_true)) ** 2)

    model2 = Model()
    model2.load()
    preds_after = model2.predict(train_csv_path)
    mse_after = np.mean((np.log(preds_after) - np.log(y_true)) ** 2)

    assert np.allclose(mse_before, mse_after), f"MSE mismatch: {mse_before} vs {mse_after}"

    print(f"Train Log MSE: {mse_before:.2e}")


if __name__ == "__main__":
    self_test("/Users/tony/datasets/kaggle/house_prices/for_autoresearch/train.csv")
