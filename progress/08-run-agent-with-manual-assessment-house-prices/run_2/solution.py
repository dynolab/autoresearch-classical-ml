import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping as lgbm_early_stopping
from catboost import CatBoostRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import joblib
import os

NUMERIC_FILL_STRATEGIES = {
    "LotFrontage": "median",
    "MasVnrArea": 0,
    "GarageYrBlt": 0,
}

CAT_FILL_VALUES = {
    "PoolQC": "None",
    "MiscFeature": "None",
    "Alley": "None",
    "Fence": "None",
    "MasVnrType": "None",
    "FireplaceQu": "None",
    "GarageType": "None",
    "GarageFinish": "None",
    "GarageQual": "None",
    "GarageCond": "None",
    "BsmtQual": "None",
    "BsmtCond": "None",
    "BsmtExposure": "None",
    "BsmtFinType1": "None",
    "BsmtFinType2": "None",
    "Electrical": "SBrkr",
}

DROP_COLUMNS = {"Id", "PoolQC", "MiscFeature", "Alley", "Fence"}

QUALITY_ORDINAL = {
    "Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0,
}

ORDINAL_COLUMNS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
    "HeatingQC", "KitchenQual", "FireplaceQu",
    "GarageQual", "GarageCond", "PoolQC",
]

BSMT_FIN_ORDINAL = {
    "GLQ": 6, "ALQ": 5, "BLQ": 4, "Rec": 3, "LwQ": 2, "Unf": 1, "None": 0,
}

GARAGE_FINISH_ORDINAL = {
    "Fin": 3, "RFn": 2, "Unf": 1, "None": 0,
}

BSMT_EXPOSURE_ORDINAL = {
    "Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0,
}

TARGET_ENCODE_COLUMNS = [
    "Neighborhood", "Exterior1st", "Exterior2nd",
    "MSSubClass", "MSZoning", "SaleType", "SaleCondition",
    "Condition1", "Condition2", "HouseStyle",
]

N_FOLDS = 10

TE_SMOOTHING_MIN_SAMPLES = 10


def _make_xgb(early_stopping=False):
    params = dict(
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=3,
        min_child_weight=6,
        subsample=0.7,
        colsample_bytree=0.5,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
    )
    if early_stopping:
        params["early_stopping_rounds"] = 50
    return XGBRegressor(**params)


def _make_lgbm():
    return LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=4,
        num_leaves=15,
        min_child_samples=5,
        subsample=0.7,
        colsample_bytree=0.5,
        reg_alpha=0.5,
        reg_lambda=1.5,
        random_state=42,
        verbose=-1,
    )


def _make_catboost():
    return CatBoostRegressor(
        iterations=3000,
        learning_rate=0.01,
        depth=4,
        l2_leaf_reg=3.0,
        subsample=0.7,
        colsample_bylevel=0.5,
        random_seed=42,
        verbose=0,
    )


def _make_en():
    return ElasticNet(
        alpha=0.001,
        l1_ratio=0.5,
        max_iter=10000,
        random_state=42,
    )


class Model:
    def __init__(self):
        self._base_models = None
        self._meta_weights = None
        self._fitted = False
        self._feature_columns = None
        self._cat_columns = None
        self._numeric_medians = None
        self._target_encode_maps = None
        self._global_mean = None
        self._scaler = None

    def _apply_ordinals(self, X):
        for col in ORDINAL_COLUMNS:
            if col in X.columns:
                X[col] = X[col].map(QUALITY_ORDINAL).fillna(0).astype(int)
        if "BsmtFinType1" in X.columns:
            X["BsmtFinType1"] = X["BsmtFinType1"].map(BSMT_FIN_ORDINAL).fillna(0).astype(int)
        if "BsmtFinType2" in X.columns:
            X["BsmtFinType2"] = X["BsmtFinType2"].map(BSMT_FIN_ORDINAL).fillna(0).astype(int)
        if "GarageFinish" in X.columns:
            X["GarageFinish"] = X["GarageFinish"].map(GARAGE_FINISH_ORDINAL).fillna(0).astype(int)
        if "BsmtExposure" in X.columns:
            X["BsmtExposure"] = X["BsmtExposure"].map(BSMT_EXPOSURE_ORDINAL).fillna(0).astype(int)
        return X

    def _engineer_features(self, X):
        X["TotalSF"] = X.get("GrLivArea", 0) + X.get("TotalBsmtSF", 0)
        X["TotalBath"] = X.get("FullBath", 0) + 0.5 * X.get("HalfBath", 0) + X.get("BsmtFullBath", 0) + 0.5 * X.get("BsmtHalfBath", 0)
        X["HouseAge"] = X.get("YrSold", 0) - X.get("YearBuilt", 0)
        X["RemodelAge"] = X.get("YrSold", 0) - X.get("YearRemodAdd", 0)
        X["IsRemodeled"] = (X.get("YearRemodAdd", 0) != X.get("YearBuilt", 0)).astype(int)
        X["TotalPorchSF"] = X.get("WoodDeckSF", 0) + X.get("OpenPorchSF", 0) + X.get("EnclosedPorch", 0) + X.get("ScreenPorch", 0) + X.get("3SsnPorch", 0)
        X["HasGarage"] = (X.get("GarageArea", 0) > 0).astype(int)
        X["HasBsmt"] = (X.get("TotalBsmtSF", 0) > 0).astype(int)
        X["HasFireplace"] = (X.get("Fireplaces", 0) > 0).astype(int)
        X["Has2ndFlr"] = (X.get("2ndFlrSF", 0) > 0).astype(int)
        X["HasPool"] = (X.get("PoolArea", 0) > 0).astype(int)
        X["OverallScore"] = X.get("OverallQual", 0) * X.get("OverallCond", 0)
        X["LivingQualSF"] = X.get("GrLivArea", 0) * X.get("OverallQual", 0)
        return X

    def _apply_target_encoding_train(self, X, y):
        self._target_encode_maps = {}
        self._global_mean = np.mean(y)
        for col in TARGET_ENCODE_COLUMNS:
            if col not in X.columns:
                continue
            mapping = X[[col]].copy()
            mapping["_y"] = y
            stats = mapping.groupby(col)["_y"].agg(["mean", "count"])
            smoothing = 1.0 / (1.0 + np.exp(-(stats["count"] - 1) / TE_SMOOTHING_MIN_SAMPLES))
            smoothed = self._global_mean * (1 - smoothing) + stats["mean"] * smoothing
            self._target_encode_maps[col] = smoothed.to_dict()
            X[col + "_TE"] = X[col].map(smoothed).fillna(self._global_mean)
        return X

    def _apply_target_encoding_test(self, X):
        for col in TARGET_ENCODE_COLUMNS:
            if col not in X.columns or col not in self._target_encode_maps:
                continue
            col_means = self._target_encode_maps[col]
            X[col + "_TE"] = X[col].map(col_means).fillna(self._global_mean)
        return X

    def _preprocess(self, df, is_train=False, y=None):
        X = df.copy()
        if "SalePrice" in X.columns:
            X = X.drop(columns=["SalePrice"])

        for col in list(DROP_COLUMNS):
            if col in X.columns:
                X = X.drop(columns=[col])

        for col, strategy in NUMERIC_FILL_STRATEGIES.items():
            if col in X.columns:
                if strategy == "median":
                    if is_train:
                        self._numeric_medians[col] = X[col].median()
                    X[col] = X[col].fillna(self._numeric_medians.get(col, 0))
                else:
                    X[col] = X[col].fillna(strategy)

        for col, val in CAT_FILL_VALUES.items():
            if col in X.columns:
                X[col] = X[col].fillna(val)

        remaining_num = X.select_dtypes(include=["int64", "float64"]).columns
        for col in remaining_num:
            if X[col].isnull().any():
                if is_train:
                    self._numeric_medians[col] = X[col].median()
                X[col] = X[col].fillna(self._numeric_medians.get(col, 0))

        remaining_cat = X.select_dtypes(include=["object"]).columns
        for col in remaining_cat:
            X[col] = X[col].fillna("None")

        X = self._apply_ordinals(X)
        X = self._engineer_features(X)

        if is_train and y is not None:
            X = self._apply_target_encoding_train(X, y)
        else:
            X = self._apply_target_encoding_test(X)

        for col in TARGET_ENCODE_COLUMNS:
            if col in X.columns:
                X = X.drop(columns=[col])

        self._cat_columns = X.select_dtypes(include=["object"]).columns.tolist()
        X = pd.get_dummies(X, columns=self._cat_columns, dtype=int)
        return X

    def _remove_outliers(self, df):
        mask = (df["GrLivArea"] <= 4000) | (np.log(df["SalePrice"]) > 12.5)
        return df[mask].reset_index(drop=True)

    def fit(self, train_csv_path):
        df = pd.read_csv(train_csv_path)
        df = self._remove_outliers(df)
        self._numeric_medians = {}
        y = np.log(df["SalePrice"].values)
        X = self._preprocess(df, is_train=True, y=y)
        self._feature_columns = X.columns.tolist()

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X.values)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        n_models = 4
        meta_features_train = np.zeros((X.shape[0], n_models))

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            X_tr_s, X_val_s = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            xgb_fold = _make_xgb(early_stopping=True)
            xgb_fold.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            meta_features_train[val_idx, 0] = xgb_fold.predict(X_val)

            lgbm_fold = _make_lgbm()
            lgbm_fold.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgbm_early_stopping(50, verbose=False)],
            )
            meta_features_train[val_idx, 1] = lgbm_fold.predict(X_val)

            cat_fold = _make_catboost()
            cat_fold.fit(
                X_tr, y_tr,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=0,
            )
            meta_features_train[val_idx, 2] = cat_fold.predict(X_val)

            en_fold = _make_en()
            en_fold.fit(X_tr_s, y_tr)
            meta_features_train[val_idx, 3] = en_fold.predict(X_val_s)

        fold_scores = []
        for i in range(n_models):
            score = np.mean((meta_features_train[:, i] - y) ** 2)
            fold_scores.append(score)

        xgb = _make_xgb()
        xgb.fit(X, y, verbose=False)

        lgbm = _make_lgbm()
        lgbm.fit(X, y)

        cat = _make_catboost()
        cat.fit(X, y, verbose=0)

        en = _make_en()
        en.fit(X_scaled, y)

        self._base_models = [xgb, lgbm, cat, en]

        inv_scores = [1.0 / s for s in fold_scores]
        total = sum(inv_scores)
        self._meta_weights = np.array([w / total for w in inv_scores])

        self._fitted = True

        print(f"OOF scores per model: {[f'{s:.6f}' for s in fold_scores]}")
        print(f"Meta weights: {self._meta_weights}")

        return self

    def predict(self, csv_path):
        df = pd.read_csv(csv_path)
        X = self._preprocess(df, is_train=False)
        X = X.reindex(columns=self._feature_columns, fill_value=0)

        X_scaled = self._scaler.transform(X.values)

        meta_features_test = np.column_stack([
            self._base_models[0].predict(X),
            self._base_models[1].predict(X),
            self._base_models[2].predict(X),
            self._base_models[3].predict(X_scaled),
        ])
        weighted_pred = meta_features_test @ self._meta_weights
        return np.exp(weighted_pred)

    def save(self, path="model.bin"):
        joblib.dump({
            "base_models": self._base_models,
            "meta_weights": self._meta_weights,
            "feature_columns": self._feature_columns,
            "cat_columns": self._cat_columns,
            "numeric_medians": self._numeric_medians,
            "target_encode_maps": self._target_encode_maps,
            "global_mean": self._global_mean,
            "scaler": self._scaler,
        }, path, compress=3)

    def load(self, path="model.bin"):
        data = joblib.load(path)
        self._base_models = data["base_models"]
        self._meta_weights = data["meta_weights"]
        self._feature_columns = data["feature_columns"]
        self._cat_columns = data["cat_columns"]
        self._numeric_medians = data["numeric_medians"]
        self._target_encode_maps = data["target_encode_maps"]
        self._global_mean = data["global_mean"]
        self._scaler = data["scaler"]
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
