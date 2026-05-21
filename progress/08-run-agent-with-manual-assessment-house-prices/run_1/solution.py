import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import ElasticNet, Ridge, RidgeCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, StratifiedKFold
import joblib
import os

ORDINAL_MAPS = {
    "ExterQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "ExterCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtQual": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtCond": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "BsmtExposure": {"NA": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "HeatingQC": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "KitchenQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "FireplaceQu": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageQual": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "GarageCond": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
    "PoolQC": {"NA": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
    "BsmtFinType1": {"NA": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"NA": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"NA": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Fence": {"NA": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
    "LotShape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
    "LandSlope": {"Sev": 1, "Mod": 2, "Gtl": 3},
    "Functional": {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8},
    "Street": {"Grvl": 1, "Pave": 2},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "Utilities": {"ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4},
    "CentralAir": {"N": 0, "Y": 1},
}

NUMERIC_FILL = {
    "LotFrontage": 0,
    "MasVnrArea": 0,
    "GarageYrBlt": 0,
    "BsmtFinSF1": 0,
    "BsmtFinSF2": 0,
    "BsmtUnfSF": 0,
    "TotalBsmtSF": 0,
    "BsmtFullBath": 0,
    "BsmtHalfBath": 0,
    "GarageCars": 0,
    "GarageArea": 0,
}

CAT_FILL_NA = [
    "Alley", "MasVnrType", "FireplaceQu", "PoolQC", "Fence",
    "MiscFeature", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
]

SKEWED_COLS = [
    "LotArea", "LotFrontage", "MasVnrArea", "BsmtFinSF1", "BsmtUnfSF",
    "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea", "GarageArea",
    "WoodDeckSF", "OpenPorchSF", "TotalSF", "EnclosedPorch",
    "ScreenPorch", "TotalPorchSF",
]

TE_COLS = ["Neighborhood", "MSSubClass", "Exterior1st", "Exterior2nd",
           "Condition1", "Condition2", "SaleType", "SaleCondition",
           "HouseStyle", "BldgType", "RoofMatl", "Functional",
           "MSZoning", "Foundation", "GarageType", "LotConfig",
           "RoofStyle", "Heating", "Electrical"]

N_SEEDS_XGB = 5
N_SEEDS_LGB = 5
N_SEEDS_GBR = 3
N_SEEDS_CAT = 3
N_SEEDS_KRR = 3
N_FOLDS = 5
TE_SMOOTH = 10


class Model:
    def __init__(self):
        self._fitted = False
        self._feature_columns = None
        self._label_encoders = {}
        self._xgb_models = []
        self._lgb_models = []
        self._gbr_models = []
        self._cat_models = []
        self._krr_models = []
        self._enet_models = []
        self._meta_model = None
        self._te_maps = {}
        self._te_global_mean = 0.0
        self._cat_features = []
        self._nbh_agg = None
        self._meta_model = None

    @staticmethod
    def _build_meta_features(base_preds):
        n = base_preds.shape[1]
        feats = [base_preds]
        feats.append(base_preds.mean(axis=1).reshape(-1, 1))
        feats.append(base_preds.std(axis=1).reshape(-1, 1))
        feats.append(base_preds.max(axis=1).reshape(-1, 1))
        feats.append(base_preds.min(axis=1).reshape(-1, 1))
        feats.append((base_preds.max(axis=1) - base_preds.min(axis=1)).reshape(-1, 1))
        med = np.median(base_preds, axis=1).reshape(-1, 1)
        feats.append(med)
        return np.hstack(feats)

    def _preprocess(self, df, is_train=False, y=None):
        df = df.drop(columns=["Id"], errors="ignore")

        if "Neighborhood" in df.columns and "LotFrontage" in df.columns:
            lf_medians = df.groupby("Neighborhood")["LotFrontage"].median()
            df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
                lambda x: x.fillna(lf_medians.get(x.name, 0))
            )
            df["LotFrontage"] = df["LotFrontage"].fillna(0)

        for col, fill_val in NUMERIC_FILL.items():
            if col in df.columns and col != "LotFrontage":
                df[col] = df[col].fillna(fill_val)

        for col in CAT_FILL_NA:
            if col in df.columns:
                df[col] = df[col].fillna("NA")

        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].fillna("None")

        for col, mapping in ORDINAL_MAPS.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0).astype(int)

        df["MSSubClass"] = df["MSSubClass"].astype(str)

        df["TotalSF"] = df.get("TotalBsmtSF", 0) + df.get("1stFlrSF", 0) + df.get("2ndFlrSF", 0)
        df["TotalBath"] = (
            df.get("FullBath", 0)
            + 0.5 * df.get("HalfBath", 0)
            + df.get("BsmtFullBath", 0)
            + 0.5 * df.get("BsmtHalfBath", 0)
        )
        df["TotalPorchSF"] = (
            df.get("OpenPorchSF", 0)
            + df.get("EnclosedPorch", 0)
            + df.get("3SsnPorch", 0)
            + df.get("ScreenPorch", 0)
            + df.get("WoodDeckSF", 0)
        )
        df["HasGarage"] = (df.get("GarageArea", 0) > 0).astype(int)
        df["HasBsmt"] = (df.get("TotalBsmtSF", 0) > 0).astype(int)
        df["HasFireplace"] = (df.get("Fireplaces", 0) > 0).astype(int)
        df["HasPool"] = (df.get("PoolArea", 0) > 0).astype(int)
        df["Has2ndFlr"] = (df.get("2ndFlrSF", 0) > 0).astype(int)
        df["HasMasVnr"] = (df.get("MasVnrArea", 0) > 0).astype(int)
        df["HasWoodDeck"] = (df.get("WoodDeckSF", 0) > 0).astype(int)
        df["HasLowQualFin"] = (df.get("LowQualFinSF", 0) > 0).astype(int)
        df["HasPoolOrDeck"] = ((df.get("PoolArea", 0) > 0) | (df.get("WoodDeckSF", 0) > 0)).astype(int)

        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
        df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
        df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]
        df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
        df["IsNew"] = (df["YrSold"] == df["YearBuilt"]).astype(int)
        df["YrSinceRemod"] = df["YearRemodAdd"] - df["YearBuilt"]

        df["OverallQual_GrLivArea"] = df["OverallQual"] * df["GrLivArea"]
        df["OverallQual_TotalSF"] = df["OverallQual"] * df["TotalSF"]
        df["GarageCars_Area"] = df["GarageCars"] * df["GarageArea"]
        df["OverallQual_sq"] = df["OverallQual"] ** 2
        df["OverallCond_Qual"] = df["OverallCond"] * df["OverallQual"]
        df["LivingAreaPerRoom"] = df["GrLivArea"] / (df["TotRmsAbvGrd"].replace(0, 1))
        df["OverallQual_HouseAge"] = df["OverallQual"] / (df["HouseAge"].replace(0, 1).clip(lower=1))
        df["GrLivArea_LotArea"] = df["GrLivArea"] * df.get("LotArea", 0)
        df["TotalBath_Qual"] = df["TotalBath"] * df["OverallQual"]

        df["OverallQual_GarageArea"] = df["OverallQual"] * df.get("GarageArea", 0)
        df["OverallQual_TotalBath"] = df["OverallQual"] * df["TotalBath"]
        df["OverallQual_sq_GrLivArea"] = df["OverallQual_sq"] * df["GrLivArea"]
        df["RemodAge_Qual"] = df["RemodAge"] * df["OverallQual"]
        df["KitchenQual_Num"] = df.get("KitchenQual", 0) * df["OverallQual"]
        df["ExterQual_Num"] = df.get("ExterQual", 0) * df["OverallQual"]
        df["TotalSF_log1p"] = np.log1p(df["TotalSF"].astype(float))
        df["GrLivArea_log1p"] = np.log1p(df["GrLivArea"].astype(float))

        df["Neighborhood_Qual"] = df["OverallQual"].astype(str) + "_" + df.get("Neighborhood", "None").astype(str)
        df["SeasonSold"] = df["MoSold"].map({1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4, 12: 1}).astype(int)
        df["OverallQual_cubed"] = df["OverallQual"] ** 3
        df["GrLivArea_sq"] = df["GrLivArea"] ** 2
        df["TotalSF_sq"] = df["TotalSF"] ** 2
        df["BsmtRatio"] = df.get("TotalBsmtSF", 0) / (df["TotalSF"].replace(0, 1))
        df["GarageRatio"] = df.get("GarageArea", 0) / (df["TotalSF"].replace(0, 1))
        df["PorchRatio"] = df["TotalPorchSF"] / (df["TotalSF"].replace(0, 1))
        df["OverallQual_BsmtQual"] = df["OverallQual"] * df.get("BsmtQual", 0)

        if is_train and y is not None:
            self._te_maps = {}
            self._te_global_mean = y.mean()
            for col in TE_COLS:
                if col in df.columns:
                    global_mean = y.mean()
                    tmp = pd.DataFrame({col: df[col].values, "target": y})
                    stats = tmp.groupby(col)["target"].agg(["mean", "count"])
                    smoothed = (stats["count"] * stats["mean"] + TE_SMOOTH * global_mean) / (stats["count"] + TE_SMOOTH)
                    self._te_maps[col] = smoothed.to_dict()

            kf_te = KFold(n_splits=N_FOLDS, shuffle=True, random_state=2024)
            for col in TE_COLS:
                if col in df.columns:
                    te_values = np.full(len(df), self._te_global_mean)
                    for te_train_idx, te_val_idx in kf_te.split(df):
                        fold_global_mean = y[te_train_idx].mean()
                        fold_col = df[col].values[te_train_idx]
                        tmp = pd.DataFrame({col: fold_col, "target": y[te_train_idx]})
                        stats = tmp.groupby(col)["target"].agg(["mean", "count"])
                        smoothed = (stats["count"] * stats["mean"] + TE_SMOOTH * fold_global_mean) / (stats["count"] + TE_SMOOTH)
                        te_map = smoothed.to_dict()
                        vals = df[col].values[te_val_idx]
                        te_values[te_val_idx] = pd.Series(vals).map(te_map).fillna(fold_global_mean).values
                    df[col + "_te"] = te_values
        else:
            for col in TE_COLS:
                if col in df.columns and col in self._te_maps:
                    te_map = self._te_maps[col]
                    df[col + "_te"] = df[col].map(te_map).fillna(self._te_global_mean)

        if "Neighborhood_te" in df.columns and "OverallQual" in df.columns:
            df["NeighborhoodQual_te"] = df["Neighborhood_te"] * df["OverallQual"]
        if "MSSubClass_te" in df.columns and "OverallQual" in df.columns:
            df["MSSubClassQual_te"] = df["MSSubClass_te"] * df["OverallQual"]
        if "HouseStyle_te" in df.columns and "OverallQual" in df.columns:
            df["HouseStyleQual_te"] = df["HouseStyle_te"] * df["OverallQual"]
        if "Exterior1st_te" in df.columns and "Exterior2nd_te" in df.columns:
            df["Ext1Ext2_te_avg"] = (df["Exterior1st_te"] + df["Exterior2nd_te"]) / 2
        if "MSZoning_te" in df.columns and "OverallQual" in df.columns:
            df["MSZoningQual_te"] = df["MSZoning_te"] * df["OverallQual"]
        if "Foundation_te" in df.columns and "OverallQual" in df.columns:
            df["FoundationQual_te"] = df["Foundation_te"] * df["OverallQual"]
        if "Neighborhood_te" in df.columns and "TotalSF" in df.columns:
            df["Neighborhood_te_TotalSF"] = df["Neighborhood_te"] * df["TotalSF"]
        if "GarageType_te" in df.columns and "OverallQual" in df.columns:
            df["GarageTypeQual_te"] = df["GarageType_te"] * df["OverallQual"]
        if "SaleCondition_te" in df.columns and "OverallQual" in df.columns:
            df["SaleConditionQual_te"] = df["SaleCondition_te"] * df["OverallQual"]
        if "Condition1_te" in df.columns and "OverallQual" in df.columns:
            df["Condition1Qual_te"] = df["Condition1_te"] * df["OverallQual"]

        if is_train:
            self._nbh_agg = df.groupby("Neighborhood").agg(
                NBH_MeanGrLivArea=("GrLivArea", "mean"),
                NBH_MeanLotArea=("LotArea", "mean"),
                NBH_MeanOverallQual=("OverallQual", "mean"),
                NBH_MeanYearBuilt=("YearBuilt", "mean"),
            ).to_dict("index")
        if self._nbh_agg is not None and "Neighborhood" in df.columns:
            nbh_feats = df["Neighborhood"].map(self._nbh_agg).fillna(pd.Series(self._nbh_agg.get("NAmes", {})))
            df["NBH_MeanGrLivArea"] = nbh_feats.apply(lambda x: x.get("NBH_MeanGrLivArea", 0) if isinstance(x, dict) else 0)
            df["NBH_MeanLotArea"] = nbh_feats.apply(lambda x: x.get("NBH_MeanLotArea", 0) if isinstance(x, dict) else 0)
            df["NBH_MeanOverallQual"] = nbh_feats.apply(lambda x: x.get("NBH_MeanOverallQual", 0) if isinstance(x, dict) else 0)
            df["NBH_MeanYearBuilt"] = nbh_feats.apply(lambda x: x.get("NBH_MeanYearBuilt", 0) if isinstance(x, dict) else 0)
            df["NBH_Qual_Diff"] = df["OverallQual"] - df["NBH_MeanOverallQual"]
            df["NBH_Area_Diff"] = df["GrLivArea"] - df["NBH_MeanGrLivArea"]

        for col in SKEWED_COLS:
            if col in df.columns:
                df[col + "_log1p"] = np.log1p(df[col].astype(float))

        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if is_train:
            self._cat_features = cat_cols
            self._label_encoders = {}
            for col in cat_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self._label_encoders[col] = le
        else:
            for col in cat_cols:
                if col in self._label_encoders:
                    le = self._label_encoders[col]
                    df[col] = df[col].astype(str).map(
                        lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
                    )

        return df

    def fit(self, train_csv_path):
        df = pd.read_csv(train_csv_path)
        y = np.log1p(df["SalePrice"].values)
        X = df.drop(columns=["SalePrice"])
        X = self._preprocess(X, is_train=True, y=y)

        mask = df["GrLivArea"].values <= 4000
        X_clean = X[mask].reset_index(drop=True)
        y_clean = y[mask]

        self._feature_columns = X_clean.columns.tolist()

        self._xgb_models = []
        self._lgb_models = []
        self._gbr_models = []
        self._cat_models = []
        self._krr_models = []
        self._enet_models = []

        oof_xgb = np.zeros(len(X_clean))
        oof_lgb = np.zeros(len(X_clean))
        oof_gbr = np.zeros(len(X_clean))
        oof_cat = np.zeros(len(X_clean))
        oof_krr = np.zeros(len(X_clean))
        oof_enet = np.zeros(len(X_clean))

        for seed in range(N_SEEDS_XGB):
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + seed)
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_clean)):
                X_tr, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
                y_tr, y_val = y_clean[train_idx], y_clean[val_idx]

                xgb = XGBRegressor(
                    n_estimators=5000,
                    learning_rate=0.005,
                    max_depth=4,
                    min_child_weight=5,
                    subsample=0.7,
                    colsample_bytree=0.6,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    gamma=0,
                    random_state=seed,
                    early_stopping_rounds=100,
                )
                xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                self._xgb_models.append(xgb)
                oof_xgb[val_idx] += xgb.predict(X_val)

        oof_xgb /= N_SEEDS_XGB

        for seed in range(N_SEEDS_LGB):
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=77 + seed)
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_clean)):
                X_tr, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
                y_tr, y_val = y_clean[train_idx], y_clean[val_idx]

                lgb = LGBMRegressor(
                    n_estimators=5000,
                    learning_rate=0.01,
                    max_depth=4,
                    num_leaves=15,
                    min_child_samples=10,
                    subsample=0.7,
                    colsample_bytree=0.6,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=seed,
                    verbose=-1,
                )
                lgb.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        __import__("lightgbm").early_stopping(100, verbose=False),
                        __import__("lightgbm").log_evaluation(0),
                    ],
                )
                self._lgb_models.append(lgb)
                oof_lgb[val_idx] += lgb.predict(X_val)

        oof_lgb /= N_SEEDS_LGB

        for seed in range(N_SEEDS_GBR):
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=123 + seed)
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_clean)):
                X_tr, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
                y_tr, y_val = y_clean[train_idx], y_clean[val_idx]

                gbr = GradientBoostingRegressor(
                    n_estimators=2000,
                    learning_rate=0.01,
                    max_depth=4,
                    min_samples_leaf=10,
                    subsample=0.7,
                    max_features=0.6,
                    n_iter_no_change=50,
                    validation_fraction=0.15,
                    random_state=seed,
                )
                gbr.fit(X_tr, y_tr)
                self._gbr_models.append(gbr)
                oof_gbr[val_idx] += gbr.predict(X_val)

        oof_gbr /= N_SEEDS_GBR

        cat_feature_indices = [X_clean.columns.get_loc(c) for c in self._cat_features if c in X_clean.columns]

        for seed in range(N_SEEDS_CAT):
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=200 + seed)
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_clean)):
                X_tr, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
                y_tr, y_val = y_clean[train_idx], y_clean[val_idx]

                cat = CatBoostRegressor(
                    iterations=5000,
                    learning_rate=0.01,
                    depth=4,
                    l2_leaf_reg=5.0,
                    subsample=0.7,
                    colsample_bylevel=0.6,
                    random_seed=seed,
                    early_stopping_rounds=100,
                    verbose=0,
                )
                cat.fit(
                    X_tr, y_tr,
                    eval_set=(X_val, y_val),
                    cat_features=cat_feature_indices,
                )
                self._cat_models.append(cat)
                oof_cat[val_idx] += cat.predict(X_val)

        oof_cat /= N_SEEDS_CAT

        for seed in range(N_SEEDS_KRR):
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=300 + seed)
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_clean)):
                X_tr, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
                y_tr, y_val = y_clean[train_idx], y_clean[val_idx]

                krr = Pipeline([
                    ("scaler", RobustScaler()),
                    ("krr", KernelRidge(alpha=0.6, kernel="polynomial", degree=2, coef0=2.5)),
                ])
                krr.fit(X_tr, y_tr)
                self._krr_models.append(krr)
                oof_krr[val_idx] += krr.predict(X_val)

        oof_krr /= N_SEEDS_KRR

        y_bins = pd.qcut(y_clean, q=10, duplicates="drop").astype(str)
        kf_enet = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        for train_idx, val_idx in kf_enet.split(X_clean, y_bins):
            X_tr, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
            y_tr = y_clean[train_idx]
            enet = Pipeline([
                ("scaler", StandardScaler()),
                ("enet", ElasticNet(alpha=0.0005, l1_ratio=0.5, max_iter=10000)),
            ])
            enet.fit(X_tr, y_tr)
            self._enet_models.append(enet)
            oof_enet[val_idx] = enet.predict(X_val)

        base_meta = np.column_stack([oof_xgb, oof_lgb, oof_gbr, oof_cat, oof_krr, oof_enet])
        meta_features = Model._build_meta_features(base_meta)

        self._meta_model = RidgeCV(alphas=np.logspace(-3, 3, 50), scoring="neg_mean_squared_error")
        self._meta_model.fit(meta_features, y_clean)

        self._fitted = True
        return self

    def predict(self, csv_path):
        df = pd.read_csv(csv_path)
        if "SalePrice" in df.columns:
            X = df.drop(columns=["SalePrice"])
        else:
            X = df
        X = self._preprocess(X, is_train=False)
        X = X.reindex(columns=self._feature_columns, fill_value=0)

        xgb_preds = np.zeros(len(X))
        for model in self._xgb_models:
            xgb_preds += model.predict(X)
        xgb_preds /= len(self._xgb_models)

        lgb_preds = np.zeros(len(X))
        for model in self._lgb_models:
            lgb_preds += model.predict(X)
        lgb_preds /= len(self._lgb_models)

        gbr_preds = np.zeros(len(X))
        for model in self._gbr_models:
            gbr_preds += model.predict(X)
        gbr_preds /= len(self._gbr_models)

        cat_preds = np.zeros(len(X))
        for model in self._cat_models:
            cat_preds += model.predict(X)
        cat_preds /= len(self._cat_models)

        krr_preds = np.zeros(len(X))
        for model in self._krr_models:
            krr_preds += model.predict(X)
        krr_preds /= len(self._krr_models)

        enet_preds = np.zeros(len(X))
        for model in self._enet_models:
            enet_preds += model.predict(X)
        enet_preds /= len(self._enet_models)

        base_meta = np.column_stack([xgb_preds, lgb_preds, gbr_preds, cat_preds, krr_preds, enet_preds])
        meta_features = Model._build_meta_features(base_meta)
        log_preds = self._meta_model.predict(meta_features)

        return np.expm1(log_preds)

    def save(self, path="model.bin"):
        data = {
            "feature_columns": self._feature_columns,
            "label_encoders": self._label_encoders,
            "xgb_models": self._xgb_models,
            "lgb_models": self._lgb_models,
            "gbr_models": self._gbr_models,
            "cat_models": self._cat_models,
            "krr_models": self._krr_models,
            "enet_models": self._enet_models,
            "te_maps": self._te_maps,
            "te_global_mean": self._te_global_mean,
            "cat_features": self._cat_features,
            "nbh_agg": self._nbh_agg,
            "meta_model": self._meta_model,
        }
        joblib.dump(data, path, compress=3)

    def load(self, path="model.bin"):
        data = joblib.load(path)
        self._feature_columns = data["feature_columns"]
        self._label_encoders = data["label_encoders"]
        self._xgb_models = data["xgb_models"]
        self._lgb_models = data["lgb_models"]
        self._gbr_models = data["gbr_models"]
        self._cat_models = data["cat_models"]
        self._krr_models = data["krr_models"]
        self._enet_models = data["enet_models"]
        self._te_maps = data["te_maps"]
        self._te_global_mean = data["te_global_mean"]
        self._cat_features = data["cat_features"]
        self._nbh_agg = data["nbh_agg"]
        self._meta_model = data["meta_model"]
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
