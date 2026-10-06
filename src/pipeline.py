"""预处理 + 训练管道：澳洲降雨/气温预测。

方案基于 Nicolas 的最终 step4 迭代（保留高缺失列、交给 XGBoost 原生处理 NaN），
并移除了他代码里的跨数据集 imputation bug —— 干脆不做任何手动 imputation。

训练 4 个目标：
  - RainTomorrow      (J+1 是否下雨, 二分类)
  - RainInTwoDays     (J+2 是否下雨, 二分类)
  - MaxTempTomorrow   (J+1 最高温, 回归)
  - MaxTempInTwoDays  (J+2 最高温, 回归)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "weatherAUS.csv"
MODEL_DIR = BASE_DIR / "models"

TARGETS = ["RainTomorrow", "RainInTwoDays", "MaxTempTomorrow", "MaxTempInTwoDays"]
CLASS_TARGETS = ["RainTomorrow", "RainInTwoDays"]
REG_TARGETS = ["MaxTempTomorrow", "MaxTempInTwoDays"]

WIND_DIRS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S",
    "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]
SEASONS = ["Summer", "Autumn", "Winter", "Spring"]

RANDOM_STATE = 42
TEST_SIZE = 0.2


def get_season_sh(month: int) -> str:
    if month in (12, 1, 2):
        return "Summer"
    if month in (3, 4, 5):
        return "Autumn"
    if month in (6, 7, 8):
        return "Winter"
    return "Spring"


@st.cache_data(show_spinner=False)
def load_raw(path=DATA_PATH) -> pd.DataFrame:
    """Load the raw weatherAUS.csv once and share it across the whole app.

    This is the single source of truth: EDA, Data Preprocessing, Feature Engineering
    and model training all consume this same cached DataFrame (in Streamlit).
    """
    return pd.read_csv(path, na_values=["NA"])


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """构造特征 + 4 个目标。丢弃 RISK_MM（RainTomorrow 的泄漏版），保留 Date 供展示。"""
    df = df.copy()
    df["RainToday"] = df["RainToday"].map({"No": 0, "Yes": 1})
    df["RainTomorrow"] = df["RainTomorrow"].map({"No": 0, "Yes": 1})
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df = df.sort_values(["Location", "Date"]).reset_index(drop=True)
    df["season"] = df["Date"].dt.month.map(get_season_sh)

    # J+1 / J+2 目标：按站点 shift，并跳过日期断档
    g = df.groupby("Location")
    gap1 = (g["Date"].shift(-1) - df["Date"]).dt.days
    gap2 = (g["Date"].shift(-2) - df["Date"]).dt.days
    df["RainInTwoDays"] = np.where(gap2 == 2, g["RainToday"].shift(-2), np.nan)
    df["MaxTempTomorrow"] = np.where(gap1 == 1, g["MaxTemp"].shift(-1), np.nan)
    df["MaxTempInTwoDays"] = np.where(gap2 == 2, g["MaxTemp"].shift(-2), np.nan)

    df = df.drop(columns=["RISK_MM"], errors="ignore")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in TARGETS and c != "Date"]


def build_cat_dtypes(locations: list[str]) -> dict[str, pd.CategoricalDtype]:
    """固定类别顺序，保证训练/预测时 category 编码一致。"""
    return {
        "Location": pd.CategoricalDtype(categories=locations),
        "WindGustDir": pd.CategoricalDtype(categories=WIND_DIRS),
        "WindDir9am": pd.CategoricalDtype(categories=WIND_DIRS),
        "WindDir3pm": pd.CategoricalDtype(categories=WIND_DIRS),
        "season": pd.CategoricalDtype(categories=SEASONS),
    }


def cast_categories(df: pd.DataFrame, cat_dtypes: dict) -> pd.DataFrame:
    df = df.copy()
    for col, dtype in cat_dtypes.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    return df


def _classifier(spw: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_lambda=5.0,
        reg_alpha=0.1,
        scale_pos_weight=spw,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def _regressor() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_lambda=5.0,
        reg_alpha=0.1,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="mae",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def train_and_save() -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = engineer(load_raw())
    locations = sorted(df["Location"].dropna().unique().tolist())
    cat_dtypes = build_cat_dtypes(locations)
    df = cast_categories(df, cat_dtypes)
    features = feature_columns(df)

    models: dict[str, object] = {}
    metrics: dict[str, dict] = {}

    for target in TARGETS:
        sub = df.dropna(subset=[target]).reset_index(drop=True)
        X = sub[features]
        y = sub[target]
        stratify = y if target in CLASS_TARGETS else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
        )

        if target in CLASS_TARGETS:
            spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
            model = _classifier(spw)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics[target] = {
                "type": "classification",
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "pos_rate_train": round(float(y_train.mean()), 4),
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
                "precision": round(precision_score(y_test, y_pred), 4),
                "recall": round(recall_score(y_test, y_pred), 4),
                "f1": round(f1_score(y_test, y_pred), 4),
                "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            }
        else:
            model = _regressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics[target] = {
                "type": "regression",
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "mae": round(float(mean_absolute_error(y_test, y_pred)), 3),
                "rmse": round(float(root_mean_squared_error(y_test, y_pred)), 3),
                "r2": round(r2_score(y_test, y_pred), 4),
            }

        models[target] = model
        joblib.dump(model, MODEL_DIR / f"{target}.joblib")
        print(f"[saved] {target}: {metrics[target]}")

    # 主模型（明日降雨）的特征重要度
    importance = {
        features[i]: float(models["RainTomorrow"].feature_importances_[i])
        for i in range(len(features))
    }

    meta = {
        "features": features,
        "cat_dtypes": {k: list(v.categories) for k, v in cat_dtypes.items()},
        "locations": locations,
        "targets": TARGETS,
        "metrics": metrics,
        "importance_rain_tomorrow": importance,
        "random_state": RANDOM_STATE,
    }
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] meta.json -> models/ ({len(features)} features, {len(locations)} locations)")
    return meta


def load_models() -> dict:
    meta = json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))
    models = {t: joblib.load(MODEL_DIR / f"{t}.joblib") for t in meta["targets"]}
    meta["models"] = models
    return meta


if __name__ == "__main__":
    train_and_save()
