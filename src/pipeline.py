"""数据加载 + 特征工程 + 训练管道（规范版）。

对齐「Pré-traitement」规范：
  - 时间切分：训练 ≤ 2015-11-09，测试 > 该日（不再随机切分）。
  - 删高缺失列：Sunshine / Evaporation / Cloud9am / Cloud3pm（>20% 缺失）。
  - 缺失值填补：9h/15h 互相推断，再按站填补（站内无统计量退回全局）。
  - 删高相关列：MaxTemp / Pressure3pm / Temp9am（三对 Pearson > 0.8）。
  - 编码：风向 / 季节 / 月份用 sin/cos 三角编码；Location 用目标编码，J+1、J+2 各一版。
  - 双 horizon：J+1（RainTomorrow / MaxTempTomorrow）与 J+2（RainInTwoDays / MaxTempInTwoDays）分开训练。

特征变换逻辑在此处定义，训练（train_and_save）与预测（forecast）共用，保证口径一致。
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
from xgboost import XGBClassifier, XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "weatherAUS.csv"
MODEL_DIR = BASE_DIR / "models"

# --------------------------------------------------------------------------- #
# 规范常量
# --------------------------------------------------------------------------- #
CUTOFF_DATE = "2015-11-09"  # 时间切分点：训练 ≤ 该日，测试 > 该日

# 高缺失列（>20% 缺失），直接删除
COLS_DROP_HIGH_MISSING = ["Sunshine", "Evaporation", "Cloud3pm", "Cloud9am"]

# 高相关列（三对 Pearson > 0.8 各删其一）
COLS_DROP_CORR = ["MaxTemp", "Pressure3pm", "Temp9am"]

# 9h / 15h 成对变量：一个缺失时用同一天另一个推断
TIME_PAIRS = [
    ("Temp9am", "Temp3pm"),
    ("Humidity9am", "Humidity3pm"),
    ("Pressure9am", "Pressure3pm"),
    ("WindSpeed9am", "WindSpeed3pm"),
]

WIND_DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
WIND_ANGLE = {d: i * (360.0 / len(WIND_DIRS)) for i, d in enumerate(WIND_DIRS)}

SEASONS = ["Summer", "Autumn", "Winter", "Spring"]
SEASON_ANGLE = {s: i * (np.pi / 2) for i, s in enumerate(SEASONS)}

TARGETS = ["RainTomorrow", "RainInTwoDays", "MaxTempTomorrow", "MaxTempInTwoDays"]
CLASS_TARGETS = ["RainTomorrow", "RainInTwoDays"]
REG_TARGETS = ["MaxTempTomorrow", "MaxTempInTwoDays"]
TARGET_HORIZON = {
    "RainTomorrow": "J1",
    "MaxTempTomorrow": "J1",
    "RainInTwoDays": "J2",
    "MaxTempInTwoDays": "J2",
}

# 按站填补：偏态 → median，近正态 → mean，类别 → mode
COLS_MEDIAN = ["Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Humidity9am"]
COLS_MEAN = ["MinTemp", "MaxTemp", "Temp9am", "Temp3pm", "Humidity3pm", "Pressure9am", "Pressure3pm"]
COLS_MODE = ["WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"]

RANDOM_STATE = 42

# 最终特征列顺序（J+1 / J+2 共用同一套列名，仅 Location_encoded 的取值不同）
FEATURES = [
    "MinTemp", "Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm", "Pressure9am", "Temp3pm",
    "Temp_diff", "Humidity_diff", "Pressure_diff",
    "WindGustDir_sin", "WindGustDir_cos",
    "WindDir9am_sin", "WindDir9am_cos",
    "WindDir3pm_sin", "WindDir3pm_cos",
    "Season_sin", "Season_cos",
    "Month_sin", "Month_cos",
    "RainToday", "Location_encoded",
]


def get_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Summer"
    if month in (3, 4, 5):
        return "Autumn"
    if month in (6, 7, 8):
        return "Winter"
    return "Spring"


@st.cache_data(show_spinner=False)
def load_raw(path=DATA_PATH) -> pd.DataFrame:
    """加载 weatherAUS.csv，整站共享同一份缓存。"""
    return pd.read_csv(path, na_values=["NA"])


# --------------------------------------------------------------------------- #
# 特征变换
# --------------------------------------------------------------------------- #
def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """映射 RainToday/RainTomorrow 为 0/1，并构建 J+2 降雨、J+1/J+2 最高温目标。"""
    df = df.copy()
    df["RainToday"] = df["RainToday"].map({"No": 0, "Yes": 1})
    df["RainTomorrow"] = df["RainTomorrow"].map({"No": 0, "Yes": 1})
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df = df.sort_values(["Location", "Date"]).reset_index(drop=True)

    g = df.groupby("Location")
    gap1 = (g["Date"].shift(-1) - df["Date"]).dt.days
    gap2 = (g["Date"].shift(-2) - df["Date"]).dt.days
    # 只在日期连续（间隔恰好 1 / 2 天）时才算有效目标，避免跨站点断档
    df["RainInTwoDays"] = np.where(gap2 == 2, g["RainToday"].shift(-2), np.nan)
    df["MaxTempTomorrow"] = np.where(gap1 == 1, g["MaxTemp"].shift(-1), np.nan)
    df["MaxTempInTwoDays"] = np.where(gap2 == 2, g["MaxTemp"].shift(-2), np.nan)
    return df


def verify_j2_target(df: pd.DataFrame) -> float:
    """自检：用 +1 天偏移重建 RainTomorrow，应 100% 命中（验证 J+2 join 的 +2 偏移正确）。"""
    g = df.groupby("Location")
    gap1 = (g["Date"].shift(-1) - df["Date"]).dt.days
    rebuilt = np.where(gap1 == 1, g["RainToday"].shift(-1), np.nan)
    valid = df["RainTomorrow"].notna() & pd.notna(rebuilt)
    return float((df.loc[valid, "RainTomorrow"] == rebuilt[valid]).mean())


def _cross_fill_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """9h 缺失 → 用同一天 15h 推断；反之亦然。缺失的列直接跳过。"""
    df = df.copy()
    for c9, c3 in TIME_PAIRS:
        if c9 not in df.columns or c3 not in df.columns:
            continue
        m9 = df[c9].isna() & df[c3].notna()
        m3 = df[c3].isna() & df[c9].notna()
        df.loc[m9, c9] = df.loc[m9, c3]
        df.loc[m3, c3] = df.loc[m3, c9]
    return df


def _fit_imputers(df_train: pd.DataFrame) -> dict:
    """在训练集上拟合按站统计量（median/mean/mode），站内全缺失退回全局。"""
    imputers = {}
    for col in COLS_MEDIAN:
        station = df_train.groupby("Location")[col].median()
        imputers[col] = ("median", station, df_train[col].median())
    for col in COLS_MEAN:
        station = df_train.groupby("Location")[col].mean()
        imputers[col] = ("mean", station, df_train[col].mean())
    for col in COLS_MODE:
        station = df_train.groupby("Location")[col].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan
        )
        imputers[col] = ("mode", station, df_train[col].mode().iloc[0])
    return imputers


def _apply_imputers(df: pd.DataFrame, imputers: dict) -> pd.DataFrame:
    df = df.copy()
    for col, (_, station, global_v) in imputers.items():
        if col not in df.columns:
            continue
        vals = df[col].copy()
        for loc in df["Location"].unique():
            if loc in station.index and pd.notna(station[loc]):
                mask = df["Location"] == loc
                vals[mask] = vals[mask].fillna(station[loc])
        df[col] = vals.fillna(global_v)
    return df


def _encode_wind(series: pd.Series):
    theta = series.map(WIND_ANGLE).astype(float) * np.pi / 180.0
    return np.sin(theta), np.cos(theta)


def _encode_season(series: pd.Series):
    theta = series.map(SEASON_ANGLE).astype(float)
    return np.sin(theta), np.cos(theta)


def _encode_month(series: pd.Series):
    theta = (series.astype(float) - 1) * 2 * np.pi / 12.0
    return np.sin(theta), np.cos(theta)


def fit_transformer(df_train: pd.DataFrame) -> dict:
    """在时间切分后的训练集上拟合所有统计量，返回 transformer。"""
    y_train = df_train["RainTomorrow"]
    y_train2 = df_train["RainInTwoDays"]
    return {
        "loc_rate": y_train.groupby(df_train["Location"]).mean().to_dict(),
        "loc_rate2": y_train2.groupby(df_train["Location"]).mean().to_dict(),
        "global_rate": float(y_train.mean()),
        "global_rate2": float(y_train2.mean()),
        "imputers": _fit_imputers(df_train),
    }


def transform(df: pd.DataFrame, transformer: dict, horizon: str = "J1") -> pd.DataFrame:
    """把原始观测（含 Date / Location / 原始列 / RainToday 0/1）转成模型特征矩阵。

    顺序：删高缺失列 → 9h/15h 互推 → 按站填补 → 特征工程(diff) → 删高相关列
        → 三角编码 → Location 目标编码 → 选列。
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # 1) 删高缺失列
    df = df.drop(columns=[c for c in COLS_DROP_HIGH_MISSING if c in df.columns])
    # 2) 9h / 15h 互推
    df = _cross_fill_pairs(df)
    # 3) 按站填补（用训练集拟合的统计量）
    df = _apply_imputers(df, transformer["imputers"])
    # 4) 特征工程（diff 需要 Temp9am / Pressure3pm，故在删相关列之前）
    df["Month"] = df["Date"].dt.month
    df["Season"] = df["Month"].map(get_season)
    df["Temp_diff"] = df["Temp3pm"] - df["Temp9am"]
    df["Humidity_diff"] = df["Humidity3pm"] - df["Humidity9am"]
    df["Pressure_diff"] = df["Pressure3pm"] - df["Pressure9am"]
    # 5) 删高相关列
    df = df.drop(columns=[c for c in COLS_DROP_CORR if c in df.columns])
    # 6) 三角编码
    for d in ["WindGustDir", "WindDir9am", "WindDir3pm"]:
        s, c = _encode_wind(df[d])
        df[f"{d}_sin"], df[f"{d}_cos"] = s, c
    s, c = _encode_season(df["Season"])
    df["Season_sin"], df["Season_cos"] = s, c
    s, c = _encode_month(df["Month"])
    df["Month_sin"], df["Month_cos"] = s, c
    # 7) Location 目标编码（按 horizon）
    loc_rate = transformer["loc_rate"] if horizon == "J1" else transformer["loc_rate2"]
    global_rate = transformer["global_rate"] if horizon == "J1" else transformer["global_rate2"]
    df["Location_encoded"] = df["Location"].map(loc_rate).fillna(global_rate)

    return df[FEATURES]


# --------------------------------------------------------------------------- #
# transformer 序列化（meta.json 与内存 dict 互转）
# --------------------------------------------------------------------------- #
def _to_json(v):
    """numpy 标量转原生类型，字符串原样保留（mode 列填补值是 str）。"""
    if isinstance(v, np.generic):
        return v.item()
    return v


def transformer_to_json(t: dict) -> dict:
    return {
        "loc_rate": {k: _to_json(v) for k, v in t["loc_rate"].items()},
        "loc_rate2": {k: _to_json(v) for k, v in t["loc_rate2"].items()},
        "global_rate": float(t["global_rate"]),
        "global_rate2": float(t["global_rate2"]),
        "imputers": {
            col: {
                "method": method,
                "station": {k: _to_json(v) for k, v in station.to_dict().items()},
                "global": _to_json(global_v),
            }
            for col, (method, station, global_v) in t["imputers"].items()
        },
    }


def transformer_from_meta(meta: dict) -> dict:
    t = meta["transformer"]
    imputers = {}
    for col, d in t["imputers"].items():
        imputers[col] = (d["method"], pd.Series(d["station"]), d["global"])
    return {
        "loc_rate": t["loc_rate"],
        "loc_rate2": t["loc_rate2"],
        "global_rate": t["global_rate"],
        "global_rate2": t["global_rate2"],
        "imputers": imputers,
    }


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
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
        eval_metric="mae",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def train_and_save() -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = build_targets(load_raw())

    # 时间切分
    cutoff = pd.Timestamp(CUTOFF_DATE)
    train_mask = df["Date"] <= cutoff
    df_train = df[train_mask].reset_index(drop=True)
    df_test = df[~train_mask].reset_index(drop=True)

    # 自检：J+2 目标构建正确性（+1 偏移应重建 RainTomorrow）
    j2_ok = verify_j2_target(df)
    print(f"[check] J+2 offset verification (rebuild RainTomorrow via +1 shift): {j2_ok:.4%}")

    # 拟合 transformer（只用训练集统计量）
    transformer = fit_transformer(df_train)

    # 双 horizon 特征矩阵
    X_train1 = transform(df_train, transformer, "J1")
    X_test1 = transform(df_test, transformer, "J1")
    X_train2 = transform(df_train, transformer, "J2")
    X_test2 = transform(df_test, transformer, "J2")

    models: dict = {}
    metrics: dict = {}

    for target in TARGETS:
        horizon = TARGET_HORIZON[target]
        X_tr = X_train1 if horizon == "J1" else X_train2
        X_te = X_test1 if horizon == "J1" else X_test2

        # 只保留该目标非 NaN 的行
        valid_tr = df_train[target].notna().to_numpy()
        valid_te = df_test[target].notna().to_numpy()
        X_tr = X_tr[valid_tr].reset_index(drop=True)
        X_te = X_te[valid_te].reset_index(drop=True)
        y_tr = df_train.loc[valid_tr, target].reset_index(drop=True)
        y_te = df_test.loc[valid_te, target].reset_index(drop=True)

        if target in CLASS_TARGETS:
            spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
            model = _classifier(spw)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_proba = model.predict_proba(X_te)[:, 1]
            metrics[target] = {
                "type": "classification",
                "horizon": horizon,
                "n_train": int(len(y_tr)),
                "n_test": int(len(y_te)),
                "pos_rate_train": round(float(y_tr.mean()), 4),
                "accuracy": round(accuracy_score(y_te, y_pred), 4),
                "precision": round(precision_score(y_te, y_pred), 4),
                "recall": round(recall_score(y_te, y_pred), 4),
                "f1": round(f1_score(y_te, y_pred), 4),
                "roc_auc": round(roc_auc_score(y_te, y_proba), 4),
            }
        else:
            model = _regressor()
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            metrics[target] = {
                "type": "regression",
                "horizon": horizon,
                "n_train": int(len(y_tr)),
                "n_test": int(len(y_te)),
                "mae": round(float(mean_absolute_error(y_te, y_pred)), 3),
                "rmse": round(float(root_mean_squared_error(y_te, y_pred)), 3),
                "r2": round(r2_score(y_te, y_pred), 4),
            }

        models[target] = model
        joblib.dump(model, MODEL_DIR / f"{target}.joblib")
        print(f"[saved] {target} ({horizon}): {metrics[target]}")

    # RainTomorrow 主模型的特征重要度
    importance = {
        FEATURES[i]: float(models["RainTomorrow"].feature_importances_[i])
        for i in range(len(FEATURES))
    }

    meta = {
        "features": FEATURES,
        "targets": TARGETS,
        "horizon": TARGET_HORIZON,
        "metrics": metrics,
        "transformer": transformer_to_json(transformer),
        "cutoff": CUTOFF_DATE,
        "j2_verify": j2_ok,
        "importance_rain_tomorrow": importance,
        "random_state": RANDOM_STATE,
    }
    (MODEL_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[saved] meta.json -> models/ ({len(FEATURES)} features)")
    return meta


if __name__ == "__main__":
    train_and_save()
