"""Preprocessing pipeline（规范版）— 供 Data Preprocessing / Feature Engineering 页展示。

复用 src.pipeline 的特征变换，逐阶段记录中间产物。对齐「Pré-traitement」规范：
  - 时间切分：训练 ≤ 2015-11-09，测试 > 该日。
  - 删高缺失列：Sunshine / Evaporation / Cloud9am / Cloud3pm。
  - 缺失值填补：9h/15h 互推 + 按站统计量（站内无值退回全局），统计量只从训练集拟合。
  - 删高相关列：MaxTemp / Pressure3pm / Temp9am。
  - 编码：风向 / 季节 / 月份 sin/cos 三角编码；Location 目标编码，J+1 / J+2 各一版。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import jarque_bera, pearsonr, spearmanr
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.pipeline import (
    COLS_DROP_CORR,
    COLS_DROP_HIGH_MISSING,
    COLS_MEAN,
    COLS_MEDIAN,
    COLS_MODE,
    CUTOFF_DATE,
    TARGETS,
    TIME_PAIRS,
    _apply_imputers,
    _cross_fill_pairs,
    build_targets,
    fit_transformer,
    get_season,
    load_raw,
    transform,
)

# Bump this to invalidate cached pipeline results whenever the preprocessing logic changes.
_PIPELINE_VERSION = "2026-10-06-spec"

# 兼容 data_processing.py 的旧引用
COLS_DROP = COLS_DROP_HIGH_MISSING

# 三角编码列（[-1,1]，不参与缩放）
_TRIG_COLS = [
    "WindGustDir_sin", "WindGustDir_cos",
    "WindDir9am_sin", "WindDir9am_cos",
    "WindDir3pm_sin", "WindDir3pm_cos",
    "Season_sin", "Season_cos",
    "Month_sin", "Month_cos",
]

# 缩放列（删高相关列之后）
_STANDARD_COLS = ["MinTemp", "Temp3pm", "Humidity3pm", "Pressure9am", "Temp_diff", "Humidity_diff", "Location_encoded"]
_ROBUST_COLS = ["Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Pressure_diff"]

# Jarque-Bera 检验的连续变量（删高相关列之后）
_CONTINUOUS_COLS = [
    "MinTemp", "Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm", "Pressure9am", "Temp3pm",
    "Temp_diff", "Humidity_diff", "Pressure_diff",
]

# Pearson / Spearman 与目标的相关性分析变量
_PS_VARS = [
    "MinTemp", "Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm", "Pressure9am", "Temp3pm",
    "Temp_diff", "Humidity_diff", "Pressure_diff",
    "Month_sin", "Month_cos", "Season_sin", "Season_cos", "Location_encoded",
]

_CROSS_FILL_FLAT = {c for pair in TIME_PAIRS for c in pair}
_TIME_PAIR_FLAT = [c for pair in TIME_PAIRS for c in pair]


@st.cache_data(show_spinner="Running the preprocessing pipeline…")
def run_preprocessing(_cache_version: str = _PIPELINE_VERSION) -> dict:
    out: dict = {}
    df = load_raw()

    # ---- Step 1: missing values ----
    out["missing_pct"] = (df.isna().sum() / len(df) * 100).sort_values(ascending=False).round(2)
    out["rows_total"] = len(df)
    out["rows_with_na"] = int(df.isna().any(axis=1).sum())

    # ---- Step 2: drop high-missing columns (+ leakage column RISK_MM) ----
    out["shape_before"] = df.shape
    df = df.drop(columns=[c for c in COLS_DROP_HIGH_MISSING if c in df.columns])
    leakage_col = "RISK_MM" if "RISK_MM" in df.columns else None
    if leakage_col:
        df = df.drop(columns=[leakage_col])
    out["leakage_col"] = leakage_col
    out["shape_after_drop"] = df.shape

    # ---- Step 3: build targets (map RainToday/RainTomorrow, J+2 target) ----
    df = build_targets(df)
    out["rows_before_dropna"] = len(df)
    df = df.dropna(subset=["RainTomorrow", "RainInTwoDays"]).reset_index(drop=True)
    out["rows_after_dropna"] = len(df)

    # ---- Step 4: temporal split (train ≤ cutoff, test > cutoff) ----
    cutoff = pd.Timestamp(CUTOFF_DATE)
    train_mask = df["Date"] <= cutoff
    df_train = df[train_mask].reset_index(drop=True)
    df_test = df[~train_mask].reset_index(drop=True)

    y_train = df_train["RainTomorrow"].reset_index(drop=True)
    y_train2 = df_train["RainInTwoDays"].reset_index(drop=True)
    y_test = df_test["RainTomorrow"].reset_index(drop=True)
    y_test2 = df_test["RainInTwoDays"].reset_index(drop=True)

    # 供 modelling / interpretation 按站点分析（与 X_test_final 行对齐）
    out["test_locations"] = df_test["Location"].reset_index(drop=True)
    out["test_dates"] = df_test["Date"].reset_index(drop=True)

    out["cutoff"] = CUTOFF_DATE
    n_feat_cols = len(df_train.columns) - len(TARGETS)
    out["dim_train"] = (len(df_train), n_feat_cols)
    out["dim_test"] = (len(df_test), n_feat_cols)
    out["train_date_min"] = str(df_train["Date"].min().date())
    out["train_date_max"] = str(df_train["Date"].max().date())
    out["test_date_min"] = str(df_test["Date"].min().date())
    out["test_date_max"] = str(df_test["Date"].max().date())
    label_map = {0: "No", 1: "Yes"}
    out["prop_train"] = (y_train.map(label_map).value_counts(normalize=True) * 100).round(2)
    out["prop_test"] = (y_test.map(label_map).value_counts(normalize=True) * 100).round(2)

    # ---- Step 5: IQR outlier detection (train only, numeric cols) ----
    X_train_num = df_train.select_dtypes(include=[np.number]).drop(
        columns=["RainToday", "RainTomorrow", "RainInTwoDays", "MaxTempTomorrow", "MaxTempInTwoDays"],
        errors="ignore",
    )
    out["numeric_cols"] = list(X_train_num.columns)
    out["X_train_num"] = X_train_num.copy()
    iqr_rows = []
    for col in X_train_num.columns:
        q1, q3 = X_train_num[col].quantile(0.25), X_train_num[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (X_train_num[col] < lo) | (X_train_num[col] > hi)
        nb = int(mask.sum())
        nn = int(X_train_num[col].notna().sum())
        iqr_rows.append({
            "Variable": col, "Q1": round(q1, 2), "Q3": round(q3, 2),
            "Lower bound": round(lo, 2), "Upper bound": round(hi, 2),
            "Outliers": nb, "Outliers %": round(nb / nn * 100, 2),
        })
    out["iqr_table"] = (
        pd.DataFrame(iqr_rows).sort_values("Outliers %", ascending=False).reset_index(drop=True)
    )

    # ---- Step 6: skewness (informative only; imputation is per-station) ----
    skew_rows = []
    for col in X_train_num.columns:
        skew_rows.append({
            "Variable": col, "Skewness": round(X_train_num[col].skew(), 3),
            "Missing": int(X_train_num[col].isna().sum()),
        })
    out["skew_table"] = (
        pd.DataFrame(skew_rows)
        .reindex(pd.DataFrame(skew_rows)["Skewness"].abs().sort_values(ascending=False).index)
        .reset_index(drop=True)
    )

    # ---- Step 7: imputation stats (9h/15h cross-fill count + per-station imputers) ----
    before_na = int(df_train[_TIME_PAIR_FLAT].isna().sum().sum())
    after_cross_na = int(_cross_fill_pairs(df_train[_TIME_PAIR_FLAT]).isna().sum().sum())
    out["cross_fill_filled"] = before_na - after_cross_na

    transformer = fit_transformer(df_train)

    impute_rows = []
    for col in COLS_MEDIAN:
        impute_rows.append({"Variable": col, "Statistic": "median", "9h/15h cross-fill": "✓" if col in _CROSS_FILL_FLAT else "", "Per-station": "✓"})
    for col in COLS_MEAN:
        impute_rows.append({"Variable": col, "Statistic": "mean", "9h/15h cross-fill": "✓" if col in _CROSS_FILL_FLAT else "", "Per-station": "✓"})
    for col in COLS_MODE:
        impute_rows.append({"Variable": col, "Statistic": "mode", "9h/15h cross-fill": "✓" if col in _CROSS_FILL_FLAT else "", "Per-station": "✓"})
    out["impute_table"] = pd.DataFrame(impute_rows)

    # ---- Step 8: final feature matrices via shared transform (J1 / J2) ----
    X_train_enc1 = transform(df_train, transformer, "J1")
    X_test_enc1 = transform(df_test, transformer, "J1")
    X_train_enc2 = transform(df_train, transformer, "J2")
    X_test_enc2 = transform(df_test, transformer, "J2")

    out["train_na_after"] = int(X_train_enc1.isna().sum().sum())
    out["test_na_after"] = int(X_test_enc1.isna().sum().sum())

    # 填补后、编码前的数据（供 Data Preprocessing 页「Cleaned Data」展示）
    X_imp = _apply_imputers(_cross_fill_pairs(df_train), transformer["imputers"])
    target_cols = [c for c in TARGETS if c in X_imp.columns]
    out["cleaned_head"] = X_imp.drop(columns=target_cols).head(12).copy()
    out["cleaned_shape"] = (len(X_imp), len(X_imp.columns) - len(target_cols))
    out["cleaned_cols"] = [c for c in X_imp.columns if c not in target_cols]

    loc_rate = pd.Series(transformer["loc_rate"])
    loc_rate2 = pd.Series(transformer["loc_rate2"])
    out["loc_rate_pct"] = (loc_rate * 100).sort_values(ascending=False).round(2)
    out["loc_rate_pct2"] = (loc_rate2 * 100).sort_values(ascending=False).round(2)
    out["global_rate"] = round(transformer["global_rate"], 4)
    out["global_rate2"] = round(transformer["global_rate2"], 4)
    out["unseen_locations"] = int((~df_test["Location"].isin(transformer["loc_rate"].keys())).sum())
    out["enc_dim"] = X_train_enc1.shape
    out["trig_cols"] = _TRIG_COLS
    out["corr_dropped"] = COLS_DROP_CORR

    # ---- Step 9: feature engineering correlation (new variables vs target) ----
    feat_cols = ["Temp_diff", "Humidity_diff", "Pressure_diff", "Month_sin", "Month_cos"]
    out["feature_corr"] = (
        X_train_enc1[feat_cols].corrwith(y_train.astype(float))
        .sort_values(key=lambda s: s.abs(), ascending=False).round(4)
    )
    out["fe_head"] = X_train_enc1.head(12).copy()
    out["fe_shape"] = X_train_enc1.shape
    out["fe_new_cols"] = ["Temp_diff", "Humidity_diff", "Pressure_diff"] + _TRIG_COLS + ["Location_encoded"]

    # ---- Step 10: Pearson vs Spearman ----
    ps_rows = []
    for col in _PS_VARS:
        p = pearsonr(X_train_enc1[col], y_train.astype(float))[0]
        s = spearmanr(X_train_enc1[col], y_train.astype(float))[0]
        ps_rows.append({
            "Variable": col, "Pearson": round(p, 4), "Spearman": round(s, 4),
            "Gap (S−P)": round(s - p, 4),
            "Non-linear gain": round(abs(s) - abs(p), 4),
        })
    out["ps_table"] = (
        pd.DataFrame(ps_rows)
        .reindex(pd.DataFrame(ps_rows)["Spearman"].abs().sort_values(ascending=False).index)
        .reset_index(drop=True)
    )

    # ---- Step 11: Jarque-Bera normality test ----
    jb_rows = []
    for col in _CONTINUOUS_COLS:
        jb_stat, jb_p = jarque_bera(X_train_enc1[col])
        jb_rows.append({
            "Variable": col, "JB statistic": round(jb_stat, 1),
            "Skewness": round(X_train_enc1[col].skew(), 3),
            "p-value (not decisive)": f"{jb_p:.2e}",
        })
    out["jb_table"] = pd.DataFrame(jb_rows).sort_values("JB statistic", ascending=True).reset_index(drop=True)

    # ---- Step 12: scaling (fitted on X_train only) ----
    def _scale(Xtr, Xte):
        scaler_std = StandardScaler()
        Xtr[_STANDARD_COLS] = scaler_std.fit_transform(Xtr[_STANDARD_COLS])
        Xte[_STANDARD_COLS] = scaler_std.transform(Xte[_STANDARD_COLS])
        scaler_rob = RobustScaler()
        Xtr[_ROBUST_COLS] = scaler_rob.fit_transform(Xtr[_ROBUST_COLS])
        Xte[_ROBUST_COLS] = scaler_rob.transform(Xte[_ROBUST_COLS])
        return Xtr, Xte

    X_train_enc1, X_test_enc1 = _scale(X_train_enc1, X_test_enc1)
    X_train_enc2, X_test_enc2 = _scale(X_train_enc2, X_test_enc2)

    out["std_mean"] = X_train_enc1[_STANDARD_COLS].mean().round(6)
    out["rob_median"] = X_train_enc1[_ROBUST_COLS].median().round(6)
    out["final_dim"] = X_train_enc1.shape

    # ---- final model-ready matrices ----
    out["X_train_final"] = X_train_enc1
    out["X_test_final"] = X_test_enc1
    out["X_train_final2"] = X_train_enc2
    out["X_test_final2"] = X_test_enc2
    out["y_train"] = y_train.astype(int)
    out["y_test"] = y_test.astype(int)
    out["y_train2"] = y_train2.astype(int)
    out["y_test2"] = y_test2.astype(int)

    return out
