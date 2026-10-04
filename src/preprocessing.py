"""Preprocessing pipeline (pure data logic) — shared by the Data Preprocessing
and Feature Engineering pages.

Reproduces the steps of docs/02_preprocessing_avec_2jours_prediction.ipynb up to
the final scaled feature matrix, and returns every intermediate artifact in a dict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import jarque_bera, pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.pipeline import load_raw

# Bump this to invalidate cached pipeline results whenever the preprocessing logic changes.
_PIPELINE_VERSION = "2026-10-04c"

COLS_DROP = ["Sunshine", "Evaporation", "Cloud3pm", "Cloud9am"]

COLS_MEDIAN = ["Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Humidity9am"]
COLS_MEAN = ["MinTemp", "MaxTemp", "Temp9am", "Temp3pm", "Humidity3pm", "Pressure9am", "Pressure3pm"]
COLS_MODE = ["WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"]

CONTINUOUS_COLS = [
    "MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Temp9am", "Temp3pm",
    "Temp_diff", "Humidity_diff", "Pressure_diff",
]


def get_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Summer"
    if month in (3, 4, 5):
        return "Autumn"
    if month in (6, 7, 8):
        return "Winter"
    return "Spring"


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
    df = df.drop(columns=COLS_DROP)
    leakage_col = "RISK_MM" if "RISK_MM" in df.columns else None
    if leakage_col:
        df = df.drop(columns=[leakage_col])
    out["leakage_col"] = leakage_col
    out["shape_after_drop"] = df.shape

    # ---- Step 3: build the two-day target (merge on Location + Date-2) ----
    df["Date"] = pd.to_datetime(df["Date"])
    target_2j = df[["Location", "Date", "RainToday"]].copy()
    target_2j["Date"] = target_2j["Date"] - pd.Timedelta(days=2)
    target_2j = target_2j.rename(columns={"RainToday": "RainInTwoDays"})
    df = df.merge(target_2j, on=["Location", "Date"], how="left", validate="one_to_one")
    out["rows_before_dropna"] = len(df)
    df = df.dropna(subset=["RainTomorrow", "RainInTwoDays"])
    out["rows_after_dropna"] = len(df)

    # ---- Step 4: stratified train/test split (80/20 on both targets) ----
    y = df[["RainTomorrow", "RainInTwoDays"]]
    X = df.drop(columns=["RainTomorrow", "RainInTwoDays"])
    strat_key = y.astype(str).agg("_".join, axis=1)
    X_train, X_test, y_df_train, y_df_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat_key
    )
    y_train = y_df_train["RainTomorrow"]
    y_train2 = y_df_train["RainInTwoDays"]
    y_test = y_df_test["RainTomorrow"]
    y_test2 = y_df_test["RainInTwoDays"]
    out["dim_train"] = X_train.shape
    out["dim_test"] = X_test.shape
    out["prop_train"] = (y_train.value_counts(normalize=True) * 100).round(2)
    out["prop_test"] = (y_test.value_counts(normalize=True) * 100).round(2)

    # ---- Step 5: IQR outlier detection (train only) ----
    X_train_num = X_train.select_dtypes(include=[np.number])
    out["numeric_cols"] = list(X_train_num.columns)
    out["X_train_num"] = X_train_num.copy()  # pre-imputation snapshot (for boxplots / distributions)
    iqr_rows = []
    for col in X_train_num.columns:
        q1 = X_train_num[col].quantile(0.25)
        q3 = X_train_num[col].quantile(0.75)
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

    # ---- Step 6: skewness (decides median vs mean imputation) ----
    skew_rows = []
    for col in X_train_num.columns:
        s = X_train_num[col].skew()
        skew_rows.append({
            "Variable": col, "Skewness": round(s, 3),
            "Imputation rule": "median" if abs(s) >= 0.5 else "mean",
            "Missing": int(X_train[col].isna().sum()),
        })
    out["skew_table"] = (
        pd.DataFrame(skew_rows)
        .reindex(pd.DataFrame(skew_rows)["Skewness"].abs().sort_values(ascending=False).index)
        .reset_index(drop=True)
    )

    # ---- Step 7: imputation (stats computed on X_train only, applied to both) ----
    impute_values: dict = {}
    for col in COLS_MEDIAN:
        impute_values[col] = X_train[col].median()
    for col in COLS_MEAN:
        impute_values[col] = X_train[col].mean()
    for col in COLS_MODE:
        impute_values[col] = X_train[col].mode()[0]
    for col, val in impute_values.items():
        X_train[col] = X_train[col].fillna(val)
        X_test[col] = X_test[col].fillna(val)

    method_map = {c: "median" for c in COLS_MEDIAN}
    method_map.update({c: "mean" for c in COLS_MEAN})
    method_map.update({c: "mode" for c in COLS_MODE})
    impute_table = pd.DataFrame([
        {
            "Variable": col,
            "Method": method_map[col],
            "Imputed value": f"{val:.2f}" if isinstance(val, (int, float, np.floating)) else str(val),
        }
        for col, val in impute_values.items()
    ])
    out["impute_table"] = impute_table
    out["train_na_after"] = int(X_train.isna().sum().sum())
    out["test_na_after"] = int(X_test.isna().sum().sum())

    # snapshot of the cleaned data (post-imputation, 0 NaN) — the output of this stage,
    # which is exactly the input handed to Feature Engineering
    out["cleaned_head"] = X_train.head(12).copy()
    out["cleaned_shape"] = X_train.shape
    out["cleaned_cols"] = list(X_train.columns)

    # ---- Step 8: feature engineering (row-wise, no fitted stats) ----
    def engineer_vars(Z):
        Z = Z.copy()
        Z["Date"] = pd.to_datetime(Z["Date"])
        Z["Month"] = Z["Date"].dt.month
        Z["Season"] = Z["Month"].apply(get_season)
        Z["Temp_diff"] = Z["Temp3pm"] - Z["Temp9am"]
        Z["Humidity_diff"] = Z["Humidity3pm"] - Z["Humidity9am"]
        Z["Pressure_diff"] = Z["Pressure3pm"] - Z["Pressure9am"]
        return Z

    X_train = engineer_vars(X_train)
    X_test = engineer_vars(X_test)

    y_train_bin = (y_train == "Yes").astype(int)
    feat_cols = ["Temp_diff", "Humidity_diff", "Pressure_diff", "Month"]
    out["feature_corr"] = (
        X_train[feat_cols].corrwith(y_train_bin).sort_values(key=lambda s: s.abs(), ascending=False).round(4)
    )

    # snapshot of the feature-engineered data (new variables added, before encoding/scaling)
    out["fe_head"] = X_train.head(12).copy()
    out["fe_shape"] = X_train.shape
    out["fe_new_cols"] = ["Month", "Season", "Temp_diff", "Humidity_diff", "Pressure_diff"]

    # ---- Step 9: encoding ----
    y_train_enc = (y_train == "Yes").astype(int)
    y_test_enc = (y_test == "Yes").astype(int)
    y_train2_enc = (y_train2 == "Yes").astype(int)
    y_test2_enc = (y_test2 == "Yes").astype(int)

    X_train["RainToday"] = X_train["RainToday"].map({"No": 0, "Yes": 1})
    X_test["RainToday"] = X_test["RainToday"].map({"No": 0, "Yes": 1})

    loc_rate = y_train_enc.groupby(X_train["Location"]).mean()
    global_rate = float(y_train_enc.mean())
    X_train["Location_encoded"] = X_train["Location"].map(loc_rate)
    X_test["Location_encoded"] = X_test["Location"].map(loc_rate)
    unseen = int(X_test["Location_encoded"].isna().sum())
    X_test["Location_encoded"] = X_test["Location_encoded"].fillna(global_rate)

    X_train = X_train.drop(columns=["Location", "Date"])
    X_test = X_test.drop(columns=["Location", "Date"])

    cols_onehot = ["WindGustDir", "WindDir9am", "WindDir3pm", "Season"]
    X_train = pd.get_dummies(X_train, columns=cols_onehot, drop_first=True, dtype=int)
    X_test = pd.get_dummies(X_test, columns=cols_onehot, drop_first=True, dtype=int)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    out["loc_rate_pct"] = (loc_rate * 100).sort_values(ascending=False).round(2)
    out["unseen_locations"] = unseen
    out["global_rate"] = round(global_rate, 4)
    out["enc_dim"] = X_train.shape

    # ---- Step 10: Pearson vs Spearman (on encoded, pre-scaling X_train) ----
    ps_vars = [
        "MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
        "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Temp9am", "Temp3pm",
        "Temp_diff", "Humidity_diff", "Pressure_diff", "Month", "Location_encoded",
    ]
    ps_rows = []
    for col in ps_vars:
        p = pearsonr(X_train[col], y_train_enc)[0]
        s = spearmanr(X_train[col], y_train_enc)[0]
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

    # ---- Step 11: Jarque-Bera normality test (pre-scaling) ----
    jb_rows = []
    for col in CONTINUOUS_COLS:
        jb_stat, jb_p = jarque_bera(X_train[col])
        jb_rows.append({
            "Variable": col, "JB statistic": round(jb_stat, 1),
            "Skewness": round(X_train[col].skew(), 3),
            "p-value (not decisive)": f"{jb_p:.2e}",
        })
    out["jb_table"] = pd.DataFrame(jb_rows).sort_values("JB statistic", ascending=True).reset_index(drop=True)

    # ---- Step 12: scaling (fitted on X_train only) ----
    standard_cols = [
        "MinTemp", "MaxTemp", "Temp9am", "Temp3pm", "Humidity3pm",
        "Pressure9am", "Pressure3pm", "Temp_diff", "Humidity_diff", "Location_encoded",
    ]
    robust_cols = [
        "Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Pressure_diff",
    ]
    scaler_std = StandardScaler()
    X_train[standard_cols] = scaler_std.fit_transform(X_train[standard_cols])
    X_test[standard_cols] = scaler_std.transform(X_test[standard_cols])
    scaler_rob = RobustScaler()
    X_train[robust_cols] = scaler_rob.fit_transform(X_train[robust_cols])
    X_test[robust_cols] = scaler_rob.transform(X_test[robust_cols])

    out["std_mean"] = X_train[standard_cols].mean().round(6)
    out["rob_median"] = X_train[robust_cols].median().round(6)
    out["final_dim"] = X_train.shape

    # ---- final model-ready matrices (input to the Modelling page) ----
    out["X_train_final"] = X_train
    out["X_test_final"] = X_test
    out["y_train"] = y_train_enc
    out["y_test"] = y_test_enc
    out["y_train2"] = y_train2_enc
    out["y_test2"] = y_test2_enc

    return out
