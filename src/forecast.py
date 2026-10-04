"""Shared forecasting logic — model/data loading and prediction helpers.

Used by the main app (manual forecast) and the Data Understanding page (historical replay).
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
import streamlit as st

from src.pipeline import CLASS_TARGETS, MODEL_DIR, cast_categories, engineer, load_raw

# Bump to invalidate cached forecast data when the pipeline / models change.
_FORECAST_VERSION = "2026-10-04"


@st.cache_data(show_spinner=False)
def load_meta(_cache_version: str = _FORECAST_VERSION) -> dict:
    return json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_models(_cache_version: str = _FORECAST_VERSION) -> dict:
    meta = load_meta(_cache_version)
    return {t: joblib.load(MODEL_DIR / f"{t}.joblib") for t in meta["targets"]}


@st.cache_data(show_spinner="Preparing data…")
def get_engineered(_cache_version: str = _FORECAST_VERSION) -> tuple[pd.DataFrame, dict]:
    meta = load_meta(_cache_version)
    raw = load_raw()
    df = engineer(raw)
    cat_dtypes = {k: pd.CategoricalDtype(categories=v) for k, v in meta["cat_dtypes"].items()}
    df = cast_categories(df, cat_dtypes)
    return df, meta


def rain_bar(prob: float, label: str) -> None:
    pct = prob * 100
    color = "#2563eb" if prob >= 0.5 else "#f59e0b"
    emoji = "🌧️" if prob >= 0.5 else "☀️"
    verdict = "Rain likely" if prob >= 0.5 else "No rain likely"
    st.markdown(
        f"""
        <div style="font-size:0.95rem;color:#6b7280;margin-bottom:4px;">{label}</div>
        <div style="background:#e5e7eb;border-radius:999px;height:16px;width:100%;">
          <div style="background:{color};width:{pct:.1f}%;height:16px;border-radius:999px;"></div>
        </div>
        <div style="margin-top:8px;">
          <span style="font-size:2.0rem;font-weight:700;">{emoji} {pct:.0f}%</span>
          <span style="margin-left:12px;color:#6b7280;">{verdict}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def predict_row(models: dict, meta: dict, X_row: pd.DataFrame) -> dict:
    X = X_row[meta["features"]].copy()
    cat_dtypes = {k: pd.CategoricalDtype(categories=v) for k, v in meta["cat_dtypes"].items()}
    for col, dtype in cat_dtypes.items():
        if col in X.columns and not isinstance(X[col].dtype, pd.CategoricalDtype):
            X[col] = X[col].astype(dtype)
    out = {}
    for t in meta["targets"]:
        if t in CLASS_TARGETS:
            out[t] = float(models[t].predict_proba(X)[0, 1])
        else:
            out[t] = float(models[t].predict(X)[0])
    return out


def actual_summary(row: pd.Series) -> dict:
    return {
        "RainTomorrow": row["RainTomorrow"],
        "RainInTwoDays": row["RainInTwoDays"],
        "MaxTempTomorrow": row["MaxTempTomorrow"],
        "MaxTempInTwoDays": row["MaxTempInTwoDays"],
    }
