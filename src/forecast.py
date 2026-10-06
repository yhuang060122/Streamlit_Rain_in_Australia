"""Shared forecasting logic — model/data loading and prediction helpers.

预测时复用 src.pipeline 的 transform，保证与训练时的特征口径一致。
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
import streamlit as st

from src.pipeline import (
    CLASS_TARGETS,
    MODEL_DIR,
    build_targets,
    load_raw,
    transform,
    transformer_from_meta,
)

# Bump to invalidate cached forecast data when the pipeline / models change.
_FORECAST_VERSION = "2026-10-06-spec"


@st.cache_data(show_spinner=False)
def load_meta(_cache_version: str = _FORECAST_VERSION) -> dict:
    return json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_models(_cache_version: str = _FORECAST_VERSION) -> dict:
    meta = load_meta(_cache_version)
    return {t: joblib.load(MODEL_DIR / f"{t}.joblib") for t in meta["targets"]}


@st.cache_data(show_spinner="Preparing data…")
def get_engineered(_cache_version: str = _FORECAST_VERSION) -> pd.DataFrame:
    """返回 build_targets 后的数据（原始观测 + 4 个目标），供 historical replay 展示与取行。"""
    return build_targets(load_raw())


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


def predict_row(models: dict, meta: dict, X_raw: pd.DataFrame) -> dict:
    """对原始观测行（含 Date / Location / 原始列 / RainToday 0/1）做变换并预测。

    J+1 目标（RainTomorrow / MaxTempTomorrow）用 J+1 特征，J+2 目标用 J+2 特征。
    """
    transformer = transformer_from_meta(meta)
    X1 = transform(X_raw, transformer, "J1")
    X2 = transform(X_raw, transformer, "J2")

    out = {}
    for t in meta["targets"]:
        horizon = meta["horizon"][t]
        X = X1 if horizon == "J1" else X2
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
