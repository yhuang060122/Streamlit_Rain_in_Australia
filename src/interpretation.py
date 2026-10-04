"""Model Interpretation page renderer — feature importance + global/local SHAP.

Explains the XGBoost classifier trained on the model-ready matrix produced by the
Feature Engineering stage (same data as the Modelling page), so the data flow is
preserved end to end.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import shap
from xgboost import XGBClassifier

from src.gating import mark_done, require
from src.preprocessing import run_preprocessing

# Bump to invalidate cached SHAP results whenever the explanation logic changes.
_INTERPRETATION_VERSION = "2026-10-04"


@st.cache_resource(show_spinner="Training model & building SHAP explainer…")
def get_explainer(_cache_version: str = _INTERPRETATION_VERSION):
    res = run_preprocessing()
    X_train = res["X_train_final"]
    y_train = res["y_train"]
    spw = float((y_train == 0).sum() / (y_train == 1).sum())
    model = XGBClassifier(scale_pos_weight=spw, eval_metric="logloss", random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)
    return model, explainer


def _positive_class(values):
    """Normalise the SHAP output of a binary classifier to the positive (rain) class."""
    if isinstance(values, list):
        return values[1] if len(values) == 2 else values[0]
    arr = np.asarray(values)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def _base_value(explainer) -> float:
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        return float(ev[1] if len(ev) > 1 else ev[0])
    return float(ev)


@st.cache_data(show_spinner="Computing global SHAP values…")
def get_global_shap(_version: str = _INTERPRETATION_VERSION):
    model, explainer = get_explainer()
    res = run_preprocessing()
    X_train = res["X_train_final"]
    feature_names = list(X_train.columns)
    X_explain = X_train.sample(2000, random_state=42)
    sv = _positive_class(explainer.shap_values(X_explain))
    importances = dict(zip(feature_names, model.feature_importances_.tolist()))
    return X_explain, sv, feature_names, importances


def render() -> None:
    st.set_page_config(page_title="Model Interpretation — Rain in Australia", page_icon="🔬", layout="wide")
    require("modeling")
    mark_done("model_interpretation")

    model, explainer = get_explainer()
    res = run_preprocessing()
    X_test, y_test = res["X_test_final"], res["y_test"]

    st.title("🔬 Model Interpretation")
    st.caption(
        "Explains the XGBoost classifier (1-day target `RainTomorrow`) trained on the Feature Engineering output, "
        "using feature importance and TreeSHAP (global beeswarm + single-prediction waterfall)."
    )

    tabs = st.tabs(
        [
            "1️⃣ Feature Importance",
            "2️⃣ Global SHAP (Beeswarm)",
            "3️⃣ Single Prediction (Waterfall)",
        ]
    )

    X_explain, sv, feature_names, importances = get_global_shap()

    # ------------------------------------------------------------------ tab 1
    with tabs[0]:
        st.subheader("Feature importance (XGBoost gain)")
        imp_sorted = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:20]
        imp_df = pd.DataFrame(imp_sorted, columns=["Feature", "Importance"]).set_index("Feature")
        st.bar_chart(imp_df, height=480)

    # ------------------------------------------------------------------ tab 2
    with tabs[1]:
        st.subheader("Global SHAP — beeswarm (top 20 features)")
        st.caption(
            "Each dot is one prediction. The x-axis is the SHAP value (impact on the rain probability); "
            "color is the feature value (red = high, blue = low)."
        )
        base = _base_value(explainer)
        exp = shap.Explanation(values=sv, base_values=base, data=X_explain.values, feature_names=feature_names)
        fig = plt.figure(figsize=(11, 10))
        shap.plots.beeswarm(exp, max_display=20, show=False)
        st.pyplot(fig)
        plt.close(fig)

    # ------------------------------------------------------------------ tab 3
    with tabs[2]:
        st.subheader("Single prediction — waterfall")
        if "interp_idx" not in st.session_state:
            st.session_state["interp_idx"] = int(np.random.default_rng(0).integers(len(X_test)))

        col_btn, col_info = st.columns([1, 3])
        if col_btn.button("🎲 Random sample"):
            st.session_state["interp_idx"] = int(np.random.default_rng().integers(len(X_test)))

        idx = st.session_state["interp_idx"]
        row = X_test.iloc[[idx]]
        sv_row = _positive_class(explainer.shap_values(row))
        sv_row = np.asarray(sv_row)[0]
        base = _base_value(explainer)
        pred = float(model.predict_proba(row)[0, 1])
        actual = int(y_test.iloc[idx])

        col_info.markdown(
            f"**Test sample #{idx}** — predicted rain probability **{pred:.0%}**, "
            f"actual **{'Yes (rain)' if actual == 1 else 'No (no rain)'}**"
        )

        exp = shap.Explanation(
            values=sv_row, base_values=base, data=row.iloc[0].values, feature_names=feature_names
        )
        fig = plt.figure(figsize=(10, 12))
        shap.plots.waterfall(exp, max_display=15, show=False)
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            f"Base value (expected log-odds) = {base:.4f}; the bars push the prediction up (red) or down (blue) "
            "to reach the final probability."
        )
