"""Model Interpretation page — multi-model feature importance + SHAP + per-station analysis.

复用 src.modeling.run_modeling 的最终模型与评估结果（同一份缓存），不重复训练。
对齐「Modélisation」规范里的「interprétation des modèles」与「analyse par station」。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import streamlit as st
from sklearn.inspection import permutation_importance

from src.gating import mark_done, require
from src.modeling import MODEL_NAMES, run_modeling

_INTERPRETATION_VERSION = "2026-10-06-spec"


def _feature_importance(model, name: str, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """按模型类型提取特征重要度（RF/XGB 用内置，LR 用系数，KNN/MLP 用置换重要度）。"""
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    if hasattr(model, "coef_"):
        return np.abs(model.coef_).flatten()
    # KNN / MLP：置换重要度（少量样本、顺序评估，避免嵌套并行）
    X_sub = X.sample(min(100, len(X)), random_state=42)
    y_sub = y.loc[X_sub.index]
    r = permutation_importance(model, X_sub, y_sub, n_repeats=2, random_state=42,
                               scoring="f1", n_jobs=1)
    return r.importances_mean


def _importance_table(h: dict) -> pd.DataFrame:
    """各模型特征重要度（归一化），按 RainTomorrow 模型的重要性排序。"""
    X, y = h["X_test"], h["test_y"]
    feats = h["feature_names"]
    cols = {}
    for name in MODEL_NAMES:
        imp = _feature_importance(h["models"][name], name, X, y)
        imp = imp / max(imp.sum(), 1e-9)
        cols[name] = imp
    df = pd.DataFrame(cols, index=feats)
    df["mean"] = df.mean(axis=1)
    return df.sort_values("mean", ascending=False)


def _base_value(explainer) -> float:
    ev = explainer.expected_value
    if isinstance(ev, (list, tuple, np.ndarray)):
        return float(np.asarray(ev).ravel()[0])
    return float(ev)


def render() -> None:
    st.set_page_config(page_title="Model Interpretation — Rain in Australia", page_icon="🔬", layout="wide")
    require("modeling")
    mark_done("model_interpretation")

    m = run_modeling()

    st.title("🔬 Model Interpretation")
    st.caption(
        "Multi-model feature importance, SHAP explanations (XGBoost) and per-station analysis — "
        "for the J+1 and J+2 targets, reusing the models trained on the Modelling page."
    )

    horizon = st.radio("Target", ["J+1 (RainTomorrow)", "J+2 (RainInTwoDays)"], horizontal=True)
    h = m["j1"] if horizon.startswith("J+1") else m["j2"]
    label = "J+1" if horizon.startswith("J+1") else "J+2"

    tabs = st.tabs(["1️⃣ Feature Importance", "2️⃣ SHAP (XGBoost)", "3️⃣ Per-station Analysis"])

    # ------------------------------------------------------------------ tab 1
    with tabs[0]:
        st.subheader(f"Feature importance across models ({label})")
        imp = _importance_table(h)
        st.dataframe(imp.round(4), width="stretch")

        top = imp.head(15)
        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(top))
        width = 0.15
        palette = sns.color_palette("crest", len(MODEL_NAMES))
        for i, name in enumerate(MODEL_NAMES):
            ax.barh(x + (i - 2) * width, top[name], width, label=name, color=palette[i])
        ax.set_yticks(x)
        ax.set_yticklabels(top.index)
        ax.invert_yaxis()
        ax.set_xlabel("Normalised importance")
        ax.set_title(f"Top-15 features by importance ({label})")
        ax.legend(loc="lower right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ------------------------------------------------------------------ tab 2
    with tabs[1]:
        st.subheader(f"SHAP — XGBoost ({label})")
        xgb = h["models"]["XGBoost"]
        X = h["X_test"]
        y = h["test_y"]
        feat_names = h["feature_names"]

        explainer = shap.TreeExplainer(xgb)
        X_explain = X.sample(min(1000, len(X)), random_state=42)

        st.caption("Global beeswarm — each dot is a prediction; color is the feature value (red high, blue low).")
        sv = explainer.shap_values(X_explain)
        base = _base_value(explainer)
        exp = shap.Explanation(values=sv, base_values=base, data=X_explain.values, feature_names=feat_names)
        fig = plt.figure(figsize=(11, 9))
        shap.plots.beeswarm(exp, max_display=20, show=False)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("#### Single prediction — waterfall")
        if "interp_idx" not in st.session_state:
            st.session_state["interp_idx"] = int(np.random.default_rng(0).integers(len(X)))
        if st.button("🎲 Random sample"):
            st.session_state["interp_idx"] = int(np.random.default_rng().integers(len(X)))
        idx = st.session_state["interp_idx"]
        row = X.iloc[[idx]]
        sv_row = explainer.shap_values(row)
        pred = float(xgb.predict_proba(row)[0, 1])
        actual = int(y.iloc[idx])
        st.markdown(
            f"**Sample #{idx}** — predicted rain probability **{pred:.0%}**, "
            f"actual **{'Yes (rain)' if actual == 1 else 'No (no rain)'}**"
        )
        exp_row = shap.Explanation(values=np.asarray(sv_row)[0], base_values=base,
                                   data=row.iloc[0].values, feature_names=feat_names)
        fig = plt.figure(figsize=(10, 12))
        shap.plots.waterfall(exp_row, max_display=15, show=False)
        st.pyplot(fig)
        plt.close(fig)

    # ------------------------------------------------------------------ tab 3
    with tabs[2]:
        st.subheader(f"Per-station analysis ({label})")
        st.caption(f"Test-set F1 by station, using the best model ({h['best_model']}).")
        station = h["station"]
        st.dataframe(station, width="stretch", hide_index=True)

        if not station.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            data = station.sort_values("F1", ascending=True)
            ax.barh(data["Station"], data["F1"], color="#2563eb")
            ax.set_xlabel("F1 (test set)")
            ax.set_title(f"F1 by station ({label}, best model {h['best_model']})")
            ax.grid(axis="x", alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
