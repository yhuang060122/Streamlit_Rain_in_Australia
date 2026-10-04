"""Modelling page renderer — reproduces docs/03_modelling_avec_prediction_a deux_jours_avec_reechantillonnage.ipynb.

Consumes the model-ready matrices produced by the Feature Engineering stage (src/preprocessing.run_preprocessing),
so the data flows: raw CSV → EDA → Data Preprocessing → Feature Engineering → Modelling.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

from src.gating import mark_done, require
from src.preprocessing import run_preprocessing

# Bump to invalidate cached modelling results whenever the modelling logic changes.
_MODELING_VERSION = "2026-10-04b"

MODEL_NAMES = ["LogisticRegression", "RandomForest", "XGBoost"]


def _weighted_models(spw: float) -> dict:
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(scale_pos_weight=spw, eval_metric="logloss", random_state=42, n_jobs=-1),
    }


def _unweighted_models() -> dict:
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(scale_pos_weight=1, eval_metric="logloss", random_state=42, n_jobs=-1),
    }


def _compare_table(modeles: dict, predictions: dict, y_true: pd.Series) -> pd.DataFrame:
    rows = []
    for name in modeles:
        y_pred, y_proba = predictions[name]
        rows.append({
            "Model": name,
            "F1": round(f1_score(y_true, y_pred), 4),
            "ROC_AUC": round(roc_auc_score(y_true, y_proba), 4),
            "Precision": round(precision_score(y_true, y_pred), 4),
            "Recall": round(recall_score(y_true, y_pred), 4),
            "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        })
    return pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner="Training models & running the resampling analysis… (first run takes ~1-2 minutes)")
def run_modeling(_cache_version: str = _MODELING_VERSION) -> dict:
    res = run_preprocessing(_cache_version)
    X_train, X_test = res["X_train_final"], res["X_test_final"]
    y_train, y_test = res["y_train"], res["y_test"]
    y_train2, y_test2 = res["y_train2"], res["y_test2"]

    # Subsample the training set to keep the demo fast. SMOTETomek's Tomek-links step
    # is effectively O(n²) on the full 211k rows, so we cap training at 50k rows.
    n_subsample = 50_000
    if len(X_train) > n_subsample:
        idx = np.random.default_rng(42).choice(len(X_train), n_subsample, replace=False)
        X_train = X_train.iloc[idx].reset_index(drop=True)
        y_train = y_train.iloc[idx].reset_index(drop=True)
        y_train2 = y_train2.iloc[idx].reset_index(drop=True)
    subsampled = len(X_train) < res["X_train_final"].shape[0]

    spw = float((y_train == 0).sum() / (y_train == 1).sum())
    spw2 = float((y_train2 == 0).sum() / (y_train2 == 1).sum())

    # ---- 1-day prediction: three weighted models ----
    modeles = _weighted_models(spw)
    pred1 = {}
    for name, m in modeles.items():
        m.fit(X_train, y_train)
        pred1[name] = (m.predict(X_test), m.predict_proba(X_test)[:, 1])
    compare1 = _compare_table(modeles, pred1, y_test)
    cm1 = {name: confusion_matrix(y_test, pred1[name][0]) for name in modeles}
    roc1 = {
        name: {"fpr": roc_curve(y_test, pred1[name][1])[0],
               "tpr": roc_curve(y_test, pred1[name][1])[1],
               "auc": roc_auc_score(y_test, pred1[name][1])}
        for name in modeles
    }

    # ---- resampling analysis: unweighted models × 5 strategies ----
    modeles_unw = _unweighted_models()
    strategies = {
        "No resampling": None,
        "RandomOverSampler": RandomOverSampler(random_state=42),
        "SMOTE": SMOTE(random_state=42),
        "SMOTETomek": SMOTETomek(random_state=42),
        "RandomUnderSampler": RandomUnderSampler(random_state=42),
    }
    results = []
    for strat, sampler in strategies.items():
        if sampler is None:
            Xs, ys = X_train, y_train
        else:
            Xs, ys = sampler.fit_resample(X_train, y_train)
        for name, m in modeles_unw.items():
            m.fit(Xs, ys)
            y_pred = m.predict(X_test)
            y_proba = m.predict_proba(X_test)[:, 1]
            results.append({
                "Strategy": strat,
                "Model": name,
                "F1": round(f1_score(y_test, y_pred), 4),
                "ROC_AUC": round(roc_auc_score(y_test, y_proba), 4),
                "Precision": round(precision_score(y_test, y_pred), 4),
                "Recall": round(recall_score(y_test, y_pred), 4),
            })
    df_results = pd.DataFrame(results)
    baseline = compare1[["Model", "F1", "ROC_AUC", "Precision", "Recall"]].copy()
    baseline["Strategy"] = "Baseline (weighted)"
    resample = pd.concat(
        [baseline[["Strategy", "Model", "F1", "ROC_AUC", "Precision", "Recall"]], df_results],
        ignore_index=True,
    )

    # ---- 2-day prediction: three weighted models ----
    modeles2 = _weighted_models(spw2)
    pred2 = {}
    for name, m in modeles2.items():
        m.fit(X_train, y_train2)
        pred2[name] = (m.predict(X_test), m.predict_proba(X_test)[:, 1])
    compare2 = _compare_table(modeles2, pred2, y_test2)
    cm2 = {name: confusion_matrix(y_test2, pred2[name][0]) for name in modeles2}
    roc2 = {
        name: {"fpr": roc_curve(y_test2, pred2[name][1])[0],
               "tpr": roc_curve(y_test2, pred2[name][1])[1],
               "auc": roc_auc_score(y_test2, pred2[name][1])}
        for name in modeles2
    }

    return {
        "scale_pos_weight": round(spw, 4),
        "scale_pos_weight2": round(spw2, 4),
        "n_train": int(len(X_train)),
        "subsampled": subsampled,
        "compare1": compare1,
        "cm1": cm1,
        "roc1": roc1,
        "resample": resample,
        "compare2": compare2,
        "cm2": cm2,
        "roc2": roc2,
    }


def _confusion_figure(cms: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, name in zip(axes, cms):
        sns.heatmap(cms[name], annot=True, fmt="d", cmap="crest", cbar=False, ax=ax)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted rain")
        ax.set_ylabel("Actual rain")
        ax.set_xticklabels(["No", "Yes"])
        ax.set_yticklabels(["No", "Yes"])
    fig.suptitle("Confusion matrices (test set)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def _roc_figure(roc_data: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = sns.color_palette("crest", len(roc_data))
    for (name, d), color in zip(roc_data.items(), colors):
        ax.plot(d["fpr"], d["tpr"], color=color, label=f"{name} (AUC = {d['auc']:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random")
    ax.set_title("ROC curves (test set)", fontsize=13, fontweight="bold")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    sns.despine()
    fig.tight_layout()
    return fig


def render() -> None:
    st.set_page_config(page_title="Modelling — Rain in Australia", page_icon="🤖", layout="wide")
    require("feature_engineering")
    mark_done("modeling")
    sns.set_theme(style="whitegrid", context="notebook")

    m = run_modeling()

    st.title("🤖 Modelling")
    st.caption(
        "Reproduces `docs/03_modelling_avec_prediction_a deux_jours_avec_reechantillonnage.ipynb`: three classifiers "
        "(Logistic Regression, Random Forest, XGBoost) for the 1-day and 2-day targets, plus a resampling analysis. "
        "The input is the model-ready matrix produced by the **Feature Engineering** stage."
    )

    if m["subsampled"]:
        st.info(
            f"Training uses a {m['n_train']:,}-row subsample of the training set to keep the resampling analysis "
            "responsive — SMOTETomek's Tomek-links step is effectively quadratic and would take many minutes on the "
            "full ~211k rows. Metrics here are indicative; the full-data XGBoost models are saved under `models/`."
        )

    tabs = st.tabs(
        [
            "1️⃣ Model Comparison (1-day)",
            "2️⃣ Resampling Analysis",
            "3️⃣ Two-day Prediction",
        ]
    )

    # ------------------------------------------------------------------ tab 1
    with tabs[0]:
        st.subheader("Three classifiers (1-day target: RainTomorrow)")
        st.caption(f"Class imbalance handled per model — XGBoost `scale_pos_weight = {m['scale_pos_weight']}`.")
        st.dataframe(m["compare1"], width="stretch", hide_index=True)

        st.markdown("#### Confusion matrices")
        st.pyplot(_confusion_figure(m["cm1"]))

        st.markdown("#### ROC curves")
        st.pyplot(_roc_figure(m["roc1"]))

    # ------------------------------------------------------------------ tab 2
    with tabs[1]:
        st.subheader("Effect of resampling on F1")
        st.markdown(
            "Five resampling strategies (plus the weighted baseline) × three classifiers, evaluated on the test set. "
            "Resampling mostly trades precision for recall; its effect on F1 is model-dependent."
        )
        st.dataframe(m["resample"], width="stretch", hide_index=True)

        fig, ax = plt.subplots(figsize=(11, 6))
        sns.barplot(data=m["resample"], x="Model", y="F1", hue="Strategy", ax=ax)
        ax.set_title("Effect of resampling on F1 score", fontsize=13, fontweight="bold")
        ax.set_xlabel("Model")
        ax.set_ylabel("F1")
        for container in ax.containers:
            ax.bar_label(container, fontsize=9, rotation=45, fmt="{:.3f}", label_type="edge")
        ax.legend(title="Method", bbox_to_anchor=(1.01, 1.05))
        sns.despine()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ------------------------------------------------------------------ tab 3
    with tabs[2]:
        st.subheader("Three classifiers (2-day target: RainInTwoDays)")
        st.caption(f"XGBoost `scale_pos_weight = {m['scale_pos_weight2']}`.")
        st.dataframe(m["compare2"], width="stretch", hide_index=True)

        st.markdown("#### Confusion matrices")
        st.pyplot(_confusion_figure(m["cm2"]))

        st.markdown("#### ROC curves")
        st.pyplot(_roc_figure(m["roc2"]))
