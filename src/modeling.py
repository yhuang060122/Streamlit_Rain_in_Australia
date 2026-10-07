"""Modelling page — spec-aligned modelling pipeline.

对齐「Modélisation」规范：
  - 5 个模型（逻辑回归 / 随机森林 / XGBoost / KNN / 神经网络）+ 朴素基线。
  - 时间交叉验证（TimeSeriesSplit, 5 折）评估。
  - 类别不平衡：none / 加权 / SMOTENC 在每折内比较（重采样只在训练部分）。
  - 超参数搜索（随机森林 / XGBoost / 神经网络；LR、KNN 保持默认）。
  - 决策阈值逐模型优化（交叉验证内扫阈值，最大化 F1）。
  - 最终评估：F1 / Precision / Recall / Accuracy / AUC + 分类报告 + 混淆矩阵 + 时间 + 模型大小。
  - 过拟合控制、按站点分析；J+2 完整重复同一流程。

为控制运行时间：训练集子采样 + 耗时步骤 @st.cache_data 缓存。
"""

from __future__ import annotations

import io
import joblib
import pickle
import time
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from src.gating import mark_done, require
from src.preprocessing import run_preprocessing

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Bump to invalidate cached modelling results whenever the modelling logic changes.
_MODELING_VERSION = "2026-10-06-modelling-spec"

CV_SPLITS = 5          # 时间交叉验证折数
SEARCH_CV_SPLITS = 3   # 超参数搜索内部折数（较小时控制时间）
SUBSAMPLE = 25_000     # 训练集子采样（控制时间）
THRESHOLD_GRID = np.round(np.linspace(0.10, 0.90, 17), 3)  # 决策阈值网格

MODEL_NAMES = ["LogisticRegression", "RandomForest", "XGBoost", "KNN", "NeuralNetwork"]
SEARCH_MODELS = ["RandomForest", "XGBoost", "NeuralNetwork"]  # 需超参数搜索的模型


# --------------------------------------------------------------------------- #
# 模型 & 不平衡处理
# --------------------------------------------------------------------------- #
def _base_model(name: str):
    """默认参数模型工厂（LR/KNN 保持默认，符合规范）。"""
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, random_state=42)
    if name == "RandomForest":
        return RandomForestClassifier(random_state=42, n_jobs=-1)
    if name == "XGBoost":
        return XGBClassifier(eval_metric="logloss", random_state=42, tree_method="hist", n_jobs=-1)
    if name == "KNN":
        return KNeighborsClassifier(n_jobs=-1)
    if name == "NeuralNetwork":
        return MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, alpha=1e-3,
                             early_stopping=True, n_iter_no_change=10, random_state=42)
    raise ValueError(f"Unknown model: {name}")


def _naive_model() -> DummyClassifier:
    return DummyClassifier(strategy="prior", random_state=42)


def _smote(X):
    """SMOTENC（混合数据 SMOTE）；本项目特征已全部三角/目标编码为数值，无类别列，
    故回退为 SMOTE（SMOTENC 在无类别特征时与之等价）。"""
    cat_idx = [i for i, c in enumerate(X.columns) if X[c].dtype in ("object", "category")]
    if cat_idx:
        return SMOTENC(categorical_features=cat_idx, random_state=42)
    return SMOTE(random_state=42)


def _fit_with_imbalance(model, X, y, method: str, spw: float):
    """按 method（none/weighted/smotenc）配置并 fit 模型，返回 fit 后的模型。"""
    X_use, y_use = X, y
    if method == "weighted":
        if hasattr(model, "class_weight"):
            model.set_params(class_weight="balanced")
        if hasattr(model, "scale_pos_weight"):
            model.set_params(scale_pos_weight=spw)
    elif method == "smotenc":
        X_use, y_use = _smote(X).fit_resample(X, y)
    model.fit(X_use, y_use)
    return model


def _make_model(name: str, method: str, spw: float, params: dict | None):
    """构造 + 配置不平衡 + 应用搜索到的超参数（未 fit）。"""
    model = _base_model(name)
    if params:
        model.set_params(**params)
    if method == "weighted":
        if hasattr(model, "class_weight"):
            model.set_params(class_weight="balanced")
        if hasattr(model, "scale_pos_weight"):
            model.set_params(scale_pos_weight=spw)
    return model


# --------------------------------------------------------------------------- #
# 阶段 1：不平衡处理选择（时间交叉验证）
# --------------------------------------------------------------------------- #
def _cv_evaluate(name: str, X, y, method: str, spw: float) -> float:
    """在 TimeSeriesSplit(CV_SPLITS) 上评估指定模型+不平衡方式，返回平均 F1。"""
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    scores = []
    for tr_idx, va_idx in tscv.split(X):
        model = _fit_with_imbalance(_base_model(name), X.iloc[tr_idx], y.iloc[tr_idx], method, spw)
        y_pred = model.predict(X.iloc[va_idx])
        scores.append(f1_score(y.iloc[va_idx], y_pred))
    return float(np.mean(scores))


def _select_imbalance(X, y, spw: float) -> pd.DataFrame:
    """对每个模型比较 none/weighted/smotenc，选平均 F1 最优的方式。"""
    rows = []
    for name in MODEL_NAMES:
        best_method, best_f1 = "none", -1.0
        per_method = {}
        for method in ["none", "weighted", "smotenc"]:
            f1 = _cv_evaluate(name, X, y, method, spw)
            per_method[method] = round(f1, 4)
            if f1 > best_f1:
                best_f1, best_method = f1, method
        rows.append({
            "Model": name,
            "None F1": per_method["none"],
            "Weighted F1": per_method["weighted"],
            "SMOTENC F1": per_method["smotenc"],
            "Selected": best_method,
            "CV F1": round(best_f1, 4),
        })
    return pd.DataFrame(rows).sort_values("CV F1", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 阶段 2：超参数搜索（RF / XGB / NN）
# --------------------------------------------------------------------------- #
_PARAM_GRIDS = {
    "RandomForest": {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 8, 16],
        "min_samples_leaf": [1, 3],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 400],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    },
    "NeuralNetwork": {
        "hidden_layer_sizes": [(32,), (64,), (64, 32)],
        "alpha": [1e-4, 1e-3],
    },
}


def _search_params(name: str, X, y, method: str, spw: float, n_iter: int = 6) -> tuple[dict, float]:
    """对单个模型做随机搜索（时间交叉验证），返回 (best_params, best_f1)。"""
    model = _make_model(name, method, spw, None)
    X_use, y_use = X, y
    if method == "smotenc":
        X_use, y_use = _smote(X).fit_resample(X, y)
    tscv = TimeSeriesSplit(n_splits=SEARCH_CV_SPLITS)
    search = RandomizedSearchCV(
        model, _PARAM_GRIDS[name], n_iter=n_iter, cv=tscv, scoring="f1",
        random_state=42, n_jobs=1, error_score="raise",
    )
    search.fit(X_use, y_use)
    return search.best_params_, float(search.best_score_)


# --------------------------------------------------------------------------- #
# 阶段 3：决策阈值优化（时间交叉验证 OOF）
# --------------------------------------------------------------------------- #
def _optimize_threshold(name: str, X, y, method: str, spw: float, params: dict | None) -> tuple[float, float]:
    """在时间交叉验证上收集 OOF 概率并扫阈值，返回 (best_threshold, best_f1)。"""
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    proba_parts, y_parts = [], []
    for tr_idx, va_idx in tscv.split(X):
        model = _make_model(name, method, spw, params)
        X_use, y_use = X.iloc[tr_idx], y.iloc[tr_idx]
        if method == "smotenc":
            X_use, y_use = _smote(X_use).fit_resample(X_use, y_use)
        model.fit(X_use, y_use)
        proba_parts.append(model.predict_proba(X.iloc[va_idx])[:, 1])
        y_parts.append(y.iloc[va_idx].to_numpy())
    proba = np.concatenate(proba_parts)
    y_true = np.concatenate(y_parts)

    best_t, best_f1 = 0.5, -1.0
    for t in THRESHOLD_GRID:
        f1 = f1_score(y_true, (proba >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return round(float(best_t), 3), round(best_f1, 4)


# --------------------------------------------------------------------------- #
# 阶段 4：最终评估 + 报告 + 时间 + 模型大小
# --------------------------------------------------------------------------- #
def _final_evaluate(name: str, method: str, spw: float, params: dict | None, threshold: float,
                    X_train, y_train, X_test, y_test):
    """在完整（子采样）训练集上 fit 最优配置，在测试集上用最优阈值评估。"""
    model = _make_model(name, method, spw, params)
    X_use, y_use = X_train, y_train
    if method == "smotenc":
        X_use, y_use = _smote(X_train).fit_resample(X_train, y_train)

    t0 = time.time()
    model.fit(X_use, y_use)
    train_time = time.time() - t0

    t1 = time.time()
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    pred_time = time.time() - t1

    metrics = {
        "F1": round(f1_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, proba), 4),
    }
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    size_mb = len(pickle.dumps(model)) / 1e6

    return {
        "model": model,
        "metrics": metrics,
        "report": report,
        "cm": cm,
        "train_time": round(train_time, 3),
        "pred_time": round(pred_time, 3),
        "size_mb": round(size_mb, 2),
        "proba": proba,
    }


# --------------------------------------------------------------------------- #
# 单 horizon 完整流程
# --------------------------------------------------------------------------- #
def _run_horizon(X_train, X_test, y_train, y_test, test_locations, spw: float) -> dict:
    # 子采样（控制时间）
    if len(X_train) > SUBSAMPLE:
        idx = np.random.default_rng(42).choice(len(X_train), SUBSAMPLE, replace=False)
        X_train = X_train.iloc[idx].reset_index(drop=True)
        y_train = y_train.iloc[idx].reset_index(drop=True)

    # 朴素基线
    naive = _naive_model()
    naive.fit(X_train, y_train)
    y_pred_n = naive.predict(X_test)
    proba_n = naive.predict_proba(X_test)[:, 1]
    naive_metrics = {
        "F1": round(f1_score(y_test, y_pred_n), 4),
        "Precision": round(precision_score(y_test, y_pred_n), 4),
        "Recall": round(recall_score(y_test, y_pred_n), 4),
        "Accuracy": round(accuracy_score(y_test, y_pred_n), 4),
        "AUC": round(roc_auc_score(y_test, proba_n), 4),
    }

    # 阶段 1：不平衡选择
    imbalance = _select_imbalance(X_train, y_train, spw)

    # 阶段 2：超参数搜索（RF/XGB/NN）
    best_params: dict = {m: {} for m in MODEL_NAMES}
    for name in SEARCH_MODELS:
        method = imbalance.loc[imbalance["Model"] == name, "Selected"].iloc[0]
        params, _ = _search_params(name, X_train, y_train, method, spw)
        best_params[name] = params

    # 阶段 3：决策阈值优化
    thresholds: dict = {}
    for name in MODEL_NAMES:
        method = imbalance.loc[imbalance["Model"] == name, "Selected"].iloc[0]
        t, _ = _optimize_threshold(name, X_train, y_train, method, spw, best_params[name])
        thresholds[name] = t

    # 阶段 4：最终评估
    final: dict = {}
    metric_rows = [{"Model": "Naive (prior)", **naive_metrics}]
    for name in MODEL_NAMES:
        method = imbalance.loc[imbalance["Model"] == name, "Selected"].iloc[0]
        final[name] = _final_evaluate(
            name, method, spw, best_params[name], thresholds[name], X_train, y_train, X_test, y_test
        )
        metric_rows.append({"Model": name, **final[name]["metrics"]})
    metrics_df = pd.DataFrame(metric_rows).sort_values("F1", ascending=False).reset_index(drop=True)

    # 过拟合控制：CV F1（阈值优化阶段的 OOF F1）vs 测试 F1
    overfitting_rows = []
    for name in MODEL_NAMES:
        cv_f1 = imbalance.loc[imbalance["Model"] == name, "CV F1"].iloc[0]
        test_f1 = final[name]["metrics"]["F1"]
        overfitting_rows.append({
            "Model": name,
            "CV F1": cv_f1,
            "Test F1": test_f1,
            "Gap (CV − Test)": round(cv_f1 - test_f1, 4),
        })
    overfitting_df = pd.DataFrame(overfitting_rows)

    # 按站点分析：用 F1 最优的模型，测试集按 Location 拆解
    best_model_name = metrics_df.loc[metrics_df["Model"] != "Naive (prior)", "Model"].iloc[0]
    best_proba = final[best_model_name]["proba"]
    best_thr = thresholds[best_model_name]
    best_pred = (best_proba >= best_thr).astype(int)
    station_rows = []
    loc_series = pd.Series(test_locations, index=y_test.index)
    for loc in sorted(loc_series.unique()):
        m = loc_series == loc
        y_loc, p_loc = y_test[m], best_pred[m]
        if len(y_loc) > 50 and y_loc.nunique() == 2:
            station_rows.append({
                "Station": loc,
                "N": int(len(y_loc)),
                "F1": round(f1_score(y_loc, p_loc), 4),
                "Pos rate": round(float(y_loc.mean()), 4),
            })
    if station_rows:
        station_df = pd.DataFrame(station_rows).sort_values("F1", ascending=False).reset_index(drop=True)
    else:
        station_df = pd.DataFrame(columns=["Station", "N", "F1", "Pos rate"])

    # ROC 数据（供渲染，避免在渲染层重复预测）
    roc = {}
    for n in MODEL_NAMES:
        proba = final[n]["proba"]
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc[n] = {"fpr": fpr, "tpr": tpr, "auc": round(roc_auc_score(y_test, proba), 4)}

    return {
        "imbalance": imbalance,
        "naive": naive_metrics,
        "best_params": best_params,
        "thresholds": thresholds,
        "metrics": metrics_df,
        "reports": {n: final[n]["report"] for n in MODEL_NAMES},
        "cms": {n: final[n]["cm"] for n in MODEL_NAMES},
        "roc": roc,
        "times": pd.DataFrame([
            {"Model": n, "Train (s)": final[n]["train_time"], "Predict (s)": final[n]["pred_time"],
             "Size (MB)": final[n]["size_mb"]}
            for n in MODEL_NAMES
        ]),
        "overfitting": overfitting_df,
        "station": station_df,
        "best_model": best_model_name,
        "models": {n: final[n]["model"] for n in MODEL_NAMES},
        "test_locations": loc_series,
        "test_y": y_test,
        "test_proba": {n: final[n]["proba"] for n in MODEL_NAMES},
        "X_test": X_test,
        "feature_names": list(X_test.columns),
    }


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Running the modelling pipeline… (first run takes a few minutes)")
def run_modeling(_cache_version: str = _MODELING_VERSION) -> dict:
    res = run_preprocessing(_cache_version)
    X_train, X_test = res["X_train_final"], res["X_test_final"]
    X_train2, X_test2 = res["X_train_final2"], res["X_test_final2"]
    y_train, y_test = res["y_train"], res["y_test"]
    y_train2, y_test2 = res["y_train2"], res["y_test2"]
    test_locations = res["test_locations"]

    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    spw2 = float((y_train2 == 0).sum() / max((y_train2 == 1).sum(), 1))

    j1 = _run_horizon(X_train, X_test, y_train, y_test, test_locations, spw)
    j2 = _run_horizon(X_train2, X_test2, y_train2, y_test2, test_locations, spw2)

    return {
        "j1": j1,
        "j2": j2,
        "scale_pos_weight": round(spw, 4),
        "scale_pos_weight2": round(spw2, 4),
        "n_train": int(len(X_train)),
        "subsampled": bool(len(X_train) > SUBSAMPLE),
    }


# --------------------------------------------------------------------------- #
# 渲染辅助
# --------------------------------------------------------------------------- #
def _serialize(obj) -> bytes:
    """把对象 joblib 序列化到内存字节（供 st.download_button 下载）。"""
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    return buf.getvalue()


def _cm_figure(cms: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, len(cms), figsize=(4 * len(cms), 4))
    if len(cms) == 1:
        axes = [axes]
    for ax, name in zip(axes, cms):
        sns.heatmap(cms[name], annot=True, fmt="d", cmap="crest", cbar=False, ax=ax)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels(["No", "Yes"])
        ax.set_yticklabels(["No", "Yes"])
    fig.suptitle("Confusion matrices (test set)", fontsize=13, fontweight="bold")
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


def _render_horizon(tabs, h: dict, title: str, spw: float) -> None:
    """渲染单个 horizon 的所有 tab。"""
    with tabs[0]:
        st.subheader(f"{title} — class imbalance selection (time CV, {CV_SPLITS} folds)")
        st.caption(
            f"For each model, `none` / `weighted` / `SMOTENC` are compared fold-by-fold on the training part only; "
            f"the best is selected. XGBoost `scale_pos_weight = {spw}` when weighted."
        )
        st.dataframe(h["imbalance"], width="stretch", hide_index=True)

    with tabs[1]:
        st.subheader(f"{title} — hyperparameter search (RF / XGBoost / NeuralNetwork)")
        st.caption("LR and KNN keep default parameters. Search uses randomized CV over a time split.")
        params_rows = [
            {"Model": m, "Best params": str(h["best_params"][m]), "Threshold": h["thresholds"][m]}
            for m in MODEL_NAMES
        ]
        st.dataframe(pd.DataFrame(params_rows), width="stretch", hide_index=True)
        st.markdown("**Decision thresholds** (optimised on the time CV, maximising F1):")
        st.dataframe(
            pd.DataFrame([{"Model": m, "Threshold": h["thresholds"][m]} for m in MODEL_NAMES]),
            width="stretch", hide_index=True,
        )

    with tabs[2]:
        st.subheader(f"{title} — final evaluation (test set)")
        st.dataframe(h["metrics"], width="stretch", hide_index=True)
        st.markdown("#### Confusion matrices")
        st.pyplot(_cm_figure(h["cms"]))
        st.markdown("#### ROC curves")
        st.pyplot(_roc_figure(h["roc"]))

    with tabs[3]:
        st.subheader(f"{title} — classification reports")
        model = st.selectbox("Model", MODEL_NAMES, key=f"report_{title}")
        report = h["reports"][model]
        st.dataframe(
            pd.DataFrame(report).T.round(4), width="stretch",
        )

    with tabs[4]:
        st.subheader(f"{title} — overfitting & costs")
        st.markdown("**Overfitting control** (CV F1 vs test F1 — a large gap signals overfitting):")
        st.dataframe(h["overfitting"], width="stretch", hide_index=True)
        st.markdown("**Training / prediction time & model size:**")
        st.dataframe(h["times"], width="stretch", hide_index=True)

    with tabs[5]:
        st.subheader(f"{title} — per-station analysis")
        st.caption(f"Test-set F1 by station, using the best model ({h['best_model']}).")
        st.dataframe(h["station"], width="stretch", hide_index=True)


def render() -> None:
    st.set_page_config(page_title="Modelling — Rain in Australia", page_icon="🤖", layout="wide")
    require("feature_engineering")
    mark_done("modeling")
    sns.set_theme(style="whitegrid", context="notebook")

    m = run_modeling()

    st.title("🤖 Modelling")
    st.caption(
        "Five classifiers (Logistic Regression, Random Forest, XGBoost, KNN, Neural Network) compared to a naive "
        "baseline, with temporal cross-validation, class-imbalance selection (weighting vs SMOTENC), hyperparameter "
        "search, threshold optimisation, and a full evaluation — repeated for the J+1 and J+2 targets."
    )

    if m["subsampled"]:
        st.info(
            f"Training uses a {m['n_train']:,}-row subsample to keep the pipeline responsive. "
            "Metrics are indicative; the full-data XGBoost models live under `models/`."
        )

    j1, j2 = m["j1"], m["j2"]

    st.header("🌦️ J+1 — RainTomorrow")
    tabs1 = st.tabs(["1️⃣ Imbalance", "2️⃣ Hyperparameters", "3️⃣ Evaluation", "4️⃣ Reports", "5️⃣ Overfitting & Costs", "6️⃣ Per-station"])
    _render_horizon(tabs1, j1, "J+1", m["scale_pos_weight"])

    st.header("🌧️ J+2 — RainInTwoDays")
    tabs2 = st.tabs(["1️⃣ Imbalance", "2️⃣ Hyperparameters", "3️⃣ Evaluation", "4️⃣ Reports", "5️⃣ Overfitting & Costs", "6️⃣ Per-station"])
    _render_horizon(tabs2, j2, "J+2", m["scale_pos_weight2"])

    # ------------------------------------------------------------------ 保存模型
    with st.expander("💾 Save a trained model", expanded=False):
        st.markdown(
            "Serialize a trained model — plus its threshold, features, imbalance method and metrics — "
            "to a `.joblib` file you can download and reuse."
        )
        c1, c2, c3 = st.columns(3)
        save_horizon = c1.selectbox("Target", ["J+1", "J+2"], key="save_horizon")
        h_save = m["j1"] if save_horizon == "J+1" else m["j2"]
        best = h_save["best_model"]
        options = [f"Best model ({best})"] + [n for n in MODEL_NAMES]
        save_choice = c2.selectbox("Model", options, key="save_model")

        name = best if save_choice.startswith("Best") else save_choice
        model = h_save["models"][name]
        threshold = h_save["thresholds"][name]
        method = h_save["imbalance"].loc[h_save["imbalance"]["Model"] == name, "Selected"].iloc[0]
        f1 = h_save["metrics"].loc[h_save["metrics"]["Model"] == name, "F1"].iloc[0]
        c3.metric("Test F1", f"{f1:.4f}")

        st.caption(
            f"**{name}** · {save_horizon} · imbalance `{method}` · threshold `{threshold}` · "
            f"{len(h_save['feature_names'])} features."
        )

        payload = {
            "model": model,
            "model_name": name,
            "horizon": save_horizon,
            "threshold": threshold,
            "imbalance": method,
            "best_params": h_save["best_params"].get(name, {}),
            "features": h_save["feature_names"],
            "metrics": h_save["metrics"].loc[h_save["metrics"]["Model"] == name].to_dict("records")[0],
        }
        st.download_button(
            "⬇️ Download .joblib",
            data=_serialize(payload),
            file_name=f"{name}_{save_horizon}.joblib",
            mime="application/octet-stream",
            key="dl_model",
        )
