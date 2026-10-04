"""Feature Engineering page renderer — feature engineering, encoding and scaling.

Reproduces the later stages of docs/02_preprocessing_avec_2jours_prediction.ipynb.
The cleaning stage lives in src/data_processing.py.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.gating import mark_done, require
from src.preprocessing import run_preprocessing


def render() -> None:
    st.set_page_config(page_title="Feature Engineering — Rain in Australia", page_icon="🔧", layout="wide")
    require("data_preprocessing")
    mark_done("feature_engineering")
    sns.set_theme(style="whitegrid", context="notebook")

    res = run_preprocessing()

    st.title("🔧 Feature Engineering")
    st.caption(
        "Reproduces the feature-engineering stage of `docs/02_preprocessing_avec_2jours_prediction.ipynb`: "
        "new variables, categorical encoding, and scaling. The cleaning stage is on the **Data Preprocessing** page."
    )

    tabs = st.tabs(
        [
            "1️⃣ Feature Engineering",
            "2️⃣ Engineered Data",
            "3️⃣ Encoding",
            "4️⃣ Correlation, Normality & Scaling",
        ]
    )

    # ------------------------------------------------------------------ tab 1
    with tabs[0]:
        st.subheader("Feature engineering (row-wise, no fitted statistics)")
        st.markdown(
            "New variables: `Month`, `Season` (Southern-hemisphere), and intra-day differences "
            "`Temp_diff`, `Humidity_diff`, `Pressure_diff` (3pm − 9am)."
        )
        fig, ax = plt.subplots(figsize=(9, 6))
        corr = res["feature_corr"]
        colors = sns.color_palette("crest", len(corr))
        bars = ax.bar(corr.index, corr.values, color=colors)
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 3 if h >= 0 else -12), textcoords="offset points",
                        ha="center", va="bottom" if h >= 0 else "top", fontsize=10)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title("Correlation of new variables with the target")
        ax.set_xlabel("New variable")
        ax.set_ylabel("Pearson correlation")
        sns.despine()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "`Month` shows near-zero linear correlation — seasonality is non-monotonic, so its signal is "
            "carried by `Season` and only usable by non-linear models."
        )

    # ------------------------------------------------------------------ tab 2
    with tabs[1]:
        st.subheader("Engineered data (after feature engineering)")
        st.markdown(
            "This is the dataset right after the new variables are added — before categorical encoding and "
            "scaling. It still keeps `Location`, `Date` and the raw physical columns."
        )
        c1, c2 = st.columns(2)
        c1.metric("X_train (feature-engineered)", f"{res['fe_shape'][0]:,} rows × {res['fe_shape'][1]} cols")
        c2.metric("New variables added", len(res["fe_new_cols"]))
        st.caption(f"New variables: `{', '.join(res['fe_new_cols'])}`")
        st.dataframe(res["fe_head"], width="stretch")

    # ------------------------------------------------------------------ tab 3
    with tabs[2]:
        st.subheader("Encoding")
        st.markdown(
            "- `RainToday` → 0/1\n"
            "- `Location` → target encoding (rain rate per station, fitted on train only)\n"
            "- `Date` → dropped (captured by `Month`/`Season`)\n"
            "- Wind direction + `Season` → one-hot (N−1)"
        )
        if res["unseen_locations"] > 0:
            st.warning(f"{res['unseen_locations']} test rows had an unseen Location — filled with the global rate ({res['global_rate']}).")
        else:
            st.caption("No unseen locations in the test set (all stations seen in training).")
        st.metric("Encoded feature matrix (X_train)", f"{res['enc_dim'][0]:,} rows × {res['enc_dim'][1]} cols")

        st.subheader("Location target encoding (rain-tomorrow rate per station)")
        loc = res["loc_rate_pct"]
        cmap = sns.color_palette("crest", as_cmap=True)
        norm = plt.Normalize(loc.min(), loc.max())
        data = loc.iloc[::-1]
        fig, ax = plt.subplots(figsize=(10, 14))
        ax.barh(data.index, data.values, color=cmap(norm(data.values)))
        ax.set_title("Rain-tomorrow rate by station (target encoding, train set)")
        ax.set_xlabel("Rain-tomorrow rate (%)")
        ax.set_ylabel("Station")
        sns.despine()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top 5 wettest**")
            st.dataframe(loc.head(5).rename("rate %"), width="stretch")
        with c2:
            st.markdown("**Top 5 driest**")
            st.dataframe(loc.tail(5).rename("rate %"), width="stretch")

    # ------------------------------------------------------------------ tab 3
    with tabs[3]:
        st.subheader("Pearson vs Spearman correlation with target")
        st.dataframe(res["ps_table"], width="stretch", hide_index=True)
        fig, ax = plt.subplots(figsize=(14, 7))
        order = res["ps_table"]["Variable"].tolist()
        x = np.arange(len(order))
        w = 0.4
        c1, c2 = sns.color_palette("crest")[1], sns.color_palette("crest")[4]
        ax.bar(x - w / 2, res["ps_table"]["Pearson"], w, label="Pearson", color=c1)
        ax.bar(x + w / 2, res["ps_table"]["Spearman"], w, label="Spearman", color=c2)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title("Correlation with target: Pearson vs Spearman")
        ax.set_xlabel("Variable")
        ax.set_ylabel("Correlation")
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=90)
        ax.legend()
        sns.despine()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Jarque-Bera normality test (pre-scaling)")
        st.dataframe(res["jb_table"], width="stretch", hide_index=True)
        st.caption(
            "At large N the p-value of any normality test is essentially zero, so it is *not* used as a "
            "decision criterion. The JB statistic (lower = closer to normal) and QQ-plots drive the choice "
            "of StandardScaler vs RobustScaler."
        )

        st.subheader("Scaling (fitted on X_train only)")
        st.markdown("`StandardScaler` for near-normal variables, `RobustScaler` for skewed / heavy-tailed ones.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**StandardScaler — mean (≈ 0)**")
            st.dataframe(res["std_mean"].rename("mean").reset_index().rename(columns={"index": "Variable"}), width="stretch", hide_index=True)
        with c2:
            st.markdown("**RobustScaler — median (≈ 0)**")
            st.dataframe(res["rob_median"].rename("median").reset_index().rename(columns={"index": "Variable"}), width="stretch", hide_index=True)
        st.metric("Final feature matrix (X_train)", f"{res['final_dim'][0]:,} rows × {res['final_dim'][1]} cols")
