"""Data Preprocessing page renderer — cleaning, two-day target, split, IQR and imputation.

Reproduces the cleaning part of docs/02_preprocessing_avec_2jours_prediction.ipynb.
The later stages (feature engineering / encoding / scaling) live in src/feature_engineering.py.
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
from src.preprocessing import COLS_DROP, run_preprocessing


def render() -> None:
    st.set_page_config(page_title="Data Preprocessing — Rain in Australia", page_icon="🧹", layout="wide")
    require("eda")
    mark_done("data_preprocessing")
    sns.set_theme(style="whitegrid", context="notebook")

    res = run_preprocessing()

    st.title("🧹 Data Preprocessing")
    st.caption(
        "Reproduces the cleaning part of the « Pré-traitement » spec: missing values, "
        "column drop, two-day target, temporal split (train ≤ 2015-11-09), IQR detection, per-station "
        "imputation. Feature engineering, encoding and scaling are on the **Feature Engineering** page."
    )

    tabs = st.tabs(
        [
            "1️⃣ Missing Values & Column Drop",
            "2️⃣ Two-day Target & Split",
            "3️⃣ Outliers (IQR)",
            "4️⃣ Skewness, Distributions & Imputation",
            "5️⃣ Cleaned Data",
        ]
    )

    # ------------------------------------------------------------------ tab 1
    with tabs[0]:
        st.subheader("Missing values")
        st.dataframe(
            res["missing_pct"].rename("Missing %").reset_index().rename(columns={"index": "Variable"}),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            f"{res['rows_with_na']:,} / {res['rows_total']:,} rows contain at least one missing value "
            f"({res['rows_with_na']/res['rows_total']*100:.1f}%)."
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        pct_asc = res["missing_pct"].sort_values(ascending=True)
        ax.barh(pct_asc.index, pct_asc.values, color="steelblue")
        ax.axvline(x=10, color="orange", linestyle="--", linewidth=1.5, label="10% threshold")
        ax.axvline(x=40, color="red", linestyle="--", linewidth=1.5, label="40% threshold")
        ax.set_title("Percentage of missing values by variable")
        ax.set_xlabel("Missing values (%)")
        ax.set_ylabel("Variable")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Column drop")
        st.markdown(
            f"Before: **{res['shape_before'][0]:,} rows × {res['shape_before'][1]} cols** → "
            f"After: **{res['shape_after_drop'][0]:,} rows × {res['shape_after_drop'][1]} cols**"
        )
        st.write(f"Dropped columns: `{COLS_DROP}`")
        if res["leakage_col"]:
            st.info(
                f"Also dropped `{res['leakage_col']}` — a leakage column (next-day rainfall, by construction "
                f"`> 0 ⟺ RainTomorrow = Yes`) that the original notebook left in. Excluding it keeps the "
                f"pipeline consistent with the trained models."
            )

    # ------------------------------------------------------------------ tab 2
    with tabs[1]:
        st.subheader("Two-day target (RainInTwoDays)")
        st.markdown(
            "`RainInTwoDays` is built by shifting each station's `RainToday` by two days and joining on "
            "`(Location, Date)`. Rows missing either target are dropped."
        )
        c1, c2 = st.columns(2)
        c1.metric("Rows before dropping NaN targets", f"{res['rows_before_dropna']:,}")
        c2.metric("Rows after dropping NaN targets", f"{res['rows_after_dropna']:,}")

        st.subheader("Temporal split (train ≤ 2015-11-09, test after)")
        st.markdown(
            f"Training runs on `{res['train_date_min']}` → `{res['train_date_max']}`, testing on "
            f"`{res['test_date_min']}` → `{res['test_date_max']}`. A random split would put consecutive, "
            "nearly-identical days on both sides and inflate the scores."
        )
        c1, c2 = st.columns(2)
        c1.metric("X_train", f"{res['dim_train'][0]:,} rows × {res['dim_train'][1]} cols")
        c2.metric("X_test", f"{res['dim_test'][0]:,} rows × {res['dim_test'][1]} cols")

        classes = ["No", "Yes"]
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.arange(len(classes))
        w = 0.38
        b1 = ax.bar(x - w / 2, [res["prop_train"][c] for c in classes], w, label="Train", color=sns.color_palette("crest")[1])
        b2 = ax.bar(x + w / 2, [res["prop_test"][c] for c in classes], w, label="Test", color=sns.color_palette("crest")[4])
        for bars in (b1, b2):
            for b in bars:
                ax.annotate(f"{b.get_height():.2f}%", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10)
        ax.set_title("Class distribution after temporal split (train vs test)")
        ax.set_xlabel("Class (RainTomorrow)")
        ax.set_ylabel("Proportion (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend(title="Dataset")
        sns.despine()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ------------------------------------------------------------------ tab 3
    with tabs[2]:
        st.subheader("IQR outlier detection (computed on X_train only)")
        st.dataframe(res["iqr_table"], width="stretch", hide_index=True)
        st.caption(
            "No values are removed: meteorological extremes (storms, cyclones) are genuine and are kept. "
            "Rainfall shows ~20% 'outliers' simply because it is zero most days, so its IQR is extremely narrow."
        )

        fig, axes = plt.subplots(4, 3, figsize=(16, 12))
        for ax, col in zip(axes.ravel(), res["numeric_cols"]):
            sns.boxplot(y=res["X_train_num"][col], color=sns.color_palette("crest")[2], ax=ax)
            ax.set_title(col, fontsize=11)
            ax.set_ylabel(col)
        for j in range(len(res["numeric_cols"]), len(axes.ravel())):
            axes.ravel()[j].set_visible(False)
        fig.suptitle("Outliers by variable (train set)")
        sns.despine()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ------------------------------------------------------------------ tab 4
    with tabs[3]:
        st.subheader("Skewness (decides median vs mean imputation)")
        st.dataframe(res["skew_table"], width="stretch", hide_index=True)

        st.subheader("Distributions (train set)")
        fig, axes = plt.subplots(4, 3, figsize=(16, 12))
        for ax, col in zip(axes.ravel(), res["numeric_cols"]):
            sns.histplot(res["X_train_num"][col].dropna(), kde=True, color=sns.color_palette("crest")[2], ax=ax)
            ax.set_title(col, fontsize=11)
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
        for j in range(len(res["numeric_cols"]), len(axes.ravel())):
            axes.ravel()[j].set_visible(False)
        fig.suptitle("Distributions of numeric variables (train set)")
        sns.despine()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Imputation (9h/15h cross-fill + per-station stats)")
        st.dataframe(res["impute_table"], width="stretch", hide_index=True)
        st.caption(
            f"9h/15h cross-fill recovered {res['cross_fill_filled']:,} values. Remaining missing values after "
            f"per-station imputation — X_train: {res['train_na_after']}, X_test: {res['test_na_after']} (both must be 0)."
        )
        st.info(
            "A missing 9am/3pm reading is first deduced from the other same-day reading (correlation 0.86–0.96), "
            "then each remaining gap is filled with its station's statistic (median/mean/mode, fitted on the "
            "training set only), falling back to the global statistic when the station has none."
        )

    # ------------------------------------------------------------------ tab 5
    with tabs[4]:
        st.subheader("Cleaned data (post-imputation, 0 missing values)")
        st.markdown(
            "This is the output of the Data Preprocessing stage — a complete, NaN-free training set in "
            "physical units (before the trig / target encoding applied on the Feature Engineering page)."
        )
        c1, c2 = st.columns(2)
        c1.metric("X_train (cleaned)", f"{res['cleaned_shape'][0]:,} rows × {res['cleaned_shape'][1]} cols")
        c2.metric("Remaining missing values", "0")
        st.caption(f"Columns: {', '.join(res['cleaned_cols'])}")
        st.dataframe(res["cleaned_head"], width="stretch")
        st.info(
            "Data flow: `weatherAUS.csv` (loaded once) → **EDA** → this cleaned dataset → **Feature Engineering** "
            "→ **model training**. Every stage reads the same source of truth."
        )
