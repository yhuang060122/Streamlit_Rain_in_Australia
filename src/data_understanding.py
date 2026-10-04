"""Data Understanding page renderer — dataset overview, missing values, data types, target rate.

First stage of the CRISP-DM flow (before EDA), reusing the single source of truth load_raw().
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.gating import mark_done
from src.pipeline import load_raw

CAT_COLS = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "RainTomorrow"]


@st.cache_data(show_spinner="Loading dataset…")
def get_df() -> pd.DataFrame:
    return load_raw()


def _friendly_type(df: pd.DataFrame, col: str) -> str:
    if col == "Date":
        return "date"
    if pd.api.types.is_numeric_dtype(df[col]):
        return "numeric"
    return "categorical"


def render() -> None:
    st.set_page_config(page_title="Data Understanding — Rain in Australia", page_icon="🗂️", layout="wide")
    mark_done("data_understanding")

    df = get_df()

    st.title("🗂️ Data Understanding")
    st.caption(
        "First CRISP-DM stage — dataset overview, missing values, data types and the positive (rain) rate of the target. "
        "This is the same dataset consumed by every downstream stage."
    )

    tabs = st.tabs(
        [
            "1️⃣ Data Overview",
            "2️⃣ Missing Values",
            "3️⃣ Data Types",
            "4️⃣ Target Distribution",
        ]
    )

    # ------------------------------------------------------------------ tab 1
    with tabs[0]:
        st.subheader("Dataset overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{df.shape[1]}")
        c3.metric("Stations", f"{df['Location'].nunique()}")
        dates = pd.to_datetime(df["Date"])
        c4.metric("Date range", f"{dates.min():%Y-%m} – {dates.max():%Y-%m}")

        st.info(
            "Source: Kaggle **'Rain in Australia'** (`weatherAUS.csv`) — daily weather observations from the "
            "Australian Bureau of Meteorology (BOM), 49 stations, ~10 years (2007–2026 in this mirror)."
        )

        st.subheader("First rows")
        st.dataframe(df.head(10), width="stretch")

    # ------------------------------------------------------------------ tab 2
    with tabs[1]:
        st.subheader("Missing values by variable")
        missing = pd.DataFrame(
            {
                "Variable": df.columns,
                "Missing count": df.isna().sum().values,
                "Missing %": (df.isna().sum() / len(df) * 100).round(2).values,
            }
        ).sort_values("Missing %", ascending=False)
        st.dataframe(missing.reset_index(drop=True), width="stretch", hide_index=True)

        c1, c2 = st.columns(2)
        c1.metric("Rows with ≥1 missing value", f"{int(df.isna().any(axis=1).sum()):,}")
        c2.metric("Share of rows with missing", f"{df.isna().any(axis=1).sum()/len(df)*100:.1f}%")

        fig, ax = plt.subplots(figsize=(10, 8))
        pct_asc = missing.sort_values("Missing %", ascending=True)
        ax.barh(pct_asc["Variable"], pct_asc["Missing %"], color="indianred")
        ax.axvline(x=10, color="orange", linestyle="--", linewidth=1.5, label="10% threshold")
        ax.axvline(x=40, color="red", linestyle="--", linewidth=1.5, label="40% threshold")
        ax.set_title("Percentage of missing values by variable")
        ax.set_xlabel("Missing values (%)")
        ax.set_ylabel("Variable")
        ax.legend()
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Four variables exceed ~40% missing (Sunshine, Evaporation, Cloud3pm, Cloud9am) — these are dropped "
            "later because imputing ~40% of a column would amount to fabricating the variable."
        )

    # ------------------------------------------------------------------ tab 3
    with tabs[2]:
        st.subheader("Data types")
        types = pd.DataFrame(
            {
                "Column": df.columns,
                "Type": [_friendly_type(df, c) for c in df.columns],
                "Non-null": df.notna().sum().values,
                "Unique": df.nunique().values,
            }
        )
        st.dataframe(types, width="stretch", hide_index=True)

        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = [c for c in CAT_COLS if c in df.columns]
        c1, c2, c3 = st.columns(3)
        c1.metric("Numeric", len(num_cols))
        c2.metric("Categorical", len(cat_cols))
        c3.metric("Date", 1)

        st.markdown(f"**Numeric columns ({len(num_cols)}):** `{', '.join(num_cols)}`")
        st.markdown(f"**Categorical columns ({len(cat_cols)}):** `{', '.join(cat_cols)}`")

    # ------------------------------------------------------------------ tab 4
    with tabs[3]:
        st.subheader("Target distribution — RainTomorrow")
        yes = int((df["RainTomorrow"] == "Yes").sum())
        no = int((df["RainTomorrow"] == "No").sum())
        nan = int(df["RainTomorrow"].isna().sum())
        pos_rate = yes / (yes + no) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Rain tomorrow (Yes)", f"{yes:,}")
        c2.metric("No rain (No)", f"{no:,}")
        c3.metric("Positive rate", f"{pos_rate:.1f}%")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].pie([no, yes], labels=["No", "Yes"], autopct="%1.1f%%", startangle=90,
                    colors=["#f59e0b", "#2563eb"], explode=(0, 0.03))
        axes[0].set_title("RainTomorrow class split")
        bars = axes[1].bar(["No", "Yes"], [no, yes], color=["#f59e0b", "#2563eb"])
        for b, v in zip(bars, [no, yes]):
            axes[1].text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:,}",
                         ha="center", va="bottom", fontsize=10)
        axes[1].set_title("RainTomorrow counts")
        axes[1].set_ylabel("Count")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.warning(
            f"The target is imbalanced: only **{pos_rate:.1f}%** of days are positive (rain tomorrow). "
            "Evaluation must therefore use F1 / ROC-AUC / PR-AUC rather than accuracy — always predicting "
            "'No' would already reach ~78% accuracy."
        )
