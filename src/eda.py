"""EDA page renderer — reproduces the exploration in docs/01_exploration.ipynb.

The Streamlit entry point lives in pages/1_EDA.py, which simply calls ``render()``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.stats import chi2_contingency

from src.gating import mark_done, require
from src.pipeline import COLS_DROP_CORR, COLS_DROP_HIGH_MISSING, CUTOFF_DATE, get_season, load_raw

NUM_COLS = [
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
    "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm",
]
CAT_COLS = ["RainToday", "WindGustDir", "WindDir9am", "WindDir3pm"]


@st.cache_data(show_spinner="Loading dataset…")
def get_df() -> pd.DataFrame:
    return load_raw()


def render() -> None:
    st.set_page_config(page_title="EDA — Rain in Australia", page_icon="📊", layout="wide")
    require("data_understanding")
    mark_done("eda")
    sns.set_style("whitegrid")

    df = get_df()

    st.title("📊 Exploratory Data Analysis")
    st.caption(
        "Reproduces `docs/01_exploration.ipynb` — dataset overview, distributions, correlations, "
        "categorical analysis and boxplots."
    )

    tabs = st.tabs(
        [
            "1️⃣ Dataset Overview",
            "2️⃣ Distributions",
            "3️⃣ Correlations",
            "4️⃣ Categorical & Geography",
            "5️⃣ Boxplots vs Target",
            "6️⃣ Pré-traitement Decisions",
        ]
    )

    # ------------------------------------------------------------------ tab 1
    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{df.shape[1]}")
        c3.metric("Stations", f"{df['Location'].nunique()}")

        st.subheader("Column types")
        st.dataframe(
            pd.DataFrame({"dtype": df.dtypes.astype(str)}).reset_index().rename(columns={"index": "Column"}),
            width="stretch",
            hide_index=True,
        )

        st.subheader("Descriptive statistics (numeric)")
        st.dataframe(df.describe(), width="stretch")

        st.subheader("Missing values")
        missing = pd.DataFrame(
            {
                "Variable": df.columns,
                "Missing count": df.isna().sum().values,
                "Missing %": (df.isna().sum() / len(df) * 100).round(2).values,
            }
        )
        missing = missing[missing["Missing count"] > 0].sort_values("Missing %", ascending=False)
        st.dataframe(missing.reset_index(drop=True), width="stretch", hide_index=True)
        st.caption(
            f"Rows with at least one missing value: {int(df.isna().any(axis=1).sum()):,} "
            f"({df.isna().any(axis=1).sum()/len(df)*100:.1f}%)"
        )

        m2 = missing.sort_values("Missing %", ascending=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(m2["Variable"], m2["Missing %"], color="indianred")
        ax.axvline(x=10, color="orange", linestyle="--", linewidth=1, label="10% threshold")
        ax.axvline(x=40, color="red", linestyle="--", linewidth=1, label="40% threshold")
        ax.set_xlabel("Percentage of missing values (%)")
        ax.set_ylabel("Variable")
        ax.set_title("Percentage of missing values by variable")
        ax.legend()
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ------------------------------------------------------------------ tab 2
    with tabs[1]:
        st.subheader("Temperatures")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for ax, col, color, title in [
            (axes[0, 0], "MinTemp", "skyblue", "Min temperature (MinTemp)"),
            (axes[0, 1], "MaxTemp", "salmon", "Max temperature (MaxTemp)"),
            (axes[1, 0], "Temp9am", "lightgreen", "Temperature at 9am (Temp9am)"),
            (axes[1, 1], "Temp3pm", "orange", "Temperature at 3pm (Temp3pm)"),
        ]:
            sns.histplot(df[col], bins=50, color=color, kde=True, ax=ax)
            ax.set_title(title)
            ax.set_xlabel("Temperature (°C)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Rainfall (log scale + rainy days only)")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(df["Rainfall"], bins=50, color="steelblue", kde=False, ax=axes[0])
        axes[0].set_yscale("log")
        axes[0].set_title("Rainfall distribution — log scale")
        axes[0].set_xlabel("Rainfall (mm)")
        axes[0].set_ylabel("Frequency (log)")
        sns.histplot(df["Rainfall"][df["Rainfall"] > 0], bins=50, color="steelblue", kde=True, ax=axes[1])
        axes[1].set_title("Rainfall distribution — rainy days only")
        axes[1].set_xlabel("Rainfall (mm)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Evaporation & Sunshine")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(df["Evaporation"], bins=50, color="coral", kde=True, ax=axes[0])
        axes[0].set_title("Evaporation")
        axes[0].set_xlabel("Evaporation (mm)")
        sns.histplot(df["Sunshine"], bins=50, color="gold", kde=True, ax=axes[1])
        axes[1].set_title("Sunshine")
        axes[1].set_xlabel("Hours of sunshine")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Wind")
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        for ax, col, color, title in [
            (axes[0], "WindGustSpeed", "teal", "Wind gust speed (WindGustSpeed)"),
            (axes[1], "WindSpeed9am", "lightblue", "Wind speed 9am (WindSpeed9am)"),
            (axes[2], "WindSpeed3pm", "skyblue", "Wind speed 3pm (WindSpeed3pm)"),
        ]:
            sns.histplot(df[col], bins=50, color=color, kde=True, ax=ax)
            ax.set_title(title)
            ax.set_xlabel("Speed (km/h)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Humidity")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(df["Humidity9am"], bins=50, color="mediumseagreen", kde=True, ax=axes[0])
        axes[0].set_title("Humidity at 9am (Humidity9am)")
        axes[0].set_xlabel("Humidity (%)")
        sns.histplot(df["Humidity3pm"], bins=50, color="lightseagreen", kde=True, ax=axes[1])
        axes[1].set_title("Humidity at 3pm (Humidity3pm)")
        axes[1].set_xlabel("Humidity (%)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Atmospheric pressure")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(df["Pressure9am"], bins=50, color="darkviolet", kde=True, ax=axes[0])
        axes[0].set_title("Pressure at 9am (Pressure9am)")
        axes[0].set_xlabel("Pressure (hPa)")
        sns.histplot(df["Pressure3pm"], bins=50, color="mediumblue", kde=True, ax=axes[1])
        axes[1].set_title("Pressure at 3pm (Pressure3pm)")
        axes[1].set_xlabel("Pressure (hPa)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Cloud cover (oktas)")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, col, color, title in [
            (axes[0], "Cloud9am", "slategray", "Cloud cover at 9am (Cloud9am)"),
            (axes[1], "Cloud3pm", "darkgray", "Cloud cover at 3pm (Cloud3pm)"),
        ]:
            sns.histplot(df[col], bins=10, color=color, kde=False, discrete=True, ax=ax)
            ax.set_title(title)
            ax.set_xlabel("Cover (oktas)")
            ax.set_xticks(range(0, 10))
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ------------------------------------------------------------------ tab 3
    with tabs[2]:
        st.subheader("Correlation matrix (numeric variables)")
        corr = df[NUM_COLS].corr()
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax,
            annot_kws={"size": 7},
        )
        ax.set_title("Correlation matrix — numeric variables")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Correlation with the target (RainTomorrow)")
        target_num = df["RainTomorrow"].map({"Yes": 1, "No": 0})
        corr_target = df[NUM_COLS].corrwith(target_num).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["#2ca02c" if x > 0 else "#d62728" for x in corr_target.values]
        ax.barh(corr_target.index, corr_target.values, color=colors)
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
        ax.set_xlabel("Correlation with RainTomorrow")
        ax.set_ylabel("Variable")
        ax.set_title("Correlation of numeric variables with RainTomorrow")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(
            corr_target.rename("correlation").round(4).reset_index().rename(columns={"index": "Variable"}),
            width="stretch",
            hide_index=True,
        )

        with st.expander("⚠️ Leakage check — RISK_MM"):
            st.markdown(
                "The raw Rattle dataset includes a `RISK_MM` column (next-day rainfall in mm) which is, by "
                "construction, perfectly aligned with `RainTomorrow` (`RISK_MM > 0 ⟺ RainTomorrow = Yes`). "
                "It must be excluded from modelling — hence the training pipeline drops it."
            )
            risk = df["RISK_MM"] if "RISK_MM" in df.columns else None
            if risk is not None:
                risk_num = (risk > 0).astype(float).where(risk.notna())
                st.metric("Correlation of RISK_MM with RainTomorrow", f"{risk_num.corr(target_num):.4f}")
            else:
                st.info("`RISK_MM` is absent from the current dataset — nothing to check.")

    # ------------------------------------------------------------------ tab 4
    with tabs[3]:
        st.subheader("Rain frequency by station (top 10 wettest / driest)")
        target_num = df["RainTomorrow"].map({"Yes": 1, "No": 0})
        loc_rain = (df["RainTomorrow"] == "Yes").groupby(df["Location"]).mean() * 100
        top10_wet = loc_rain.sort_values(ascending=False).head(10)
        top10_dry = loc_rain.sort_values(ascending=True).head(10)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        top10_wet.sort_values().plot(kind="barh", color="steelblue", ax=axes[0])
        axes[0].set_title("Top 10 wettest stations")
        axes[0].set_xlabel("% of days with rain tomorrow")
        axes[0].grid(axis="x", alpha=0.3)
        top10_dry.sort_values().plot(kind="barh", color="coral", ax=axes[1])
        axes[1].set_title("Top 10 driest stations")
        axes[1].set_xlabel("% of days with rain tomorrow")
        axes[1].grid(axis="x", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Categorical variables — correlation with target (one-hot)")
        encoded = pd.get_dummies(df[CAT_COLS], drop_first=False)
        encoded["target"] = target_num
        cat_corr = encoded.drop("target", axis=1).corrwith(encoded["target"]).sort_values(ascending=False)
        top15 = cat_corr.head(15)
        bottom15 = cat_corr.tail(15)
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        top15.plot(kind="barh", color="green", ax=axes[0])
        axes[0].set_title("Top 15 categories — positive correlation with RainTomorrow")
        axes[0].grid(axis="x", alpha=0.3)
        bottom15.plot(kind="barh", color="red", ax=axes[1])
        axes[1].set_title("Top 15 categories — negative correlation with RainTomorrow")
        axes[1].grid(axis="x", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Chi-2 independence test & Cramér's V")
        variables_cat = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"]

        def interpret_cramer(v):
            if v >= 0.8:
                return "Suspected multicollinearity"
            if v >= 0.5:
                return "High"
            if v >= 0.3:
                return "Medium"
            return "Low"

        results = []
        for var in variables_cat:
            crosstab = pd.crosstab(df[var], df["RainTomorrow"])
            stat, p, dof, _ = chi2_contingency(crosstab)
            n = crosstab.values.sum()
            v = float(np.sqrt(stat / n))
            results.append({
                "Variable": var,
                "N": n,
                "Chi2": round(stat, 2),
                "dof": dof,
                "p-value": f"{p:.2e}",
                "Cramér's V": round(v, 4),
                "Interpretation": interpret_cramer(v),
            })
        res_df = pd.DataFrame(results).sort_values("Cramér's V", ascending=False).reset_index(drop=True)
        st.dataframe(res_df, width="stretch", hide_index=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_data = res_df.sort_values("Cramér's V", ascending=True)
        ax.barh(plot_data["Variable"], plot_data["Cramér's V"], color="steelblue")
        ax.axvline(x=0.1, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="Low (0.1)")
        ax.axvline(x=0.3, color="orange", linestyle="--", linewidth=1, alpha=0.6, label="Medium (0.3)")
        ax.axvline(x=0.5, color="red", linestyle="--", linewidth=1, alpha=0.6, label="High (0.5)")
        ax.set_title("Cramér's V by categorical variable (dependence with RainTomorrow)")
        ax.set_xlabel("Cramér's V")
        ax.set_ylabel("Variable")
        ax.legend()
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Rain-tomorrow rate by category")
        cible = df["RainTomorrow"].map({"Yes": 1.0, "No": 0.0})
        df_taux = df.assign(cible_pluie=cible)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for ax, var in zip(axes.ravel(), ["RainToday", "WindGustDir", "WindDir9am", "WindDir3pm"]):
            taux = df_taux.groupby(var)["cible_pluie"].mean() * 100
            taux = taux.sort_values(ascending=False)
            ax.bar(taux.index.astype(str), taux.values, color="steelblue")
            ax.set_title(f"Rain-tomorrow rate by {var}")
            ax.set_xlabel(var)
            ax.set_ylabel("% RainTomorrow = Yes")
            ax.tick_params(axis="x", rotation=90)
            ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ------------------------------------------------------------------ tab 5
    with tabs[4]:
        st.subheader("Numeric variables vs RainTomorrow (boxplots)")
        df_box = df[df["RainTomorrow"].isin(["No", "Yes"])]
        palette = {"No": sns.color_palette("crest")[1], "Yes": sns.color_palette("crest")[4]}
        fig, axes = plt.subplots(4, 4, figsize=(18, 18))
        for ax, var in zip(axes.ravel(), NUM_COLS):
            sns.boxplot(x="RainTomorrow", y=var, hue="RainTomorrow", data=df_box,
                        palette=palette, order=["No", "Yes"], legend=False, ax=ax)
            ax.set_title(var)
            ax.set_xlabel("RainTomorrow")
            ax.set_ylabel(var)
        fig.suptitle("Distribution of numeric variables by target RainTomorrow", fontsize=14, fontweight="bold")
        sns.despine()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ------------------------------------------------------------------ tab 6
    with tabs[5]:
        st.subheader("Pré-traitement decisions (mapped to the spec)")
        st.caption(
            "Each block below is the EDA justification for a step in the « Pré-traitement » spec, "
            "implemented in `src/pipeline.py`."
        )

        dates = pd.to_datetime(df["Date"])

        # ---- 1. temporal split ----
        st.markdown("### 1. Temporal split")
        cutoff = pd.Timestamp(CUTOFF_DATE)
        n_train, n_test = int((dates <= cutoff).sum()), int((dates > cutoff).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Data range", f"{dates.min():%Y-%m-%d} → {dates.max():%Y-%m-%d}")
        c2.metric("Train (≤ cutoff)", f"{n_train:,}")
        c3.metric("Test (> cutoff)", f"{n_test:,}")
        st.info(
            f"Cutoff = **{CUTOFF_DATE}**. For daily time series, validation must be temporal: a random split "
            "would put consecutive, nearly-identical days on both sides, so the model gets tested on days it "
            "almost saw in training, and its scores are overestimated."
        )
        cum = dates.value_counts().sort_index().cumsum()
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(cum.index, cum.values, color="#2563eb")
        ax.axvline(cutoff, color="red", linestyle="--", linewidth=1.5, label=f"cutoff {CUTOFF_DATE}")
        ax.set_title("Cumulative rows over time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative rows")
        ax.legend()
        sns.despine()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ---- 2. high-missingness ----
        st.markdown("### 2. High-missingness columns (>20%) — dropped")
        missing_pct = (df.isna().mean() * 100).round(2)
        high = missing_pct[missing_pct > 20].sort_values(ascending=False)
        st.dataframe(
            high.rename("Missing %").reset_index().rename(columns={"index": "Variable"}),
            width="stretch", hide_index=True,
        )
        st.markdown(f"→ Dropped columns: `{', '.join(COLS_DROP_HIGH_MISSING)}`")

        # ---- 3. correlated pairs ----
        st.markdown("### 3. Highly correlated pairs (|Pearson| > 0.8) — resolved by dropping 3 columns")
        corr = df[NUM_COLS].corr()
        cols = list(NUM_COLS)
        pairs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                v = corr.iloc[i, j]
                if abs(v) > 0.8:
                    pairs.append({"Pair": f"{cols[i]} ↔ {cols[j]}", "Pearson": round(float(v), 4)})
        pairs = sorted(pairs, key=lambda x: -abs(x["Pearson"]))
        if pairs:
            st.dataframe(pd.DataFrame(pairs), width="stretch", hide_index=True)
        else:
            st.caption("No pair exceeds |0.8|.")
        st.markdown(
            f"→ Dropped columns: `{', '.join(COLS_DROP_CORR)}`. Dropping these three removes every pair above "
            "(`MaxTemp`↔`Temp3pm`/`Temp9am`, `Pressure9am`↔`Pressure3pm`, `MinTemp`↔`Temp9am`, `Temp9am`↔`Temp3pm`)."
        )

        # ---- 4. leakage ----
        st.markdown("### 4. Leakage column RISK_MM")
        if "RISK_MM" in df.columns:
            st.warning("`RISK_MM` is present — it is excluded from modelling (leakage).")
        else:
            st.success("`RISK_MM` is absent from the dataset — nothing to exclude.")

        # ---- 5. cyclical encoding ----
        st.markdown("### 5. Cyclical encoding (month / season / wind → sin/cos)")
        df_t = df.copy()
        df_t["Month"] = dates.dt.month
        df_t["Season"] = df_t["Month"].map(get_season)
        month_rate = (df_t["RainTomorrow"] == "Yes").groupby(df_t["Month"]).mean() * 100
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(month_rate.index, month_rate.values, marker="o", color="#2563eb")
        ax.set_title("Rain-tomorrow rate by month (cyclical pattern)")
        ax.set_xlabel("Month")
        ax.set_ylabel("% RainTomorrow = Yes")
        ax.set_xticks(range(1, 13))
        ax.grid(alpha=0.3)
        sns.despine()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Month is cyclical (December sits next to January), so it is encoded as sin/cos instead of a raw "
            "1–12 number. `Season` and the 16 wind directions are cyclical too and get the same treatment."
        )

        # ---- 6. Location target encoding ----
        st.markdown("### 6. Location target encoding (rain rate per station)")
        loc_rate = (df["RainTomorrow"] == "Yes").groupby(df["Location"]).mean().sort_values(ascending=False) * 100
        c1, c2 = st.columns(2)
        c1.metric("Wettest station", f"{loc_rate.index[0]} ({loc_rate.iloc[0]:.0f}%)")
        c2.metric("Driest station", f"{loc_rate.index[-1]} ({loc_rate.iloc[-1]:.0f}%)")
        st.caption(
            "`Location` is strongly tied to the target, so each station is replaced by its rain rate "
            "(target encoding) — with one version per horizon (J+1 and J+2)."
        )
