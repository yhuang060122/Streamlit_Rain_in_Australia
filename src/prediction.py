"""Prediction page renderer — historical replay + manual forecast.

This is the end-user forecasting page: it uses the trained XGBoost models to predict
tomorrow's / the day-after's rain and max temperature, either by replaying a historical
day or by entering today's observations manually.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.forecast import actual_summary, get_engineered, load_meta, load_models, predict_row, rain_bar
from src.pipeline import MODEL_DIR, SEASONS, WIND_DIRS, cast_categories

# default values + fallback for missing values when sampling a random observation
NUM_DEFAULTS = {
    "MinTemp": 12.0, "MaxTemp": 23.0, "Temp9am": 17.0, "Temp3pm": 22.0,
    "Humidity9am": 70.0, "Humidity3pm": 52.0, "Pressure9am": 1017.0, "Pressure3pm": 1015.0,
    "WindGustSpeed": 40.0, "WindSpeed9am": 13.0, "WindSpeed3pm": 19.0,
    "Rainfall": 0.0, "Evaporation": 5.0, "Sunshine": 7.6, "Cloud9am": 4.0, "Cloud3pm": 4.0,
}
WIND_DEFAULTS = {"WindGustDir": "W", "WindDir9am": "N", "WindDir3pm": "NW"}


def render() -> None:
    st.set_page_config(page_title="Prediction — Rain in Australia", page_icon="🌧️", layout="wide")

    st.title("🌧️ Prediction")
    st.caption(
        "Use the trained XGBoost models to predict tomorrow's / the day-after's rain and max temperature."
    )

    if not (MODEL_DIR / "meta.json").exists():
        st.warning("Models not trained yet. Run `.venv/Scripts/python.exe -m src.pipeline` first, then relaunch the app.")
        st.stop()

    meta = load_meta()
    models = load_models()
    df_eng, _ = get_engineered()

    tabs = st.tabs(["📅 Historical Replay", "✍️ Manual Input", "🧠 Model & Parameters"])

    # ---------------------------------------------------------------- tab 1
    with tabs[0]:
        def _random_hr_date() -> None:
            loc = st.session_state.get("hr_station", meta["locations"][0])
            avail = (
                df_eng.loc[df_eng["Location"] == loc]
                .dropna(subset=["RainTomorrow"])["Date"]
                .dt.date
            )
            st.session_state["hr_date"] = avail.sample(1).iloc[0]

        title_c, btn_c = st.columns([5, 1])
        title_c.subheader("Historical replay")
        btn_c.button("🎲 Random", use_container_width=True, key="hr_rand", on_click=_random_hr_date)

        st.caption(
            "Pick a station and date to see that day's observations and the trained model's predictions "
            "versus the actual outcome."
        )

        c1, c2 = st.columns(2)
        location = c1.selectbox("Station", meta["locations"], key="hr_station")

        loc_df = df_eng[df_eng["Location"] == location]
        avail_dates = loc_df[loc_df["RainTomorrow"].notna()]["Date"].dt.date
        min_d, max_d = avail_dates.min(), avail_dates.max()

        if "hr_date" not in st.session_state or not (min_d <= st.session_state["hr_date"] <= max_d):
            st.session_state["hr_date"] = max_d

        date = c2.date_input("Date", min_value=min_d, max_value=max_d, key="hr_date")
        date = pd.Timestamp(date)

        row_df = df_eng[(df_eng["Location"] == location) & (df_eng["Date"] == date)]
        if row_df.empty:
            st.error("No observation for this station on the selected date.")
        else:
            row = row_df.iloc[0]

            st.markdown(f"**{location} · {date.date()} — observed conditions (model inputs)**")
            c = st.columns(6)
            c[0].metric("Min temp", f"{row['MinTemp']:.1f} °C")
            c[1].metric("Max temp", f"{row['MaxTemp']:.1f} °C")
            c[2].metric("Rainfall", f"{row['Rainfall']:.1f} mm")
            c[3].metric("Humidity (3pm)", f"{row['Humidity3pm']:.0f} %")
            c[4].metric("Pressure (3pm)", f"{row['Pressure3pm']:.1f} hPa")
            c[5].metric("Rained today?", "Yes" if row["RainToday"] == 1 else "No")

            pred = predict_row(models, meta, row_df)
            actual = actual_summary(row)

            st.markdown("### Predictions")
            colL, colR = st.columns(2)
            with colL:
                rain_bar(pred["RainTomorrow"], "Rain tomorrow probability (RainTomorrow)")
                st.markdown("")
                st.metric(
                    "Max temp tomorrow",
                    f"{pred['MaxTempTomorrow']:.1f} °C",
                    delta=f"Actual {actual['MaxTempTomorrow']:.1f} °C" if pd.notna(actual["MaxTempTomorrow"]) else None,
                )
            with colR:
                rain_bar(pred["RainInTwoDays"], "Rain in 2 days probability (RainInTwoDays)")
                st.markdown("")
                st.metric(
                    "Max temp in 2 days",
                    f"{pred['MaxTempInTwoDays']:.1f} °C",
                    delta=f"Actual {actual['MaxTempInTwoDays']:.1f} °C" if pd.notna(actual["MaxTempInTwoDays"]) else None,
                )

            st.markdown("### Actual vs predicted")
            rows = []
            for key, label in [
                ("RainTomorrow", "Rain tomorrow"),
                ("RainInTwoDays", "Rain in 2 days"),
            ]:
                if pd.notna(actual[key]):
                    pred_yes = pred[key] >= 0.5
                    act_yes = bool(actual[key])
                    ok = pred_yes == act_yes
                    rows.append({
                        "Target": label,
                        "Actual": "🌧️ Rain" if act_yes else "☀️ No rain",
                        "Predicted": f"{pred[key]*100:.0f}%",
                        "Verdict": "✅ Correct" if ok else "❌ Wrong",
                    })
            for key, label in [
                ("MaxTempTomorrow", "Max temp tomorrow"),
                ("MaxTempInTwoDays", "Max temp in 2 days"),
            ]:
                if pd.notna(actual[key]):
                    err = pred[key] - actual[key]
                    rows.append({
                        "Target": label,
                        "Actual": f"{actual[key]:.1f} °C",
                        "Predicted": f"{pred[key]:.1f} °C",
                        "Verdict": f"Error {err:+.1f} °C",
                    })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # ---------------------------------------------------------------- tab 2
    with tabs[1]:
        def _random_manual() -> None:
            sample = df_eng.sample(1).iloc[0]
            for f in NUM_DEFAULTS:
                v = sample[f]
                st.session_state[f"mi_{f}"] = float(v) if pd.notna(v) else NUM_DEFAULTS[f]
            for f in WIND_DEFAULTS:
                v = sample[f]
                st.session_state[f"mi_{f}"] = str(v) if pd.notna(v) else WIND_DEFAULTS[f]
            st.session_state["mi_RainToday"] = "Yes" if sample["RainToday"] == 1 else "No"
            st.session_state["mi_location"] = str(sample["Location"])
            st.session_state["mi_season"] = str(sample["season"])

        title_c, btn_c = st.columns([5, 1])
        title_c.subheader("Manual input")
        btn_c.button("🎲 Random", use_container_width=True, key="mi_rand", on_click=_random_manual)

        st.caption(
            "Enter today's observations to predict tomorrow's and the day-after's rain and max temperature. "
            "Missing values (NaN) are supported — XGBoost handles them natively. Use Random to fill the form "
            "with a real historical observation."
        )

        # initialise state for every field (the Random button overwrites these)
        for f, d in NUM_DEFAULTS.items():
            st.session_state.setdefault(f"mi_{f}", d)
        for f, d in WIND_DEFAULTS.items():
            st.session_state.setdefault(f"mi_{f}", d)
        st.session_state.setdefault("mi_RainToday", "No")
        st.session_state.setdefault("mi_location", meta["locations"][0])
        st.session_state.setdefault("mi_season", "Summer")

        c1, c2 = st.columns(2)
        location = c1.selectbox("Location", meta["locations"], key="mi_location")
        season = c2.selectbox("Season", SEASONS, key="mi_season")

        inp = {"Location": location, "season": season}

        c = st.columns(4)
        inp["MinTemp"] = c[0].number_input("Min temp (°C)", -20.0, 50.0, key="mi_MinTemp")
        inp["MaxTemp"] = c[1].number_input("Max temp (°C)", -20.0, 50.0, key="mi_MaxTemp")
        inp["Temp9am"] = c[2].number_input("Temp 9am (°C)", -20.0, 50.0, key="mi_Temp9am")
        inp["Temp3pm"] = c[3].number_input("Temp 3pm (°C)", -20.0, 55.0, key="mi_Temp3pm")

        c = st.columns(4)
        inp["Humidity9am"] = c[0].number_input("Humidity 9am (%)", 0.0, 100.0, key="mi_Humidity9am")
        inp["Humidity3pm"] = c[1].number_input("Humidity 3pm (%)", 0.0, 100.0, key="mi_Humidity3pm")
        inp["Pressure9am"] = c[2].number_input("Pressure 9am (hPa)", 980.0, 1050.0, key="mi_Pressure9am")
        inp["Pressure3pm"] = c[3].number_input("Pressure 3pm (hPa)", 980.0, 1050.0, key="mi_Pressure3pm")

        c = st.columns(4)
        inp["WindGustDir"] = c[0].selectbox("Wind gust direction", WIND_DIRS, key="mi_WindGustDir")
        inp["WindGustSpeed"] = c[1].number_input("Wind gust speed (km/h)", 0.0, 160.0, key="mi_WindGustSpeed")
        inp["WindDir9am"] = c[2].selectbox("Wind direction 9am", WIND_DIRS, key="mi_WindDir9am")
        inp["WindDir3pm"] = c[3].selectbox("Wind direction 3pm", WIND_DIRS, key="mi_WindDir3pm")

        c = st.columns(4)
        inp["WindSpeed9am"] = c[0].number_input("Wind speed 9am (km/h)", 0.0, 130.0, key="mi_WindSpeed9am")
        inp["WindSpeed3pm"] = c[1].number_input("Wind speed 3pm (km/h)", 0.0, 130.0, key="mi_WindSpeed3pm")
        inp["Rainfall"] = c[2].number_input("Rainfall today (mm)", 0.0, 400.0, key="mi_Rainfall")
        rain_today = c[3].selectbox("Rained today?", ["No", "Yes"], key="mi_RainToday")
        inp["RainToday"] = 1 if rain_today == "Yes" else 0

        c = st.columns(4)
        inp["Evaporation"] = c[0].number_input("Evaporation (mm)", 0.0, 150.0, key="mi_Evaporation")
        inp["Sunshine"] = c[1].number_input("Sunshine (h)", 0.0, 15.0, key="mi_Sunshine")
        inp["Cloud9am"] = c[2].number_input("Cloud 9am (oktas)", 0.0, 9.0, key="mi_Cloud9am")
        inp["Cloud3pm"] = c[3].number_input("Cloud 3pm (oktas)", 0.0, 9.0, key="mi_Cloud3pm")

        if st.button("🔮 Predict", type="primary"):
            X_row = pd.DataFrame([inp])
            cat_dtypes = {k: pd.CategoricalDtype(categories=v) for k, v in meta["cat_dtypes"].items()}
            X_row = cast_categories(X_row, cat_dtypes)
            pred = predict_row(models, meta, X_row)

            st.markdown("### Predictions")
            colL, colR = st.columns(2)
            with colL:
                rain_bar(pred["RainTomorrow"], "Rain tomorrow probability")
                st.metric("Max temp tomorrow", f"{pred['MaxTempTomorrow']:.1f} °C")
            with colR:
                rain_bar(pred["RainInTwoDays"], "Rain in 2 days probability")
                st.metric("Max temp in 2 days", f"{pred['MaxTempInTwoDays']:.1f} °C")

    # ---------------------------------------------------------------- tab 3
    with tabs[2]:
        st.subheader("How the model works")

        st.markdown(
            "The predictions come from **XGBoost** (eXtreme Gradient Boosting), a gradient-boosted "
            "decision-tree ensemble. One model is trained per target in `src/pipeline.py` and saved to "
            "`models/*.joblib`; this page loads them and applies them to your inputs."
        )

        st.markdown(
            "**Why XGBoost?** On the Modelling page it beat the logistic-regression and random-forest "
            "baselines on F1 / ROC-AUC, and it fits this dataset well:"
        )
        st.markdown(
            "- **Missing values handled natively** — the weather records are full of gaps, so no manual imputation is needed.\n"
            "- **Categorical features handled natively** (`enable_categorical=True`) — Location, wind directions, season.\n"
            "- **Class imbalance handled** via `scale_pos_weight` — only ~22% of days have rain.\n"
            "- **Regularised** (`reg_lambda` / `reg_alpha`) — keeps 200k+ noisy rows from overfitting."
        )

        # ---- the four models
        st.markdown("### The four models")
        model_table = pd.DataFrame(
            [
                {"Target": "RainTomorrow", "Task": "Binary classification", "Model": "XGBClassifier", "Output": "P(rain tomorrow)"},
                {"Target": "RainInTwoDays", "Task": "Binary classification", "Model": "XGBClassifier", "Output": "P(rain in 2 days)"},
                {"Target": "MaxTempTomorrow", "Task": "Regression", "Model": "XGBRegressor", "Output": "Max temp tomorrow (°C)"},
                {"Target": "MaxTempInTwoDays", "Task": "Regression", "Model": "XGBRegressor", "Output": "Max temp in 2 days (°C)"},
            ]
        )
        st.dataframe(model_table, width="stretch", hide_index=True)

        # ---- hyperparameters
        st.markdown("### Hyperparameters")
        st.markdown(
            "Shared by all four models (see `_classifier` / `_regressor` in `src/pipeline.py`). "
            "They were chosen for a good speed / accuracy trade-off rather than grid-searched."
        )
        spw = meta["metrics"]["RainTomorrow"]["pos_rate_train"]
        spw_val = round((1 - spw) / spw, 2)
        params_table = pd.DataFrame(
            [
                {"Parameter": "n_estimators", "Value": "400", "Meaning": "Number of boosting trees. More trees = stronger model, up to a point."},
                {"Parameter": "learning_rate", "Value": "0.05", "Meaning": "Shrinkage per tree. Small rate + many trees = stable, accurate learning."},
                {"Parameter": "max_depth", "Value": "6", "Meaning": "Maximum depth of each tree. Caps complexity to avoid memorising noise."},
                {"Parameter": "subsample", "Value": "0.85", "Meaning": "Fraction of rows sampled per tree (adds randomness against overfitting)."},
                {"Parameter": "colsample_bytree", "Value": "0.85", "Meaning": "Fraction of features sampled per tree."},
                {"Parameter": "min_child_weight", "Value": "5", "Meaning": "Minimum sample weight per leaf. Larger = more conservative splits."},
                {"Parameter": "reg_lambda", "Value": "5.0", "Meaning": "L2 regularisation on leaf weights (smooths the model)."},
                {"Parameter": "reg_alpha", "Value": "0.1", "Meaning": "L1 regularisation on leaf weights (encourages sparsity)."},
                {"Parameter": "scale_pos_weight", "Value": f"≈ {spw_val} (classification only)", "Meaning": "Weights the minority (rain) class — negative/positive ratio of the training set."},
                {"Parameter": "tree_method", "Value": "hist", "Meaning": "Histogram-based split finding — much faster, near-identical accuracy."},
                {"Parameter": "enable_categorical", "Value": "True", "Meaning": "Treats categorical columns natively instead of one-hot encoding."},
                {"Parameter": "eval_metric", "Value": "aucpr / mae", "Meaning": "Training-time metric: area under the PR curve (classification) or mean absolute error (regression)."},
                {"Parameter": "random_state", "Value": "42", "Meaning": "Random seed so training is reproducible."},
            ]
        )
        st.dataframe(
            params_table,
            width="stretch",
            hide_index=True,
            column_config={"Meaning": st.column_config.TextColumn(width="large")},
        )

        # ---- evaluation metrics
        st.markdown("### Evaluation metrics (held-out test set)")
        m = meta["metrics"]
        cls_rows, reg_rows = [], []
        for tgt, d in m.items():
            if d["type"] == "classification":
                cls_rows.append(
                    {
                        "Target": tgt,
                        "ROC-AUC": d["roc_auc"],
                        "F1": d["f1"],
                        "Precision": d["precision"],
                        "Recall": d["recall"],
                        "Accuracy": d["accuracy"],
                    }
                )
            else:
                reg_rows.append(
                    {
                        "Target": tgt,
                        "MAE (°C)": d["mae"],
                        "RMSE (°C)": d["rmse"],
                        "R²": d["r2"],
                    }
                )

        st.markdown("**Classification** — how well it separates rain vs no-rain days:")
        st.dataframe(pd.DataFrame(cls_rows), width="stretch", hide_index=True)

        st.markdown("**Regression** — how close the predicted temperature is:")
        st.dataframe(pd.DataFrame(reg_rows), width="stretch", hide_index=True)

        # ---- features
        st.markdown("### Input features")
        st.markdown(
            f"All four models consume the same {len(meta['features'])} features. Categorical columns "
            "(Location, wind directions, season) are encoded natively, not one-hot."
        )
        st.markdown(" · ".join(f"`{f}`" for f in meta["features"]))

        # ---- feature importance
        imp = meta.get("importance_rain_tomorrow", {})
        if imp:
            top = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:8]
            imp_rows = [{"Feature": k, "Importance": round(v, 4)} for k, v in top]
            st.markdown("### What drives the rain prediction")
            st.markdown("Top-8 gain-based feature importances for the RainTomorrow model:")
            st.dataframe(pd.DataFrame(imp_rows), width="stretch", hide_index=True)
