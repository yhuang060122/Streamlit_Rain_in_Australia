"""Streamlit 应用：澳大利亚明日/后日降雨与最高温预测。

运行方式（先训练好模型）：
    .venv/Scripts/python.exe -m src.pipeline      # 训练并保存 4 个模型
    .venv/Scripts/python.exe -m streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.pipeline import (
    CLASS_TARGETS,
    MODEL_DIR,
    REG_TARGETS,
    TARGETS,
    WIND_DIRS,
    SEASONS,
    cast_categories,
    engineer,
    get_season_sh,
    load_raw,
)

st.set_page_config(page_title="澳大利亚降雨预测", page_icon="🌧️", layout="wide")


# ---------------------------------------------------------------- 数据加载
@st.cache_data(show_spinner=False)
def load_meta() -> dict:
    return json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_models_4() -> dict:
    meta = load_meta()
    return {t: joblib.load(MODEL_DIR / f"{t}.joblib") for t in meta["targets"]}


@st.cache_data(show_spinner="正在准备数据…")
def get_engineered() -> tuple[pd.DataFrame, dict]:
    meta = load_meta()
    raw = load_raw()
    df = engineer(raw)
    cat_dtypes = {k: pd.CategoricalDtype(categories=v) for k, v in meta["cat_dtypes"].items()}
    df = cast_categories(df, cat_dtypes)
    return df, meta


# ---------------------------------------------------------------- 工具
def rain_bar(prob: float, label: str) -> None:
    pct = prob * 100
    color = "#2563eb" if prob >= 0.5 else "#f59e0b"
    emoji = "🌧️" if prob >= 0.5 else "☀️"
    verdict = "大概率降雨" if prob >= 0.5 else "大概率无雨"
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


def predict_row(models: dict, meta: dict, X_row: pd.DataFrame) -> dict:
    X = X_row[meta["features"]].copy()
    cat_dtypes = {k: pd.CategoricalDtype(categories=v) for k, v in meta["cat_dtypes"].items()}
    for col, dtype in cat_dtypes.items():
        if col in X.columns and not isinstance(X[col].dtype, pd.CategoricalDtype):
            X[col] = X[col].astype(dtype)
    out = {}
    for t in meta["targets"]:
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


# ---------------------------------------------------------------- 主界面
st.title("🌧️ 澳大利亚降雨与气温预测")
st.caption("数据源：BOM 澳洲气象局 · 49 个站点 · 2007–2026 逐日观测 · 模型：XGBoost")

if not (MODEL_DIR / "meta.json").exists():
    st.warning("尚未训练模型。请先运行 `.venv/Scripts/python.exe -m src.pipeline` 完成训练后再启动应用。")
    st.stop()

df, meta = get_engineered()
models = load_models_4()
features = meta["features"]
locations = meta["locations"]

mode = st.sidebar.radio("使用方式", ["📅 历史数据回放", "✍️ 手动输入"])
st.sidebar.markdown("---")

location = st.sidebar.selectbox("站点", locations)

# ---------------------------------------------------------------- 历史回放
if mode == "📅 历史数据回放":
    loc_df = df[df["Location"] == location]
    # 只保留 RainTomorrow 已知的日期，确保有“实际值”可对照
    avail = loc_df[loc_df["RainTomorrow"].notna()][["Date"]].reset_index(drop=True)
    avail = avail.sort_values("Date")

    min_date = avail["Date"].min().date()
    max_date = avail["Date"].max().date()
    if "sel_date" not in st.session_state or not (min_date <= st.session_state["sel_date"] <= max_date):
        st.session_state["sel_date"] = max_date

    if st.sidebar.button("🎲 随机示例"):
        st.session_state["sel_date"] = avail.sample(1)["Date"].iloc[0].date()

    date = st.sidebar.date_input("日期", value=st.session_state["sel_date"], min_value=min_date, max_value=max_date)
    st.session_state["sel_date"] = date
    date = pd.Timestamp(date)

    row_df = df[(df["Location"] == location) & (df["Date"] == date)]
    if row_df.empty:
        st.error("该站点在这一天没有观测记录。")
        st.stop()
    row = row_df.iloc[0]

    # 当日实况
    st.subheader(f"{location} · {date.date()}")
    st.caption("当日观测实况（模型的输入）")
    c = st.columns(6)
    c[0].metric("最低温", f"{row['MinTemp']:.1f} °C")
    c[1].metric("最高温", f"{row['MaxTemp']:.1f} °C")
    c[2].metric("降雨量", f"{row['Rainfall']:.1f} mm")
    c[3].metric("湿度(15时)", f"{row['Humidity3pm']:.0f} %")
    c[4].metric("气压(15时)", f"{row['Pressure3pm']:.1f} hPa")
    c[5].metric("今日是否已下雨", "是" if row["RainToday"] == 1 else "否")

    # 预测结果（直接传 DataFrame 切片，保留 category dtype）
    pred = predict_row(models, meta, row_df)
    actual = actual_summary(row)

    st.markdown("### 预测结果")
    colL, colR = st.columns(2)
    with colL:
        rain_bar(pred["RainTomorrow"], "明日降雨概率 (RainTomorrow)")
        st.markdown("")
        st.metric(
            "明日最高温预测",
            f"{pred['MaxTempTomorrow']:.1f} °C",
            delta=f"实际 {actual['MaxTempTomorrow']:.1f} °C" if pd.notna(actual["MaxTempTomorrow"]) else None,
        )
    with colR:
        rain_bar(pred["RainInTwoDays"], "后日降雨概率 (RainInTwoDays)")
        st.markdown("")
        st.metric(
            "后日最高温预测",
            f"{pred['MaxTempInTwoDays']:.1f} °C",
            delta=f"实际 {actual['MaxTempInTwoDays']:.1f} °C" if pd.notna(actual["MaxTempInTwoDays"]) else None,
        )

    # 实际对照
    st.markdown("### 实际 vs 预测")
    rows = []
    for key, label in [
        ("RainTomorrow", "明日降雨"),
        ("RainInTwoDays", "后日降雨"),
    ]:
        if pd.notna(actual[key]):
            pred_yes = pred[key] >= 0.5
            act_yes = bool(actual[key])
            ok = pred_yes == act_yes
            rows.append({"目标": label, "实际": "🌧️ 有雨" if act_yes else "☀️ 无雨", "预测概率": f"{pred[key]*100:.0f}%", "判定": "✅ 正确" if ok else "❌ 误判"})
    for key, label in [
        ("MaxTempTomorrow", "明日最高温"),
        ("MaxTempInTwoDays", "后日最高温"),
    ]:
        if pd.notna(actual[key]):
            err = pred[key] - actual[key]
            rows.append({"目标": label, "实际": f"{actual[key]:.1f} °C", "预测概率": f"{pred[key]:.1f} °C", "判定": f"误差 {err:+.1f} °C"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

# ---------------------------------------------------------------- 手动输入
else:
    st.subheader(f"{location} · 手动输入当日天气")
    st.caption("填写今日观测值，模型预测明日/后日降雨与最高温。留空(留 NaN)也支持，XGBoost 会原生处理。")

    # 日期仅用于推导季节
    month = st.sidebar.selectbox("月份（用于季节）", list(range(1, 13)), index=11)
    season = get_season_sh(month)

    inp = {"Location": location, "RainToday": 0, "season": season}

    c = st.columns(4)
    inp["MinTemp"] = c[0].number_input("MinTemp 最低温 (°C)", -20.0, 50.0, 12.0)
    inp["MaxTemp"] = c[1].number_input("MaxTemp 最高温 (°C)", -20.0, 50.0, 23.0)
    inp["Temp9am"] = c[2].number_input("Temp9am 9时气温 (°C)", -20.0, 50.0, 17.0)
    inp["Temp3pm"] = c[3].number_input("Temp3pm 15时气温 (°C)", -20.0, 55.0, 22.0)

    c = st.columns(4)
    inp["Humidity9am"] = c[0].number_input("Humidity9am (%)", 0.0, 100.0, 70.0)
    inp["Humidity3pm"] = c[1].number_input("Humidity3pm (%)", 0.0, 100.0, 52.0)
    inp["Pressure9am"] = c[2].number_input("Pressure9am (hPa)", 980.0, 1050.0, 1017.0)
    inp["Pressure3pm"] = c[3].number_input("Pressure3pm (hPa)", 980.0, 1050.0, 1015.0)

    c = st.columns(4)
    inp["WindGustDir"] = c[0].selectbox("WindGustDir 阵风方向", WIND_DIRS, index=12)
    inp["WindGustSpeed"] = c[1].number_input("WindGustSpeed (km/h)", 0.0, 160.0, 40.0)
    inp["WindDir9am"] = c[2].selectbox("WindDir9am 9时风向", WIND_DIRS, index=0)
    inp["WindDir3pm"] = c[3].selectbox("WindDir3pm 15时风向", WIND_DIRS, index=14)

    c = st.columns(4)
    inp["WindSpeed9am"] = c[0].number_input("WindSpeed9am (km/h)", 0.0, 130.0, 13.0)
    inp["WindSpeed3pm"] = c[1].number_input("WindSpeed3pm (km/h)", 0.0, 130.0, 19.0)
    inp["Rainfall"] = c[2].number_input("Rainfall 今日降雨量 (mm)", 0.0, 400.0, 0.0)
    inp["RainToday"] = 1 if c[3].selectbox("RainToday 今日是否下雨", ["否", "是"], index=0) == "是" else 0

    c = st.columns(4)
    inp["Evaporation"] = c[0].number_input("Evaporation (mm)", 0.0, 150.0, 5.0)
    inp["Sunshine"] = c[1].number_input("Sunshine 日照 (h)", 0.0, 15.0, 7.6)
    inp["Cloud9am"] = c[2].number_input("Cloud9am 云量 (oktas)", 0.0, 9.0, 4.0)
    inp["Cloud3pm"] = c[3].number_input("Cloud3pm 云量 (oktas)", 0.0, 9.0, 4.0)

    if st.button("🔮 预测", type="primary"):
        X_row = pd.DataFrame([inp])
        cat_dtypes = {k: pd.CategoricalDtype(categories=v) for k, v in meta["cat_dtypes"].items()}
        X_row = cast_categories(X_row, cat_dtypes)
        pred = predict_row(models, meta, X_row)

        st.markdown("### 预测结果")
        colL, colR = st.columns(2)
        with colL:
            rain_bar(pred["RainTomorrow"], "明日降雨概率")
            st.metric("明日最高温预测", f"{pred['MaxTempTomorrow']:.1f} °C")
        with colR:
            rain_bar(pred["RainInTwoDays"], "后日降雨概率")
            st.metric("后日最高温预测", f"{pred['MaxTempInTwoDays']:.1f} °C")

# ---------------------------------------------------------------- 模型信息
st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ 模型与指标"):
    for t in TARGETS:
        m = meta["metrics"][t]
        if t in CLASS_TARGETS:
            st.write(f"**{t}** · ROC-AUC {m['roc_auc']} · F1 {m['f1']}")
        else:
            st.write(f"**{t}** · MAE {m['mae']} °C · R² {m['r2']}")

st.markdown("---")
st.markdown("### 🔍 特征重要度（明日降雨模型 Top 15）")
imp = meta["importance_rain_tomorrow"]
imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:15]
imp_df = pd.DataFrame(imp_sorted, columns=["特征", "重要度"]).set_index("特征")
st.bar_chart(imp_df)

st.caption(
    "说明：模型保留全部特征（含高缺失的 Evaporation/Sunshine/Cloud 等），不手动填补缺失值，"
    "由 XGBoost 原生处理 NaN。降雨概率阈值为 50%。"
)
