"""Streamlit app entry point — defines the multi-page navigation.

Two sections, separated in the sidebar:
  - Analysis   : the CRISP-DM pipeline (Data Understanding → Model Interpretation)
  - Prediction : the end-user forecasting page

Run (train models first):
    .venv/Scripts/python.exe -m src.pipeline      # train & save the 4 models
    .venv/Scripts/python.exe -m streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from src.data_processing import render as render_data_processing
from src.data_understanding import render as render_data_understanding
from src.eda import render as render_eda
from src.feature_engineering import render as render_feature_engineering
from src.interpretation import render as render_interpretation
from src.modeling import render as render_modeling
from src.prediction import render as render_prediction

pg = st.navigation(
    {
        "Analysis": [
            st.Page(render_data_understanding, title="Data Understanding", icon="🗂️", url_path="data_understanding", default=True),
            st.Page(render_eda, title="EDA", icon="📊", url_path="eda"),
            st.Page(render_data_processing, title="Data Preprocessing", icon="🧹", url_path="data_preprocessing"),
            st.Page(render_feature_engineering, title="Feature Engineering", icon="🔧", url_path="feature_engineering"),
            st.Page(render_modeling, title="Modeling", icon="🤖", url_path="modeling"),
            st.Page(render_interpretation, title="Model Interpretation", icon="🔬", url_path="model_interpretation"),
        ],
        "Prediction": [
            st.Page(render_prediction, title="Prediction", icon="🌧️", url_path="prediction"),
        ],
    }
)
pg.run()
