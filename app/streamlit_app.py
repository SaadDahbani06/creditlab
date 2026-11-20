"""CREDITLAB research dashboard (entry point and navigation).

Run from the project root:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from creditlab.reporting.dashboard import ICON, apply_theme

st.set_page_config(page_title="CREDITLAB", layout="wide",
                   page_icon=str(ICON) if ICON.exists() else None)
PAGES = [
    st.Page("overview.py", title="Overview", default=True),
    st.Page("pages/1_Data_Explorer.py", title="Data Explorer", url_path="data_explorer"),
    st.Page("pages/2_Model_Comparison.py", title="Model Comparison", url_path="model_comparison"),
    st.Page("pages/3_Calibration.py", title="Calibration", url_path="calibration"),
    st.Page("pages/4_Portfolio_Risk.py", title="Portfolio Risk", url_path="portfolio_risk"),
    st.Page("pages/5_Stress_Testing.py", title="Stress Testing", url_path="stress_testing"),
    st.Page("pages/6_Policy_Lab.py", title="Policy Lab", url_path="policy_lab"),
    st.Page("pages/7_Methodology.py", title="Methodology", url_path="methodology"),
]
page = st.navigation(PAGES, position="top")
apply_theme(page.title)
page.run()
