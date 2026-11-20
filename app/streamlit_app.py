"""CreditLab research site (entry point and navigation).

Run from the project root:  streamlit run app/streamlit_app.py

Four read-only pages. The original interactive tools (stress testing, policy lab) are preserved in
app/legacy/ and can be run on their own; see app/legacy/README.md.
"""

from __future__ import annotations

import streamlit as st

from creditlab.reporting.dashboard import ICON, apply_theme

st.set_page_config(page_title="CreditLab", layout="wide",
                   page_icon=str(ICON) if ICON.exists() else None)
PAGES = [
    st.Page("site/overview.py", title="Overview", default=True),
    st.Page("site/modeling.py", title="Predictive Modeling", url_path="modeling"),
    st.Page("site/risk.py", title="Portfolio Risk & Findings", url_path="risk"),
    st.Page("site/methodology.py", title="Methodology & Limitations", url_path="methodology"),
]
page = st.navigation(PAGES, position="top")
apply_theme(page.title)
page.run()
