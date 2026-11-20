"""Methodology: formulas, data source, validation design, assumptions and limitations."""

from __future__ import annotations

import streamlit as st

from creditlab.reporting import dashboard as d

st.title("Methodology")

tabs = st.tabs(["Mathematics", "Data source", "Assumptions", "Limitations", "Model card"])
for tab, name in zip(tabs, ["MATHEMATICS.md", "DATA_DICTIONARY.md", "ASSUMPTIONS.md",
                            "LIMITATIONS.md", "MODEL_CARD.md"], strict=True):
    with tab:
        st.markdown(d.doc(name))
