"""Shared helpers for the Streamlit dashboard (artifact loading, Plotly styling, guards).

The dashboard never trains models. It reads artifacts produced by `creditlab all` and, on the
Stress Testing and Policy Lab pages, re-runs the same risk/policy functions with user-chosen
HYPOTHETICAL assumptions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from creditlab.utils.config import PROJECT_ROOT, Config, load_config, read_json

# ----------------------------------------------------------------------------- dashboard theme
# Red-and-white dashboard palette. The report figures (matplotlib, creditlab.reporting.style)
# keep their own palette; only the interactive dashboard uses these values.
# Categorical slots validated (dataviz validate_palette.js, light mode): lightness band, chroma
# floor, adjacent-pair CVD separation (worst ΔE 9.8) and normal-vision floor all pass; slot 3
# (orange) is below 3:1 contrast, so it is always paired with a legend and a table view.
RED = "#D7263D"
RED_DARK = "#A81C2F"
RED_SOFT = "#FDECEE"
TEXT = "#1C1B1F"
TEXT_SECONDARY = "#5F5A5C"
GRID = "#F0E9E9"
NEUTRAL = "#A7A0A2"
SERIES = [RED, "#3D6FB6", "#E58A3A", "#2A9D8F", "#9B5DB5"]

# Fixed entity -> colour mapping (colour follows the entity, never its rank).
ENTITY_COLORS = {
    "dummy": NEUTRAL,
    "logistic": SERIES[1],
    "xgboost": SERIES[0],
    "observed": TEXT_SECONDARY,
    "base": SERIES[0],
    "base_correlated": SERIES[1],
    "mild_stress": SERIES[2],
    "severe_stress": SERIES[4],
    "correlated_downturn": SERIES[3],
    "A_retain_all": NEUTRAL,
    "B_pd_threshold": SERIES[0],
    "C_expected_contribution": SERIES[1],
    "D_random_equal_size": SERIES[2],
    "raw": SERIES[0],
    "sigmoid": SERIES[1],
    "isotonic": SERIES[2],
}

LOGO = PROJECT_ROOT / "app" / "assets" / "logo.svg"
ICON = PROJECT_ROOT / "app" / "assets" / "icon.svg"

_CSS = f"""
<style>
:root {{
  --cl-red: {RED}; --cl-red-dark: {RED_DARK}; --cl-red-soft: {RED_SOFT};
  --cl-ink: {TEXT}; --cl-ink-2: {TEXT_SECONDARY}; --cl-line: #ECE4E4;
  --cl-card: #FFFFFF; --cl-canvas: #F6F3F3;
  --cl-shadow: 0 1px 2px rgba(28, 27, 31, .04), 0 6px 24px rgba(28, 27, 31, .05);
}}

/* ---- top bar: white, hairline, pill navigation like a product dashboard ---- */
header[data-testid="stHeader"] {{
  background: var(--cl-card);
  border-bottom: 1px solid var(--cl-line);
  height: 4.25rem;
}}
[data-testid="stDecoration"] {{ display: none; }}
[data-testid="stTopNav"] {{ gap: .2rem; padding: 0 .5rem 0 .15rem; }}
[data-testid="stTopNavLink"], [data-testid="stTopNavSection"] {{
  border: 1px solid var(--cl-line);
  border-radius: 999px;
  padding: .3rem .7rem;
  background: var(--cl-card);
  white-space: nowrap;
  transition: background .15s ease, border-color .15s ease, color .15s ease;
}}
[data-testid="stTopNavLink"] p, [data-testid="stTopNavSection"] p {{
  color: var(--cl-ink); font-weight: 500; font-size: .82rem; white-space: nowrap;
}}
[data-testid="stTopNavSection"] p {{ color: var(--cl-red-dark); font-weight: 600; }}
[data-testid="stTopNavLink"]:hover, [data-testid="stTopNavSection"]:hover {{
  border-color: #F3B8C0; background: var(--cl-red-soft);
}}
[data-testid="stTopNavLink"][aria-current="page"] {{
  background: var(--cl-red); border-color: var(--cl-red);
  box-shadow: 0 4px 12px rgba(215, 38, 61, .25);
}}
[data-testid="stTopNavLink"][aria-current="page"] p {{ color: #FFFFFF; font-weight: 600; }}
[data-testid="stHeaderLogo"], [data-testid="stLogo"] {{
  height: 2.2rem; max-height: 2.2rem; width: auto; max-width: none; margin-right: .9rem;
}}

/* ---- page canvas & typography ---- */
[data-testid="stAppViewContainer"] {{ background: var(--cl-canvas); }}
[data-testid="stMainBlockContainer"] {{ padding-top: 6rem; max-width: 1360px; }}
h1 {{ font-weight: 800 !important; letter-spacing: -.02em; color: var(--cl-ink); }}
h2, h3 {{ font-weight: 700 !important; letter-spacing: -.01em; color: var(--cl-ink); }}
[data-testid="stHeadingWithActionElements"] h3 {{ padding-top: 1.1rem; }}
[data-testid="stCaptionContainer"] {{ color: var(--cl-ink-2); }}

/* ---- cards: metrics, charts, tables, expanders ---- */
[data-testid="stMetric"] {{
  background: var(--cl-card);
  border: 1px solid var(--cl-line);
  border-radius: 18px;
  padding: 1rem 1.15rem 1.05rem;
  box-shadow: var(--cl-shadow);
  position: relative;
  overflow: hidden;
}}
[data-testid="stMetric"]::before {{
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
  background: var(--cl-red);
}}
[data-testid="stMetricLabel"] p {{
  color: var(--cl-ink-2); font-size: .8rem; font-weight: 600;
}}
[data-testid="stMetricValue"] {{
  color: var(--cl-ink); font-weight: 700; letter-spacing: -.02em; font-size: 1.55rem;
}}

[data-testid="stPlotlyChart"], [data-testid="stImage"] {{
  background: var(--cl-card);
  border: 1px solid var(--cl-line);
  border-radius: 18px;
  padding: .6rem .7rem .3rem;
  box-shadow: var(--cl-shadow);
}}
[data-testid="stDataFrame"] {{
  background: var(--cl-card);
  border: 1px solid var(--cl-line);
  border-radius: 14px;
  padding: .25rem;
  box-shadow: var(--cl-shadow);
}}
[data-testid="stExpander"] details {{
  background: var(--cl-card); border-radius: 16px; box-shadow: var(--cl-shadow);
}}

/* ---- callouts: soft tinted cards with a coloured edge ---- */
[data-testid="stAlertContainer"] {{
  border-radius: 14px;
  border-left: 4px solid currentColor;
}}

/* ---- sidebar (inputs on Stress Testing / Policy Lab) ---- */
[data-testid="stSidebar"] {{ border-right: 1px solid var(--cl-line); }}
[data-testid="stSidebarUserContent"] h2, [data-testid="stSidebarUserContent"] h3 {{
  color: var(--cl-red-dark);
}}

/* ---- brand strip under the page title ---- */
.cl-eyebrow {{
  display: inline-flex; align-items: center; gap: .45rem;
  font-size: .78rem; font-weight: 600; color: var(--cl-ink-2);
  margin-bottom: -.4rem;
}}
.cl-eyebrow b {{ color: var(--cl-red); font-weight: 700; }}
.cl-dot {{ width: .45rem; height: .45rem; border-radius: 50%; background: var(--cl-red); }}
</style>
"""


def apply_theme(page_title: str | None = None) -> None:
    """Inject the red-and-white dashboard theme (CSS + header logo). Presentation only."""
    import streamlit as st

    if LOGO.exists():
        st.logo(str(LOGO), size="large")
    st.html(_CSS)
    if page_title:
        st.html(f'<div class="cl-eyebrow"><span class="cl-dot"></span>'
                f'<span>CREDITLAB</span><span>/</span><b>{page_title}</b></div>')

SETUP = """Required artifacts are missing. From the project root run:

```bash
source .venv/bin/activate
creditlab all          # fetch -> prepare -> train -> evaluate -> simulate -> policy -> report
```

then reload this page."""

HYPOTHETICAL_BANNER = (
    "**Hypothetical.** LGD, EAD, revenue and stress parameters are assumptions, not observed "
    "values. Losses are one-month, in NTD, and are not observed bank losses. Research use only; "
    "not a lending or account-management system."
)


def config() -> Config:
    return load_config(PROJECT_ROOT / "configs" / "default.yaml")


def missing(cfg: Config, rel_paths: list[str]) -> list[str]:
    return [p for p in rel_paths if not (cfg.artifacts / p).exists()]


def read_csv(cfg: Config, rel: str, **kw) -> pd.DataFrame:
    return pd.read_csv(cfg.artifacts / rel, **kw)


def read(cfg: Config, rel: str):
    return read_json(cfg.artifacts / rel)


def doc(name: str) -> str:
    path = PROJECT_ROOT / "docs" / name
    return path.read_text() if path.exists() else f"*{name} not found.*"


def figure_path(cfg: Config, name: str) -> Path:
    return cfg.artifacts / "figures" / name


def style(fig, height: int = 380, title: str | None = None):
    fig.update_layout(
        template="plotly_white", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title={"text": title, "x": 0, "xanchor": "left", "y": 0.98, "yanchor": "top"}
        if title else None,
        font={"family": "Inter, -apple-system, Segoe UI, sans-serif", "size": 13,
              "color": TEXT_SECONDARY},
        title_font={"color": TEXT, "size": 15, "weight": 700},
        # title sits in the top margin, the legend in a separate row just above the plot
        margin={"l": 10, "r": 10, "t": 90 if title else 40, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0},
        hoverlabel={"bgcolor": "white", "bordercolor": GRID, "font": {"color": TEXT}},
        bargap=0.25,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    # 2px surface gap between adjacent filled marks (bars, histogram bins)
    fig.update_traces(marker_line_color="white", marker_line_width=1,
                      selector=lambda t: t.type in ("bar", "histogram"))
    return fig


__all__ = ["ENTITY_COLORS", "NEUTRAL", "SERIES", "SETUP", "HYPOTHETICAL_BANNER", "apply_theme",
           "config", "doc", "figure_path", "missing", "read", "read_csv", "style"]
