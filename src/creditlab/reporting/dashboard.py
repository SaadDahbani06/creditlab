"""Shared helpers for the CreditLab research site (artifact loading, design system, Plotly style).

The site is read-only: it never trains models or runs simulations. Every number it shows is read
from artifacts written by the command-line pipeline (`creditlab all`). The interactive tools that
re-run the risk and policy functions with user inputs are kept in `app/legacy/` (not linked from
the main navigation) and still use the helpers below.
"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from creditlab.utils.config import PROJECT_ROOT, Config, load_config, read_json

# ----------------------------------------------------------------------------- design tokens
# Two-series data palette validated with the dataviz validator (light mode): lightness band,
# chroma floor, CVD separation (worst ΔE 21.0) and contrast all pass. Red marks the selected model
# / the correlated or uncertainty-adjusted case; blue marks the comparison; gray is reference only.
RED = "#C8102E"
BLUE = "#2C6FB0"
INK = "#161616"
INK_2 = "#585858"
INK_3 = "#8A8A8A"
RULE = "#E7E7E7"
GRID = "#EFEFEF"
NEUTRAL = "#A3A3A3"
PAPER = "#FFFFFF"

# Kept for the legacy interactive pages (app/legacy/): fixed entity -> colour mapping.
TEXT, TEXT_SECONDARY = INK, INK_2
SERIES = [RED, BLUE, "#E58A3A", "#2A9D8F", "#9B5DB5"]
ENTITY_COLORS = {
    "dummy": NEUTRAL, "logistic": BLUE, "xgboost": RED, "observed": INK_2,
    "base": BLUE, "base_correlated": RED, "mild_stress": SERIES[2], "severe_stress": SERIES[4],
    "correlated_downturn": SERIES[3],
    "A_retain_all": NEUTRAL, "B_pd_threshold": RED, "C_expected_contribution": BLUE,
    "D_random_equal_size": SERIES[2],
    "raw": RED, "sigmoid": BLUE, "isotonic": SERIES[2],
}

LOGO = PROJECT_ROOT / "app" / "assets" / "logo.svg"
ICON = PROJECT_ROOT / "app" / "assets" / "icon.svg"
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

_FONTS = ("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&"
          "family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap")

_CSS = f"""
<style>
@import url('{_FONTS}');
:root {{
  --red: {RED}; --ink: {INK}; --ink-2: {INK_2}; --ink-3: {INK_3}; --rule: {RULE};
  --serif: 'Source Serif 4', Georgia, 'Times New Roman', serif;
  --sans: Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif;
}}
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background: #FFFFFF; }}
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display: none; }}

/* header: hairline, plain text navigation, active page underlined in red */
header[data-testid="stHeader"] {{ background: #FFFFFF; border-bottom: 1px solid var(--rule); }}
[data-testid="stTopNav"] {{ gap: 1.6rem; padding: 0 .4rem; }}
[data-testid="stTopNavLink"] {{
  background: transparent !important; border-radius: 0; padding: .55rem 0;
  border-bottom: 2px solid transparent;
}}
[data-testid="stTopNavLink"] p {{
  font-family: var(--sans); font-size: .875rem; font-weight: 500; color: var(--ink-2);
  white-space: nowrap;
}}
[data-testid="stTopNavLink"]:hover p {{ color: var(--ink); }}
[data-testid="stTopNavLink"][aria-current="page"] {{ border-bottom-color: var(--red); }}
[data-testid="stTopNavLink"][aria-current="page"] p {{ color: var(--ink); }}
[data-testid="stTopNavSection"] p {{ font-size: .875rem; color: var(--ink-2); }}
[data-testid="stHeaderLogo"], [data-testid="stLogo"] {{
  height: 1.55rem; max-height: 1.55rem; width: auto; max-width: none; margin-right: 1.4rem;
}}
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"] {{ display: none; }}

/* page column: narrow measure, generous whitespace */
[data-testid="stMainBlockContainer"] {{ max-width: 1080px; padding: 5.2rem 2rem 5rem; }}
@media (max-width: 640px) {{
  [data-testid="stMainBlockContainer"] {{ padding: 4.6rem 1rem 3rem; }}
  [data-testid="stTopNav"] {{ gap: 1rem; }}
}}

/* typography */
h1, h2, h3 {{ font-family: var(--serif) !important; color: var(--ink) !important;
             letter-spacing: -.01em; }}
h1 {{ font-size: 2.6rem !important; font-weight: 600 !important; line-height: 1.12 !important; }}
h2 {{ font-size: 1.55rem !important; font-weight: 600 !important; margin-top: 2.6rem !important;
     padding-bottom: .5rem !important; border-bottom: 1px solid var(--rule); }}
h3 {{ font-size: 1.15rem !important; font-weight: 600 !important; margin-top: 1.2rem !important; }}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{
  font-family: var(--sans); color: #262626; font-size: .97rem; line-height: 1.65;
}}
[data-testid="stCaptionContainer"] p {{ color: var(--ink-3) !important; font-size: .82rem !important;
                                       line-height: 1.5; }}
[data-testid="stHeadingWithActionElements"] a {{ display: none; }}  /* hide anchor icons */

/* tables and expanders: hairlines only */
[data-testid="stDataFrame"], [data-testid="stTable"] {{ border: 1px solid var(--rule);
                                                        border-radius: 4px; }}
[data-testid="stExpander"] details {{ border: 1px solid var(--rule) !important; border-radius: 4px;
                                       box-shadow: none !important; }}
[data-testid="stExpander"] summary p {{ font-weight: 500; color: var(--ink); }}
[data-testid="stAlertContainer"] {{ border-radius: 4px; }}
[data-testid="stPlotlyChart"] {{ margin-top: .3rem; }}

/* site components */
.cl-kicker {{ font-family: var(--sans); font-size: .75rem; font-weight: 600; letter-spacing: .08em;
             text-transform: uppercase; color: var(--red); margin: 0 0 .4rem; }}
.cl-lede {{ font-family: var(--sans); font-size: 1.12rem; line-height: 1.6; color: var(--ink-2);
           max-width: 46rem; margin: .2rem 0 1.6rem; }}
.cl-stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            border-top: 1px solid var(--ink); border-bottom: 1px solid var(--rule);
            margin: 1.2rem 0 .6rem; }}
.cl-stat {{ padding: 1.05rem 1.2rem 1.1rem 0; }}
.cl-stat + .cl-stat {{ padding-left: 1.2rem; border-left: 1px solid var(--rule); }}
@media (max-width: 640px) {{
  .cl-stat + .cl-stat {{ border-left: none; padding-left: 0; border-top: 1px solid var(--rule); }}
}}
.cl-stat .v {{ font-family: var(--serif); font-size: 1.9rem; font-weight: 600; color: var(--ink);
              line-height: 1.15; }}
.cl-stat .v small {{ font-family: var(--sans); font-size: .8rem; font-weight: 500;
                    color: var(--ink-3); margin-left: .3rem; }}
.cl-stat .l {{ font-family: var(--sans); font-size: .82rem; font-weight: 600; color: var(--ink);
              margin-top: .25rem; }}
.cl-stat .d {{ font-family: var(--sans); font-size: .8rem; color: var(--ink-3); margin-top: .15rem;
              line-height: 1.45; }}
.cl-note {{ font-family: var(--sans); font-size: .86rem; line-height: 1.55; color: var(--ink-2);
           border-left: 2px solid var(--red); padding: .15rem 0 .15rem .9rem; margin: .8rem 0 1.2rem; }}
.cl-steps {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 1.6rem; margin: .8rem 0 .4rem; }}
.cl-step .n {{ font-family: var(--serif); font-size: 1rem; color: var(--red); font-weight: 600; }}
.cl-step .t {{ font-family: var(--sans); font-weight: 600; color: var(--ink); margin: .2rem 0 .3rem; }}
.cl-step .b {{ font-family: var(--sans); font-size: .9rem; color: var(--ink-2); line-height: 1.55; }}
.cl-table {{ width: 100%; border-collapse: collapse; font-family: var(--sans); font-size: .88rem;
            margin: .6rem 0 .4rem; }}
.cl-table th {{ text-align: right; font-weight: 600; color: var(--ink-2); padding: .5rem .6rem;
               border-bottom: 1px solid var(--ink); font-size: .8rem; }}
.cl-table td {{ text-align: right; padding: .5rem .6rem; border-bottom: 1px solid var(--rule);
               color: var(--ink); font-variant-numeric: tabular-nums; }}
.cl-table th:first-child, .cl-table td:first-child {{ text-align: left; padding-left: 0; }}
.cl-table tr.hl td {{ font-weight: 600; }}
.cl-table td .ci {{ color: var(--ink-3); font-size: .8rem; margin-left: .35rem; }}
.cl-scroll {{ overflow-x: auto; }}
.cl-chart-title {{ font-family: var(--serif); font-size: 1.05rem; font-weight: 600; color: var(--ink);
                  margin: 1rem 0 -.2rem; line-height: 1.3; }}
@media (max-width: 640px) {{
  .cl-table {{ font-size: .76rem; }}
  .cl-table th, .cl-table td {{ padding: .4rem .3rem; }}
  .cl-table td .ci {{ display: block; margin-left: 0; }}
  h1 {{ font-size: 2rem !important; }}
}}
.cl-footer {{ font-family: var(--sans); font-size: .78rem; color: var(--ink-3);
             border-top: 1px solid var(--rule); margin-top: 3rem; padding-top: 1rem; }}
</style>
"""

SETUP = """Required research artifacts are missing. From the project root run:

```bash
source .venv/bin/activate
creditlab all          # fetch -> prepare -> train -> evaluate -> simulate -> policy -> uncertainty -> report
```

then reload this page."""

HYPOTHETICAL_BANNER = (
    "**Hypothetical.** LGD, EAD, revenue and stress parameters are assumptions, not observed "
    "values. Losses are one-month, in NTD, and are not observed bank losses. Research use only; "
    "not a lending or account-management system."
)


# ----------------------------------------------------------------------------- data access
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


# ----------------------------------------------------------------------------- formatting
def m(v: float, digits: int = 1) -> str:
    """NTD millions."""
    return f"{v / 1e6:,.{digits}f}M"


def pct(v: float, digits: int = 1) -> str:
    return f"{100 * v:.{digits}f}%"


# ----------------------------------------------------------------------------- components
def apply_theme(page_title: str | None = None) -> None:
    """Inject the site stylesheet and header logo. Presentation only."""
    import streamlit as st

    if LOGO.exists():
        st.logo(str(LOGO), size="large")
    st.html(_CSS)


def page_header(kicker: str, title: str, lede: str) -> None:
    import streamlit as st

    st.html(f'<p class="cl-kicker">{html.escape(kicker)}</p>')
    st.title(title, anchor=False)
    st.html(f'<p class="cl-lede">{lede}</p>')


def stats(items: list[tuple[str, str, str]], small: dict[int, str] | None = None) -> None:
    """A row of headline numbers: (value, label, description). Values come from artifacts."""
    import streamlit as st

    cells = []
    for i, (v, label, desc) in enumerate(items):
        sm = f"<small>{html.escape(small[i])}</small>" if small and i in small else ""
        cells.append(f'<div class="cl-stat"><div class="v">{html.escape(v)}{sm}</div>'
                     f'<div class="l">{html.escape(label)}</div>'
                     f'<div class="d">{desc}</div></div>')
    st.html(f'<div class="cl-stats">{"".join(cells)}</div>')


def note(text: str) -> None:
    import streamlit as st

    st.html(f'<div class="cl-note">{text}</div>')


def steps(items: list[tuple[str, str]]) -> None:
    import streamlit as st

    cells = "".join(f'<div class="cl-step"><div class="n">{i:02d}</div>'
                    f'<div class="t">{html.escape(t)}</div><div class="b">{b}</div></div>'
                    for i, (t, b) in enumerate(items, 1))
    st.html(f'<div class="cl-steps">{cells}</div>')


def table(header: list[str], rows: list[list[str]], highlight: set[int] | None = None) -> None:
    """Small, typographic table (pre-formatted strings; HTML allowed in cells)."""
    import streamlit as st

    th = "".join(f"<th>{h}</th>" for h in header)
    body = "".join(
        f'<tr class="{"hl" if highlight and i in highlight else ""}">'
        + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for i, r in enumerate(rows))
    st.html(f'<div class="cl-scroll"><table class="cl-table"><thead><tr>{th}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>")


def footer(cfg: Config) -> None:
    import streamlit as st

    try:
        man = read(cfg, "metrics/run_manifest.json")
        commit = (man["environment"].get("git_commit") or "not recorded")[:10]
        fp = man.get("config_fingerprint", "?")
    except Exception:
        commit, fp = "not recorded", "?"
    st.html(f'<div class="cl-footer">CreditLab · research project, not an underwriting system · '
            f'numbers read from saved pipeline artifacts (config {fp}, code {commit}).</div>')


# ----------------------------------------------------------------------------- charts
def chart(fig, title: str, height: int = 360, container=None) -> None:
    """Render a Plotly figure under an HTML title (wraps on narrow screens, unlike Plotly titles)."""
    import streamlit as st

    target = container or st
    target.html(f'<div class="cl-chart-title">{html.escape(title)}</div>')
    target.plotly_chart(style(fig, height), width="stretch", config=PLOTLY_CONFIG)


def style(fig, height: int = 360, title: str | None = None):
    fig.update_layout(
        template="plotly_white", height=height,
        paper_bgcolor=PAPER, plot_bgcolor=PAPER,
        title={"text": title, "x": 0, "xanchor": "left", "y": 0.97, "yanchor": "top"}
        if title else None,
        font={"family": "Inter, -apple-system, Segoe UI, sans-serif", "size": 12, "color": INK_2},
        title_font={"family": "Source Serif 4, Georgia, serif", "color": INK, "size": 16},
        margin={"l": 8, "r": 8, "t": 78 if title else 30, "b": 8},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0, "font": {"size": 12},
                "bgcolor": "rgba(0,0,0,0)"},
        hoverlabel={"bgcolor": "white", "bordercolor": RULE, "font": {"color": INK}},
        bargap=0.12,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=RULE, ticks="outside",
                     tickcolor=RULE, title_font={"size": 12})
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=RULE, title_font={"size": 12})
    fig.update_traces(marker_line_color="white", marker_line_width=0.5,
                      selector=lambda t: t.type in ("bar", "histogram"))
    return fig


__all__ = ["BLUE", "ENTITY_COLORS", "chart", "HYPOTHETICAL_BANNER", "INK", "INK_2", "NEUTRAL", "PLOTLY_CONFIG",
           "RED", "SERIES", "SETUP", "apply_theme", "config", "doc", "figure_path", "footer", "m",
           "missing", "note", "page_header", "pct", "read", "read_csv", "stats", "steps", "style",
           "table"]
