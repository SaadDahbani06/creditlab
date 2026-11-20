"""The research site renders without exceptions (headless AppTest).

Main site: four read-only pages behind the router. Legacy interactive pages (app/legacy/) are
kept working and tested too, because they exercise the risk and policy library functions.
"""

from __future__ import annotations

import pytest

from creditlab.utils.config import PROJECT_ROOT

SITE = ["app/site/overview.py", "app/site/modeling.py", "app/site/risk.py",
        "app/site/methodology.py"]
LEGACY = ["app/legacy/0_Overview_v1.py", "app/legacy/1_Data_Explorer.py",
          "app/legacy/2_Model_Comparison.py", "app/legacy/3_Calibration.py",
          "app/legacy/4_Portfolio_Risk.py", "app/legacy/5_Stress_Testing.py",
          "app/legacy/6_Policy_Lab.py", "app/legacy/7_Methodology.py"]

pytestmark = [pytest.mark.needs_artifacts, pytest.mark.slow]


@pytest.fixture(scope="module", autouse=True)
def _require_artifacts(cfg):
    if not (cfg.artifacts / "metrics" / "policy" / "policy_comparison.csv").exists():
        pytest.skip("research artifacts missing: run `creditlab all`")


def _run(page: str, timeout: int = 180):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT_ROOT / page), default_timeout=timeout)
    at.run()
    return at


@pytest.mark.parametrize("page", ["app/streamlit_app.py", *SITE])
def test_site_page_renders(page):
    at = _run(page)
    assert not at.exception, [e.value for e in at.exception]
    assert at.title, "page should render a title"
    assert not at.error, [e.value for e in at.error]


def test_main_site_has_no_playground_controls():
    """The research site is read-only: no sliders, seed inputs or scenario selectors."""
    for page in SITE:
        at = _run(page)
        assert not at.slider and not at.number_input and not at.selectbox, page


@pytest.mark.parametrize("page", LEGACY)
def test_legacy_page_still_renders(page):
    at = _run(page)
    assert not at.exception, [e.value for e in at.exception]


def test_legacy_stress_page_reacts_to_inputs():
    at = _run("app/legacy/5_Stress_Testing.py")
    before = [m.value for m in at.metric]
    at.slider[4].set_value(0.12).run()   # rho
    assert not at.exception
    after = [m.value for m in at.metric]
    assert before[3] != after[3], "99% VaR should change when correlation changes"


def test_missing_artifacts_show_setup_message(tmp_path, monkeypatch):
    """With an empty artifacts directory every page stops with setup instructions."""
    from streamlit.testing.v1 import AppTest

    import creditlab.reporting.dashboard as dash
    from creditlab.utils.config import Config, load_config

    base = load_config()
    raw = {**base.raw, "paths": {**base.raw["paths"], "artifacts_dir": str(tmp_path)}}
    monkeypatch.setattr(dash, "config", lambda: Config(raw=raw, source=base.source))
    for page in [*SITE[:3], *LEGACY[:-1]]:
        at = AppTest.from_file(str(PROJECT_ROOT / page), default_timeout=60)
        at.run()
        assert not at.exception, page
        assert any("creditlab all" in e.value for e in at.error), page
