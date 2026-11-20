"""Milestone 7: every dashboard page renders without exceptions (headless AppTest)."""

from __future__ import annotations

import pytest

from creditlab.utils.config import PROJECT_ROOT

PAGES = [
    "app/streamlit_app.py",
    "app/overview.py",
    "app/pages/1_Data_Explorer.py",
    "app/pages/2_Model_Comparison.py",
    "app/pages/3_Calibration.py",
    "app/pages/4_Portfolio_Risk.py",
    "app/pages/5_Stress_Testing.py",
    "app/pages/6_Policy_Lab.py",
    "app/pages/7_Methodology.py",
]

pytestmark = [pytest.mark.needs_artifacts, pytest.mark.slow]


@pytest.fixture(scope="module", autouse=True)
def _require_artifacts(cfg):
    if not (cfg.artifacts / "metrics" / "policy" / "policy_comparison.csv").exists():
        pytest.skip("research artifacts missing: run `creditlab all`")


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT_ROOT / page), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title, "page should render a title"


def test_stress_page_reacts_to_inputs():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(PROJECT_ROOT / "app/pages/5_Stress_Testing.py"), default_timeout=120)
    at.run()
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
    for page in PAGES[1:-1]:
        at = AppTest.from_file(str(PROJECT_ROOT / page), default_timeout=60)
        at.run()
        assert not at.exception, page
        assert any("creditlab all" in e.value for e in at.error), page
