"""Milestone 0: configuration, seeding and CLI wiring."""

from __future__ import annotations

import subprocess
import sys

import pytest

from creditlab.cli import STAGES, build_parser
from creditlab.utils.config import PROJECT_ROOT, load_config


def test_default_config_has_required_sections(cfg):
    for key in ["seed", "paths", "data", "features", "split", "models", "calibration",
                "risk", "simulation", "policy"]:
        assert key in cfg.raw
    s = cfg["split"]
    assert abs(s["train"] + s["validation"] + s["test"] - 1.0) < 1e-12


def test_fast_config_inherits_and_overrides(cfg, fast_cfg):
    assert fast_cfg.seed == cfg.seed
    assert fast_cfg["simulation"]["n_sims"] < cfg["simulation"]["n_sims"]
    # inherited, not overridden
    assert fast_cfg["risk"] == cfg["risk"]
    assert fast_cfg.artifacts != cfg.artifacts, "fast runs must not overwrite research artifacts"


def test_sub_seeds_deterministic_and_distinct(cfg):
    assert cfg.sub_seed("split") == load_config().sub_seed("split")
    assert cfg.sub_seed("split") != cfg.sub_seed("simulation")
    a = cfg.rng("x").standard_normal(5)
    b = cfg.rng("x").standard_normal(5)
    assert (a == b).all()


def test_sensitive_fields_excluded_from_primary_model(cfg):
    excluded = set(cfg["features"]["exclude_sensitive"]) | set(cfg["features"]["exclude_proxy"])
    assert {"SEX", "AGE", "MARRIAGE", "EDUCATION"} <= excluded


def test_parser_knows_every_stage():
    parser = build_parser()
    for stage in [*STAGES, "all"]:
        ns = parser.parse_args([stage])
        assert ns.stage == stage


@pytest.mark.parametrize("stage", ["--help", "fetch --help", "train --help", "simulate --help"])
def test_cli_help_runs(stage):
    out = subprocess.run(
        [sys.executable, "-m", "creditlab.cli", *stage.split()],
        capture_output=True, text=True, cwd=PROJECT_ROOT, check=False,
    )
    assert out.returncode == 0, out.stderr
    assert "usage: creditlab" in out.stdout
