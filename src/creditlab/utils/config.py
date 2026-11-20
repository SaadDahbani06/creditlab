"""Configuration loading, path resolution, seeding and run provenance."""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

TRACKED_PACKAGES = [
    "numpy", "pandas", "pyarrow", "duckdb", "ucimlrepo", "scikit-learn", "xgboost",
    "scipy", "matplotlib", "joblib", "pyyaml", "streamlit", "plotly",
]


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml(path: Path) -> dict:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    parent = raw.pop("extends", None)
    if parent:
        return _deep_merge(_load_yaml(path.parent / parent), raw)
    return raw


@dataclass(frozen=True)
class Config:
    """Thin wrapper around the merged YAML dictionary."""

    raw: dict[str, Any]
    source: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def path(self, key: str) -> Path:
        p = Path(self.raw["paths"][key])
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def raw_data_file(self) -> Path:
        return self.path("raw_dir") / self.raw["data"]["raw_filename"]

    @property
    def processed_dir(self) -> Path:
        return self.path("processed_dir")

    @property
    def artifacts(self) -> Path:
        return self.path("artifacts_dir")

    def artifact_dir(self, sub: str) -> Path:
        d = self.artifacts / sub
        d.mkdir(parents=True, exist_ok=True)
        return d

    def sub_seed(self, name: str) -> int:
        """Deterministic, named sub-seed so stages do not share random streams."""
        digest = hashlib.sha256(f"{self.seed}:{name}".encode()).hexdigest()
        return int(digest[:8], 16)

    def rng(self, name: str) -> np.random.Generator:
        return np.random.default_rng(self.sub_seed(name))

    def fingerprint(self) -> str:
        blob = json.dumps(self.raw, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.is_absolute():
        p = (Path.cwd() / p) if (Path.cwd() / p).exists() else PROJECT_ROOT / p
    return Config(raw=_load_yaml(p), source=p)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def environment_info() -> dict[str, Any]:
    versions = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
        "git_commit": _git_commit(),
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
        fh.write("\n")


def read_json(path: Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


def update_manifest(cfg: Config, stage: str, details: dict[str, Any]) -> None:
    """Append a stage record to artifacts/metrics/run_manifest.json."""
    path = cfg.artifact_dir("metrics") / "run_manifest.json"
    manifest = read_json(path) if path.exists() else {}
    try:
        manifest["config_source"] = str(cfg.source.relative_to(PROJECT_ROOT))
    except ValueError:  # config kept outside the project (e.g. a scratch re-run)
        manifest["config_source"] = str(cfg.source)
    manifest["config_fingerprint"] = cfg.fingerprint()
    manifest["config"] = cfg.raw
    manifest["seed"] = cfg.seed
    manifest["environment"] = environment_info()
    manifest.setdefault("stages", {})[stage] = {
        "completed_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **details,
    }
    write_json(path, manifest)
