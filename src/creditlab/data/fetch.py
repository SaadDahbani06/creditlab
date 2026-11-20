"""Download (or locally load) UCI dataset 350 and store an immutable raw copy.

The raw file keeps the official column names (ID, X1..X23, Y) exactly as served by UCI.
Semantic names (LIMIT_BAL, PAY_0, ...) are applied later in ``preprocess.to_semantic``.
"""

from __future__ import annotations

import os
import stat
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import pandas as pd

from creditlab.utils.config import Config, read_json, sha256_file, update_manifest, write_json
from creditlab.utils.logging import get_logger

log = get_logger(__name__)

RAW_COLUMNS = ["ID", *[f"X{i}" for i in range(1, 24)], "Y"]
UCI_PAGE = "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients"
UCI_CSV = "https://archive.ics.uci.edu/static/public/350/data.csv"
DOI = "10.24432/C55S3H"
LICENSE = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)"
CITATION = (
    "Yeh, I. (2009). Default of Credit Card Clients [Dataset]. "
    "UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H"
)
# Content hash (pandas row hashes of the int64 table, see ``content_sha256``) of the official
# static CSV as downloaded on 2026-11-19. Byte hashes differ across serialisations (line
# endings), so provenance compares content instead. Informational, not a hard check.
OFFICIAL_CONTENT_SHA256 = "d838ea382f3e65bc930e8a44fac70f529d16a1726713159331db85b17007e571"


class DataUnavailableError(RuntimeError):
    pass


def content_sha256(df: pd.DataFrame) -> str:
    """Order-sensitive hash of table values, independent of CSV formatting."""
    import hashlib

    values = pd.util.hash_pandas_object(df[RAW_COLUMNS].astype("int64"), index=False).values
    return hashlib.sha256(values.tobytes()).hexdigest()


def provenance_path(cfg: Config) -> Path:
    return cfg.raw_data_file.with_name("uci_350_provenance.json")


def metadata_path(cfg: Config) -> Path:
    return cfg.raw_data_file.with_name("uci_350_metadata.json")


def _from_ucimlrepo(uci_id: int) -> tuple[pd.DataFrame, dict]:
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=uci_id)
    if ds.data.ids is None:
        raise DataUnavailableError("ucimlrepo returned no ID column")
    df = pd.concat([ds.data.ids, ds.data.features, ds.data.targets], axis=1)
    meta = {
        "metadata": {k: v for k, v in ds.metadata.items() if k != "additional_info"},
        "additional_info": ds.metadata.get("additional_info"),
        "variables": ds.variables.to_dict(orient="records"),
    }
    return df, meta


def _check_columns(df: pd.DataFrame, origin: str) -> pd.DataFrame:
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in RAW_COLUMNS]
    if missing or extra:
        raise DataUnavailableError(
            f"{origin}: unexpected schema (missing={missing}, extra={extra}). "
            "Expected the official UCI layout ID, X1..X23, Y."
        )
    return df[RAW_COLUMNS]


def _make_writable(path: Path) -> None:
    if path.exists():
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)


def fetch_dataset(cfg: Config, force: bool = False, local_file: str | None = None) -> Path:
    """Fetch the dataset once and keep it immutable.

    Order of sources: explicit ``local_file`` > ucimlrepo download > ``data/raw/data.csv``
    (the official file placed manually). An existing raw file is never overwritten unless
    ``force`` is set; instead its hash is checked against the recorded provenance.
    """
    dest = cfg.raw_data_file
    dest.parent.mkdir(parents=True, exist_ok=True)
    prov_file = provenance_path(cfg)

    if dest.exists() and not force:
        actual = sha256_file(dest)
        if prov_file.exists():
            expected = read_json(prov_file)["sha256"]
            if actual != expected:
                raise DataUnavailableError(
                    f"{dest} has sha256 {actual} but provenance records {expected}. "
                    "The raw file was modified; re-run `creditlab fetch --force`."
                )
        log.info("Raw data already present and unchanged: %s (sha256 %s)", dest, actual[:12])
        return dest

    meta: dict | None = None
    if local_file:
        df = _check_columns(pd.read_csv(local_file), f"local file {local_file}")
        source = {"method": "local_file", "path": str(Path(local_file).resolve())}
    else:
        try:
            df, meta = _from_ucimlrepo(cfg["data"]["uci_id"])
            df = _check_columns(df, "ucimlrepo")
            source = {"method": "ucimlrepo",
                      "ucimlrepo_version": metadata.version("ucimlrepo")}
        except Exception as exc:  # network failure, API change, etc.
            fallback = dest.parent / "data.csv"
            if not fallback.exists():
                raise DataUnavailableError(
                    f"Programmatic download failed ({exc!r}).\n"
                    f"Download the official CSV from {UCI_CSV}\n"
                    f"(dataset page {UCI_PAGE}) and either place it at {fallback} or run\n"
                    "  creditlab fetch --local-file /path/to/data.csv"
                ) from exc
            log.warning("ucimlrepo failed (%r); using manually placed %s", exc, fallback)
            df = _check_columns(pd.read_csv(fallback), str(fallback))
            source = {"method": "local_fallback", "path": str(fallback), "error": repr(exc)}

    _make_writable(dest)
    df.to_csv(dest, index=False)
    os.chmod(dest, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # read-only: raw is immutable
    digest = sha256_file(dest)

    provenance = {
        "dataset": "Default of Credit Card Clients",
        "uci_id": cfg["data"]["uci_id"],
        "page": UCI_PAGE,
        "official_csv": UCI_CSV,
        "doi": DOI,
        "license": LICENSE,
        "citation": CITATION,
        "intro_paper": "Yeh, I.-C. & Lien, C.-H. (2009). The comparisons of data mining "
                       "techniques for the predictive accuracy of probability of default of "
                       "credit card clients. Expert Systems with Applications 36(2), 2473-2480. "
                       "https://doi.org/10.1016/j.eswa.2007.12.020",
        "retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": source,
        "file": dest.name,
        "rows": len(df),
        "columns": list(df.columns),
        "sha256": digest,
        "content_sha256": content_sha256(df),
        "content_identical_to_official_csv": content_sha256(df) == OFFICIAL_CONTENT_SHA256,
    }
    write_json(prov_file, provenance)
    if meta is not None:
        write_json(metadata_path(cfg), meta)
    update_manifest(cfg, "fetch", {"raw_file": str(dest), "sha256": digest, "rows": len(df),
                                   "source": source["method"]})
    log.info("Wrote %s (%d rows, sha256 %s, content identical to official CSV: %s)",
             dest, len(df), digest[:12], provenance["content_identical_to_official_csv"])
    return dest


def load_raw(cfg: Config) -> pd.DataFrame:
    path = cfg.raw_data_file
    if not path.exists():
        raise DataUnavailableError(f"{path} not found. Run `creditlab fetch` first.")
    df = pd.read_csv(path, dtype="int64")
    return _check_columns(df, str(path))
