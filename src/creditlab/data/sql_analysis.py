"""Build a local DuckDB analytical database and run the named queries in sql/analysis.sql."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from creditlab.data.preprocess import processed_path
from creditlab.utils.config import Config, update_manifest, write_json
from creditlab.utils.logging import get_logger

log = get_logger(__name__)

_HEADER = re.compile(r"^--\s*name:\s*(\w+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class NamedQuery:
    name: str
    description: str
    sql: str


def parse_sql_file(path: Path) -> list[NamedQuery]:
    text = Path(path).read_text()
    matches = list(_HEADER.finditer(text))
    queries = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end]
        desc_match = re.search(r"^--\s*description:\s*(.+)$", body, re.MULTILINE)
        sql_lines = [ln for ln in body.splitlines() if not ln.strip().startswith("--")]
        sql = "\n".join(sql_lines).strip().rstrip(";")
        queries.append(NamedQuery(m.group(1), desc_match.group(1).strip() if desc_match else "",
                                  sql))
    names = [q.name for q in queries]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate query names in {path}")
    return queries


def connect(cfg: Config, database: str | None = None) -> duckdb.DuckDBPyConnection:
    """Open (or create) the DuckDB database and (re)build the accounts table and views."""
    db_path = database or str(cfg.processed_dir / "creditlab.duckdb")
    con = duckdb.connect(db_path)
    parquet = processed_path(cfg).replace("'", "''")
    con.execute(f"CREATE OR REPLACE TABLE accounts AS SELECT * FROM read_parquet('{parquet}')")
    con.execute("CREATE OR REPLACE VIEW train_accounts AS "
                "SELECT * FROM accounts WHERE split = 'train'")
    return con


def run_queries(con: duckdb.DuckDBPyConnection, queries: list[NamedQuery]) -> dict[str, pd.DataFrame]:
    results = {}
    for q in queries:
        results[q.name] = con.execute(q.sql).df()
    return results


def run_sql_analysis(cfg: Config) -> dict[str, pd.DataFrame]:
    queries = parse_sql_file(cfg.path("sql_file"))
    con = connect(cfg)
    try:
        results = run_queries(con, queries)
    finally:
        con.close()
    out_dir = cfg.artifact_dir("metrics") / "sql"
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for q in queries:
        df = results[q.name]
        df.to_csv(out_dir / f"{q.name}.csv", index=False)
        index.append({"name": q.name, "description": q.description, "rows": len(df)})
        log.info("SQL %-30s -> %d rows", q.name, len(df))
    write_json(out_dir / "index.json", index)

    from creditlab.reporting.eda import make_eda_figures

    make_eda_figures(cfg, results)
    update_manifest(cfg, "sql", {"queries": [q.name for q in queries]})
    return results
