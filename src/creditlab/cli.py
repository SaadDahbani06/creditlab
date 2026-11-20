"""Command-line interface: `creditlab <stage> [--config PATH]`.

Stages run in this order (``creditlab all`` runs every stage):

  fetch     download UCI dataset 350 (or load a local copy) into data/raw/, immutable
  prepare   validate schema, build processed dataset + persisted split, run SQL analysis
  train     tune (training CV only), fit raw + calibrated models on the training split
  evaluate  select model/calibration/threshold on VALIDATION, then score the TEST split once
  simulate  expected loss, Monte Carlo VaR/ES, stress scenarios, sensitivities
  policy    hypothetical decision-policy comparison
  report    figures, tables and auto-generated report sections
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable

from creditlab.utils.logging import get_logger, setup_logging

log = get_logger("creditlab.cli")


def _fetch(cfg, args) -> None:
    from creditlab.data.fetch import fetch_dataset

    fetch_dataset(cfg, force=args.force, local_file=args.local_file)


def _prepare(cfg, args) -> None:
    from creditlab.data.preprocess import run_prepare
    from creditlab.data.sql_analysis import run_sql_analysis

    run_prepare(cfg)
    run_sql_analysis(cfg)


def _train(cfg, args) -> None:
    from creditlab.models.train import run_train

    run_train(cfg)


def _evaluate(cfg, args) -> None:
    from creditlab.models.evaluate import run_evaluate

    run_evaluate(cfg)


def _simulate(cfg, args) -> None:
    from creditlab.risk.stress import run_simulate

    run_simulate(cfg)


def _policy(cfg, args) -> None:
    from creditlab.decisions.compare import run_policy

    run_policy(cfg)


def _report(cfg, args) -> None:
    from creditlab.reporting.generate import run_report

    run_report(cfg)


STAGES: dict[str, tuple[Callable, str]] = {
    "fetch": (_fetch, "Download UCI dataset 350 (or load a local copy) into data/raw/"),
    "prepare": (_prepare, "Validate data, build processed dataset and split, run SQL analysis"),
    "train": (_train, "Tune and fit baseline, logistic and XGBoost models (+ calibration)"),
    "evaluate": (_evaluate, "Validation-based selection, then one-time test evaluation"),
    "simulate": (_simulate, "Expected loss, Monte Carlo VaR/ES and stress scenarios"),
    "policy": (_policy, "Hypothetical decision-policy comparison"),
    "report": (_report, "Generate figures, tables and report sections"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="creditlab",
        description="CREDITLAB research pipeline. Research use only: not an underwriting system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--log-level", default="INFO", help="DEBUG, INFO, WARNING (default INFO)")
    sub = parser.add_subparsers(dest="stage", required=True, metavar="STAGE")

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.add_argument(
            "--config", default=None,
            help="YAML config (default configs/default.yaml; configs/fast.yaml for smoke tests)",
        )
        return p

    for name, (_, help_text) in STAGES.items():
        p = add(name, help_text)
        if name == "fetch":
            p.add_argument("--force", action="store_true",
                           help="Overwrite an existing raw file (it is otherwise immutable)")
            p.add_argument("--local-file", default=None,
                           help="Path to the official UCI data.csv if programmatic download fails")
    p_all = add("all", "Run every stage in order: " + " -> ".join(STAGES))
    p_all.add_argument("--skip-fetch", action="store_true",
                       help="Assume data/raw already holds the dataset")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    from creditlab.utils.config import load_config

    cfg = load_config(args.config)
    log.info("Config %s (fingerprint %s, seed %d)", cfg.source.name, cfg.fingerprint(), cfg.seed)

    if args.stage == "all":
        order = [s for s in STAGES if not (s == "fetch" and args.skip_fetch)]
        args.force, args.local_file = False, None
    else:
        order = [args.stage]

    for stage in order:
        t0 = time.perf_counter()
        log.info("=== stage: %s ===", stage)
        STAGES[stage][0](cfg, args)
        log.info("=== %s finished in %.1fs ===", stage, time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
