"""Command-line interface."""

from __future__ import annotations

import argparse
import json

from .pipeline import run_pipeline, validate_data_only


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VN-Index regime-aware forecasting pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-data", help="Validate and profile the source data")
    validate.add_argument("--config", default="configs/default.yaml")
    for command in ["train", "backtest", "forecast", "report", "run-all"]:
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", default="configs/default.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate-data":
        result = validate_data_only(args.config)
    else:
        # Each research command is reproducible and materializes dependent stages.
        # Existing artifacts are overwritten atomically by their producing stage.
        result = run_pipeline(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
