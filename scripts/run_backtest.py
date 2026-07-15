"""Materialize the reproducible backtest and all prerequisites."""

import argparse

from vnindex_model.pipeline import run_pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
args = parser.parse_args()
run_pipeline(args.config)
