#!/usr/bin/env python3
import argparse
import os
import pickle
import sys

import pandas as pd


DEFAULT_COLUMNS = [
    "model",
    "pixels",
    "image",
    "true",
    "predicted",
    "success",
    "cdiff",
    "prior_probs",
    "predicted_probs",
    "perturbation",
    "duration",
]


def _load_results(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _to_dataframe(results):
    if isinstance(results, pd.DataFrame):
        return results
    if isinstance(results, list):
        if not results:
            return pd.DataFrame(columns=DEFAULT_COLUMNS)
        first = results[0]
        if isinstance(first, dict):
            return pd.DataFrame(results)
        if isinstance(first, (list, tuple)):
            cols = DEFAULT_COLUMNS
            if len(first) != len(DEFAULT_COLUMNS):
                cols = [f"col_{i}" for i in range(len(first))]
            return pd.DataFrame(results, columns=cols)
    raise TypeError(f"Unsupported results format: {type(results)}")


def _apply_filters(df, args):
    if args.model and "model" in df.columns:
        df = df[df["model"].isin(args.model)]
    if args.pixels is not None and "pixels" in df.columns:
        df = df[df["pixels"].isin(args.pixels)]
    if args.success_only and "success" in df.columns:
        df = df[df["success"]]
    return df


def _print_summary(df):
    needed = {"model", "pixels", "success"}
    if not needed.issubset(df.columns):
        print("Summary requires columns: model, pixels, success")
        return
    summary = (
        df.groupby(["model", "pixels"])["success"]
        .mean()
        .reset_index()
        .rename(columns={"success": "success_rate"})
    )
    print(summary)


def _print_time_stats(df):
    if "duration" not in df.columns:
        print("Time stats require a 'duration' column.")
        return
    stats = df["duration"].describe(percentiles=[0.5, 0.9, 0.95])[["count", "mean", "50%", "90%", "95%", "max"]]
    print(stats)


def main():
    parser = argparse.ArgumentParser(description="View one-pixel attack results pickle.")
    parser.add_argument(
        "--file",
        default="networks/results/results.pkl",
        help="Path to results pickle file.",
    )
    parser.add_argument("--head", type=int, default=5, help="Show first N rows.")
    parser.add_argument("--tail", type=int, default=0, help="Show last N rows.")
    parser.add_argument("--model", nargs="+", help="Filter by model name(s).")
    parser.add_argument("--pixels", nargs="+", type=int, help="Filter by pixel count(s).")
    parser.add_argument("--success-only", action="store_true", help="Only show successful attacks.")
    parser.add_argument("--summary", action="store_true", help="Show success rate summary.")
    parser.add_argument("--time-stats", action="store_true", help="Show duration statistics.")

    args = parser.parse_args()

    try:
        results = _load_results(args.file)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        df = _to_dataframe(results)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    df = _apply_filters(df, args)
    print(f"rows={len(df)} cols={list(df.columns)}")

    if args.summary:
        _print_summary(df)
        return
    if args.time_stats:
        _print_time_stats(df)
        return

    if args.tail and args.tail > 0:
        print(df.tail(args.tail))
    else:
        print(df.head(args.head))


if __name__ == "__main__":
    main()
