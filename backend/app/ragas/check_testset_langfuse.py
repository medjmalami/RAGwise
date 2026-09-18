#!/usr/bin/env python3
"""
Pre-flight checks before pushing a RAGAS-generated testset to Langfuse.

Usage:
    python check_ragas_langfuse_upload.py path/to/testset.csv --dataset-name ragwise_ragas_testset

Exits non-zero if any check FAILS (env vars missing, auth fails, bad rows).
WARN-level issues (e.g. duplicate dataset name, NaNs) don't block, but you
should read them before running the real upload.
"""

import argparse
import ast
import importlib.metadata
import os
import sys

import pandas as pd
from pandas._libs.lib import i8max

from app.config import settings

REQUIRED_COLUMNS = ["user_input", "reference_contexts", "reference", "synthesizer_name"]

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

results = []  # list of (status, message)


def log(status, message):
    results.append((status, message))
    print(f"[{status}] {message}")


def check_env_vars():
    for var in ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"]:
        if getattr(settings, var.lower()):
            log(PASS, f"{var} is set")
        else:
            log(FAIL, f"{var} is NOT set")


def check_langfuse_version():
    try:
        version = importlib.metadata.version("langfuse")
        major = int(version.split(".")[0])
        if major >= 3:
            log(PASS, f"langfuse package version {version} (supports get_client())")
        else:
            log(
                FAIL,
                f"langfuse package version {version} is too old — need v3+ for get_client()",
            )
    except importlib.metadata.PackageNotFoundError:
        log(FAIL, "langfuse package is not installed")


def check_auth_and_dataset(dataset_name):
    try:
        from langfuse import Langfuse
    except ImportError:
        log(FAIL, "Could not import langfuse — is it installed?")
        return

    try:
        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_base_url,
        )
    except Exception as e:
        log(FAIL, f"Langfuse() init raised an exception: {e}")
        return

    try:
        if client.auth_check():
            log(PASS, "Langfuse auth_check() succeeded")
        else:
            log(FAIL, "Langfuse auth_check() returned False — check your keys/region")
            return
    except Exception as e:
        log(FAIL, f"auth_check() raised an exception: {e}")
        return

    try:
        client.get_dataset(dataset_name)
        log(
            WARN,
            f"Dataset '{dataset_name}' already exists — re-running the upload "
            f"will ADD duplicate items unless you dedupe or rename",
        )
    except Exception:
        log(
            PASS, f"Dataset '{dataset_name}' does not exist yet (will be created fresh)"
        )


def load_dataframe(csv_path):
    if not os.path.exists(csv_path):
        log(FAIL, f"File not found: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
        log(PASS, f"Loaded {csv_path} ({len(df)} rows)")
        return df
    except Exception as e:
        log(FAIL, f"Failed to read CSV: {e}")
        return None


def check_columns(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        log(FAIL, f"Missing expected columns: {missing}")
    else:
        log(PASS, f"All expected columns present: {REQUIRED_COLUMNS}")


def check_reference_contexts(df):
    if "reference_contexts" not in df.columns:
        return

    stringified = 0
    list_of_lists = 0
    unparseable = 0

    for val in df["reference_contexts"]:
        parsed = val
        if isinstance(val, str):
            stringified += 1
            try:
                parsed = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                unparseable += 1
                continue
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], list):
            list_of_lists += 1

    if stringified > 0:
        log(
            WARN,
            f"{stringified} rows have reference_contexts as a stringified list — "
            f"run ast.literal_eval() on the column before uploading",
        )
    else:
        log(
            PASS, "reference_contexts is already a real list type (no stringified rows)"
        )

    if unparseable > 0:
        log(
            FAIL,
            f"{unparseable} rows have reference_contexts that failed to parse as a list",
        )

    if list_of_lists > 0:
        log(
            WARN,
            f"{list_of_lists} rows have nested list-of-lists reference_contexts "
            f"(likely multi-hop) — consider flattening before upload",
        )


def check_missing_values(df):
    na_counts = df[[c for c in REQUIRED_COLUMNS if c in df.columns]].isna().sum()
    any_na = False
    for col, count in na_counts.items():
        if count > 0:
            any_na = True
            log(WARN, f"Column '{col}' has {count} missing/NaN values")
    if not any_na:
        log(PASS, "No missing values in required columns")


def check_synthesizer_breakdown(df):
    if "synthesizer_name" not in df.columns:
        return
    counts = df["synthesizer_name"].value_counts()
    breakdown = ", ".join(f"{name}={n}" for name, n in counts.items())
    log(PASS, f"Synthesizer breakdown — {breakdown}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to the RAGAS testset CSV")
    parser.add_argument(
        "--dataset-name",
        default="ragwise_ragas_testset",
        help="Target Langfuse dataset name",
    )
    args = parser.parse_args()

    print("=== Environment & credentials ===")
    check_env_vars()
    check_langfuse_version()

    print("\n=== Langfuse connectivity ===")
    check_auth_and_dataset(args.dataset_name)

    print("\n=== Data checks ===")
    df = load_dataframe(args.csv_path)
    if df is not None:
        check_columns(df)
        check_reference_contexts(df)
        check_missing_values(df)
        check_synthesizer_breakdown(df)

    print("\n=== Summary ===")
    fails = [m for s, m in results if s == FAIL]
    warns = [m for s, m in results if s == WARN]
    print(
        f"{len(results) - len(fails) - len(warns)} passed, {len(warns)} warnings, {len(fails)} failed"
    )

    if fails:
        print("\nFix the FAIL items above before running the upload script.")
        sys.exit(1)
    elif warns:
        print("\nNo blocking issues, but review the WARN items above before uploading.")
        sys.exit(0)
    else:
        print("\nAll checks passed — safe to run the upload script.")
        sys.exit(0)


if __name__ == "__main__":
    main()
