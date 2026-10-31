#!/usr/bin/env python3
"""
Pre-flight checks before pushing a DeepEval goldens CSV to Langfuse.

Usage:
    python check_goldens_langfuse.py path/to/conversational_goldens.csv \
        --dataset-name ragwise_deepeval_conversational_goldens

Exits non-zero if any check FAILS (env vars missing, auth fails, bad rows,
unparseable JSON lists). WARN-level issues (e.g. duplicate dataset name,
large chunks) don't block, but you should read them before running the real
upload.
"""

import argparse
import importlib.metadata
import json
import os
import sys

import pandas as pd

from app.config.config import settings

REQUIRED_COLUMNS = [
    "golden_id",
    "scenario",
    "expected_outcome",
    "context_chunks",
    "context_chunk_ids",
    "context_scores",
]

LIST_COLUMNS = [
    "used_source_files",
    "context_paper_ids",
    "context_chunk_ids",
    "context_scores",
    "context_chunks",
    "evolutions",
]

# Soft per-item metadata size budget (Langfuse caps at ~512 KB per item once
# you include input + expected_output + metadata). Warn above this so users
# can decide whether to truncate context_chunks.
METADATA_SIZE_WARN_BYTES = 200_000

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
        log(FAIL, f"Missing required columns: {missing}")
    else:
        log(PASS, f"All required columns present: {REQUIRED_COLUMNS}")

    optional_missing = [
        c for c in LIST_COLUMNS if c not in df.columns and c not in REQUIRED_COLUMNS
    ]
    if optional_missing:
        log(WARN, f"Optional list columns missing (will default to []): {optional_missing}")


def check_json_list_columns(df):
    """
    DeepEval writes list columns as JSON strings. `ast.literal_eval` would
    choke on JSON `null` (e.g. context_scores='[null, 0.86]'), so the upload
    script uses `json.loads`. We replicate that here and surface any rows
    that fail to parse.
    """
    for col in LIST_COLUMNS:
        if col not in df.columns:
            continue
        unparseable = 0
        null_count = 0
        sample_error = None
        for val in df[col]:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            if not isinstance(val, str):
                continue
            s = val.strip()
            if not s or s.lower() == "nan":
                continue
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError as e:
                unparseable += 1
                if sample_error is None:
                    sample_error = f"{s[:80]!r} -> {e}"
                continue
            if parsed is None:
                null_count += 1
        if unparseable > 0:
            log(
                FAIL,
                f"Column '{col}': {unparseable} row(s) failed JSON parse. "
                f"First error: {sample_error}",
            )
        elif null_count > 0:
            log(
                WARN,
                f"Column '{col}': {null_count} row(s) parsed to null "
                f"(upload will treat as empty list)",
            )
        else:
            log(PASS, f"Column '{col}' parses cleanly as JSON list")


def check_null_in_context_scores(df):
    """
    DeepEval's context_scores frequently contains JSON null entries
    (e.g. when a chunk is the seed chunk with no retrieval score). The
    upload script coerces these to Python None, which is fine — but the
    user should know how many nulls exist.
    """
    if "context_scores" not in df.columns:
        return
    null_count = 0
    total = 0
    for val in df["context_scores"]:
        if not isinstance(val, str):
            continue
        try:
            parsed = json.loads(val)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, list):
            continue
        for v in parsed:
            total += 1
            if v is None:
                null_count += 1
    if total == 0:
        log(WARN, "context_scores has no parseable numeric entries")
    elif null_count > 0:
        log(
            PASS,
            f"context_scores: {null_count}/{total} entries are null "
            f"(coerced to Python None during upload)",
        )
    else:
        log(PASS, f"context_scores: all {total} entries are numeric (no nulls)")


def check_missing_values(df):
    present = [c for c in REQUIRED_COLUMNS if c in df.columns]
    na_counts = df[present].isna().sum()
    any_na = False
    for col, count in na_counts.items():
        if count > 0:
            any_na = True
            log(WARN, f"Column '{col}' has {count} missing/NaN values")
    if not any_na:
        log(PASS, "No missing values in required columns")


def check_unique_golden_ids(df):
    if "golden_id" not in df.columns:
        return
    dup = df["golden_id"].duplicated().sum()
    if dup > 0:
        log(FAIL, f"{dup} duplicate golden_id values found")
    else:
        log(PASS, f"All {len(df)} golden_id values are unique")


def check_scenario_breakdown(df):
    """Quick stats to confirm the conversational goldens look sane."""
    if "is_single_paper" in df.columns:
        counts = df["is_single_paper"].value_counts()
        breakdown = ", ".join(f"{name}={n}" for name, n in counts.items())
        log(PASS, f"is_single_paper breakdown — {breakdown}")
    if "n_papers_used" in df.columns:
        counts = df["n_papers_used"].value_counts()
        breakdown = ", ".join(f"{name} paper(s)={n}" for name, n in counts.items())
        log(PASS, f"n_papers_used breakdown — {breakdown}")


def check_metadata_size(df):
    """
    Estimate the per-item metadata payload that will be sent to Langfuse. If
    context_chunks is large, individual items may exceed Langfuse's per-item
    size budget. We don't fail, just warn so the user can truncate.
    """
    if "context_chunks" not in df.columns:
        return
    oversized = 0
    max_bytes = 0
    for val in df["context_chunks"]:
        try:
            parsed = json.loads(val) if isinstance(val, str) else val
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, list):
            continue
        size = len(json.dumps(parsed, ensure_ascii=False).encode("utf-8"))
        if size > METADATA_SIZE_WARN_BYTES:
            oversized += 1
        if size > max_bytes:
            max_bytes = size
    if oversized > 0:
        log(
            WARN,
            f"{oversized} row(s) have context_chunks larger than "
            f"{METADATA_SIZE_WARN_BYTES // 1024} KB (max {max_bytes // 1024} KB) — "
            f"Langfuse may reject these items; consider truncating context_chunks",
        )
    else:
        log(
            PASS,
            f"All context_chunks under {METADATA_SIZE_WARN_BYTES // 1024} KB "
            f"(max {max_bytes // 1024} KB)",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to the DeepEval goldens CSV")
    parser.add_argument(
        "--dataset-name",
        default="ragwise_deepeval_conversational_goldens",
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
        check_json_list_columns(df)
        check_null_in_context_scores(df)
        check_missing_values(df)
        check_unique_golden_ids(df)
        check_scenario_breakdown(df)
        check_metadata_size(df)

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
