#!/usr/bin/env python3
"""
Push a DeepEval-generated goldens CSV (conversational_goldens.csv) into
Langfuse as a dataset.

This is the DeepEval counterpart of `upload_testset_langfuse.py` (which was
written for RAGAS testsets). The mapping from CSV columns to Langfuse
dataset-item fields is:

    CSV column              ->  Langfuse dataset item
    ---------------------------------------------------------------
    scenario               ->  input
    expected_outcome       ->  expected_output
    golden_id              ->  metadata.golden_id  (+ external_id-like)
    n_papers_used          ->  metadata.n_papers_used
    used_source_files      ->  metadata.used_source_files      (list)
    is_single_paper        ->  metadata.is_single_paper        (bool)
    context_paper_ids      ->  metadata.context_paper_ids      (list)
    context_chunk_ids      ->  metadata.context_chunk_ids      (list)
    context_scores         ->  metadata.context_scores         (list, nulls->None)
    context_chunks         ->  metadata.context_chunks          (list of strings)
    scenario_quality       ->  metadata.scenario_quality       (float)
    evolutions             ->  metadata.evolutions             (list)

Note: DeepEval writes list columns as JSON (e.g. `[null, 0.86]`), so we parse
with `json.loads` rather than `ast.literal_eval` — `null` is not a Python
literal and would raise SyntaxError under `ast`.
"""

import argparse
import json
import math
import sys

import numpy as np
import pandas as pd
from langfuse import Langfuse

from app.config.config import settings


# Columns that DeepEval stores as JSON-encoded lists.
LIST_COLUMNS = [
    "used_source_files",
    "context_paper_ids",
    "context_chunk_ids",
    "context_scores",
    "context_chunks",
    "evolutions",
]

# Columns to drop into dataset-item metadata (everything except the two that
# become input / expected_output).
METADATA_COLUMNS = [
    "golden_id",
    "n_papers_used",
    "used_source_files",
    "is_single_paper",
    "context_paper_ids",
    "context_chunk_ids",
    "context_scores",
    "context_chunks",
    "scenario_quality",
    "evolutions",
]


def json_safe(val):
    """Make pandas/numpy values JSON-safe for httpx's strict encoder."""
    if val is None:
        return None
    if isinstance(val, (float, np.floating)):
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return val


def parse_json_list(val):
    """
    Parse a JSON-encoded list column.

    DeepEval writes these as JSON strings (e.g. '[null, 0.85]' or
    '["2312.10997v5", "2407.21059v1"]'). `null` is JSON syntax, so we have to
    use `json.loads` — `ast.literal_eval` would choke on it. NaN values inside
    `context_scores` would also be invalid JSON; we coerce them defensively.
    """
    if val is None:
        return []
    if isinstance(val, list):
        return [json_safe(v) for v in val]
    if not isinstance(val, str):
        return [json_safe(val)]
    s = val.strip()
    if not s or s.lower() == "nan":
        return []
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON list from {val!r}: {e}")
    if parsed is None:
        return []
    if not isinstance(parsed, list):
        parsed = [parsed]
    return [json_safe(v) for v in parsed]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to the DeepEval goldens CSV")
    parser.add_argument(
        "--dataset-name",
        default="ragwise_deepeval_conversational_goldens",
        help="Target Langfuse dataset name",
    )
    parser.add_argument(
        "--description",
        default="DeepEval-generated conversational goldens from arXiv chunk sample",
        help="Description shown on the Langfuse dataset",
    )
    args = parser.parse_args()

    # Read as object dtype and replace every NaN with Python None so httpx's
    # strict JSON encoder (allow_nan=False) never sees NaN/Infinity.
    df = pd.read_csv(args.csv_path).astype(object).where(pd.notna, None)
    print(f"Loaded {len(df)} rows from {args.csv_path}")

    # Parse every JSON-encoded list column up front so failures are loud and
    # early rather than mid-upload.
    for col in LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(parse_json_list)
        else:
            print(f"  WARNING: expected column '{col}' is missing — filling with []")
            df[col] = [[] for _ in range(len(df))]

    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_base_url,
    )

    if not client.auth_check():
        print("Auth check failed — aborting before creating anything.", file=sys.stderr)
        sys.exit(1)

    client.create_dataset(
        name=args.dataset_name,
        description=args.description,
        metadata={
            "source": "deepeval",
            "generator": "conversational_goldens",
            "n_rows": len(df),
        },
    )
    print(f"Created (or reused) dataset '{args.dataset_name}'")

    try:
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            metadata = {col: json_safe(row.get(col)) for col in METADATA_COLUMNS}
            # Drop None-valued top-level keys to keep the dataset item tidy.
            # We keep the list-valued fields (even if empty) because they are
            # structural — an empty `evolutions` list is meaningful.
            metadata = {
                k: v
                for k, v in metadata.items()
                if v is not None or k in LIST_COLUMNS
            }
            metadata["source"] = "deepeval"

            client.create_dataset_item(
                dataset_name=args.dataset_name,
                input=row["scenario"],
                expected_output=row["expected_outcome"],
                metadata=metadata,
            )
            if i % 20 == 0:
                print(f"  uploaded {i}/{len(df)} items...")
    finally:
        client.flush()

    print(f"Done — {len(df)} items uploaded to dataset '{args.dataset_name}'.")


if __name__ == "__main__":
    main()
