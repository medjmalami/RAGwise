#!/usr/bin/env python3
"""
Push a RAGAS-generated testset (CSV) into Langfuse as a dataset.
"""

import argparse
import ast
import math
import sys

import numpy as np
import pandas as pd
from langfuse import Langfuse

from app.config import settings


def parse_reference_contexts(val):
    parsed = val
    if isinstance(val, str):
        parsed = ast.literal_eval(val)
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], list):
        parsed = [c for sub in parsed for c in sub]
    return [json_safe(c) for c in (parsed or [])]


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to the RAGAS testset CSV")
    parser.add_argument(
        "--dataset-name",
        default="ragwise_ragas_testset",
        help="Target Langfuse dataset name",
    )
    parser.add_argument(
        "--description",
        default="RAGAS-generated testset from arXiv chunk sample",
        help="Description shown on the Langfuse dataset",
    )
    args = parser.parse_args()

    # Read as object dtype and replace every NaN with Python None so httpx's
    # strict JSON encoder (allow_nan=False) never sees NaN/Infinity.
    df = pd.read_csv(args.csv_path).astype(object).where(pd.notna, None)
    print(f"Loaded {len(df)} rows from {args.csv_path}")

    df["reference_contexts"] = df["reference_contexts"].apply(parse_reference_contexts)

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
        metadata={"source": "ragas", "n_rows": len(df)},
    )
    print(f"Created (or reused) dataset '{args.dataset_name}'")

    try:
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            metadata = {
                "reference_contexts": row["reference_contexts"],
                "synthesizer_name": row.get("synthesizer_name"),
                "persona_name": row.get("persona_name"),
                "query_style": row.get("query_style"),
                "query_length": row.get("query_length"),
            }
            # Drop None-valued keys to keep the dataset item tidy.
            metadata = {k: v for k, v in metadata.items() if v is not None}

            client.create_dataset_item(
                dataset_name=args.dataset_name,
                input=row["user_input"],
                expected_output=row["reference"],
                metadata=metadata,
            )
            if i % 20 == 0:
                print(f"  uploaded {i}/{len(df)} items...")
    finally:
        client.flush()

    print(f"Done — {len(df)} items uploaded to dataset '{args.dataset_name}'.")


if __name__ == "__main__":
    main()
