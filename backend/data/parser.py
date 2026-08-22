#!/usr/bin/env python3
"""
parse_arxiv_html.py

Stage 1 of the RAGwise ingestion pipeline: parse arXiv HTML (native arXiv HTML
or ar5iv fallback) into DoclingDocument JSON, ready for a downstream chunking
script to load directly (no re-parsing of HTML).

Usage:
    uv run parse_arxiv_html.py --input-dir ./html --output-dir ./parsed
    uv run parse_arxiv_html.py --input-dir ./html --output-dir ./parsed --workers 8
    uv run parse_arxiv_html.py --input-dir ./html --output-dir ./parsed --limit 50 --no-resume

Output layout:
    <output-dir>/json/<arxiv_id>.json       <- DoclingDocument, native format (pipeline input)
    <output-dir>/markdown/<arxiv_id>.md     <- human-readable debug copy, NOT used downstream
    <output-dir>/manifest.csv               <- one row per file: status, timing, char count, error
    <output-dir>/failures.log               <- full tracebacks for failed conversions

Design notes:
    - HTML parsing in Docling does not need the layout/OCR ML stack that PDF
      parsing needs, so this is CPU-bound and safe to parallelize across
      processes. Each worker builds its own DocumentConverter once (via the
      pool initializer) rather than per-file.
    - `allowed_formats=[InputFormat.HTML]` keeps the converter from wasting
      time initializing pipelines for formats you're not using.
    - Resumable: files that already have a JSON output are skipped unless
      --no-resume is passed. With ~3000 papers you will want to re-run this
      after fixing a bug or after a crash without redoing everything.
"""

import argparse
import csv
import logging
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from docling.document_converter import DocumentConverter
from tqdm import tqdm

# ── Hardcoded paths ──────────────────────────────────────────────────────
# Anchored to this script's own location (Path(__file__).resolve().parent),
# NOT to whatever directory you happen to run the command from. That means
# `python parse_arxiv_html.py` gives the same result whether you run it from
# ~/ragwise or ~/ragwise/scripts or anywhere else.
#
# If your HTML corpus lives somewhere unrelated to the repo (e.g. a separate
# data drive), just replace these with absolute paths instead — that's
# simpler than relative paths once data and code live in different places.
SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_DIR = SCRIPT_DIR / ".." / "arxiv_rag_papers" / "html"
OUTPUT_DIR = SCRIPT_DIR / ".." / "arxiv_rag_papers" / "parsed"
WORKERS = None  # None -> os.cpu_count()

# ── Worker-global state ──────────────────────────────────────────────────
# Instantiated once per worker process via the pool initializer, not per file.
_converter: Optional[DocumentConverter] = None


def _init_worker() -> None:
    global _converter
    from docling.datamodel.base_models import InputFormat

    _converter = DocumentConverter(allowed_formats=[InputFormat.HTML])


@dataclass
class ParseResult:
    arxiv_id: str
    status: str  # "success" | "partial_success" | "failure"
    char_count: int
    elapsed_s: float
    error: str = ""


def _convert_one(html_path_str: str, json_dir_str: str, md_dir_str: str) -> ParseResult:
    """Runs in a worker process. Converts one HTML file and writes JSON + MD."""
    global _converter
    html_path = Path(html_path_str)
    json_dir = Path(json_dir_str)
    md_dir = Path(md_dir_str)
    arxiv_id = html_path.stem

    t0 = time.time()
    try:
        assert _converter is not None, (
            "Converter not initialized in this worker process — "
            "_init_worker() should have set it via ProcessPoolExecutor(initializer=...)."
        )
        result = _converter.convert(html_path)

        # Docling reports a per-document conversion status; treat anything
        # other than a hard failure as usable, but record it distinctly.
        status_name = getattr(result.status, "name", str(result.status)).lower()
        if "fail" in status_name:
            return ParseResult(
                arxiv_id,
                "failure",
                0,
                time.time() - t0,
                error=f"ConversionStatus={status_name}",
            )

        doc = result.document

        json_path = json_dir / f"{arxiv_id}.json"
        doc.save_as_json(json_path)

        md_path = md_dir / f"{arxiv_id}.md"
        md_text = doc.export_to_markdown()
        md_path.write_text(md_text, encoding="utf-8")

        final_status = "success" if "success" in status_name else "partial_success"
        return ParseResult(arxiv_id, final_status, len(md_text), time.time() - t0)

    except Exception as e:
        tb = traceback.format_exc()
        return ParseResult(
            arxiv_id,
            "failure",
            0,
            time.time() - t0,
            error=f"{type(e).__name__}: {e}\n{tb}",
        )


def parse_corpus(
    input_dir: Path,
    output_dir: Path,
    workers: int,
    resume: bool,
    limit: Optional[int],
) -> None:
    json_dir = output_dir / "json"
    md_dir = output_dir / "markdown"
    json_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.csv"
    failures_path = output_dir / "failures.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "parse_run.log"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger("parse_arxiv_html")

    all_html = sorted(input_dir.glob("*.html"))
    if not all_html:
        log.warning(f"No .html files found in {input_dir}")
        return

    if resume:
        done_ids = {p.stem for p in json_dir.glob("*.json")}
        todo = [p for p in all_html if p.stem not in done_ids]
        log.info(f"Resume mode: {len(done_ids)} already parsed, {len(todo)} remaining")
    else:
        todo = all_html

    if limit is not None:
        todo = todo[:limit]

    log.info(f"Parsing {len(todo)} files with {workers} worker process(es)")

    manifest_exists = manifest_path.exists()
    manifest_f = open(manifest_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        manifest_f,
        fieldnames=["arxiv_id", "status", "char_count", "elapsed_s", "error"],
    )
    if not manifest_exists:
        writer.writeheader()

    n_success = n_partial = n_failure = 0

    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = {
            pool.submit(_convert_one, str(p), str(json_dir), str(md_dir)): p
            for p in todo
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Parsing HTML"):
            res = fut.result()
            row = asdict(res)
            # Keep the CSV readable; full tracebacks go to failures.log only.
            row["error"] = res.error.splitlines()[0] if res.error else ""
            writer.writerow(row)
            manifest_f.flush()

            if res.status == "success":
                n_success += 1
            elif res.status == "partial_success":
                n_partial += 1
                log.warning(f"{res.arxiv_id}: partial success")
            else:
                n_failure += 1
                log.error(f"{res.arxiv_id}: FAILED")
                with open(failures_path, "a", encoding="utf-8") as ff:
                    ff.write(f"\n{'=' * 80}\n{res.arxiv_id}\n{res.error}\n")

    manifest_f.close()

    log.info(
        f"Done. success={n_success} partial={n_partial} failure={n_failure} "
        f"(see {manifest_path.name} and {failures_path.name})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse arXiv HTML into DoclingDocument JSON"
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help=f"Directory of .html files (default: {INPUT_DIR})",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Where json/, markdown/, manifest.csv go (default: {OUTPUT_DIR})",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=WORKERS,
        help="Process count (default: os.cpu_count())",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N pending files (debug)",
    )
    ap.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Reprocess files even if JSON exists",
    )
    args = ap.parse_args()

    import os

    workers = args.workers or os.cpu_count() or 4

    if not args.input_dir.exists():
        raise SystemExit(f"Input dir does not exist: {args.input_dir}")

    parse_corpus(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        workers=workers,
        resume=args.resume,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
