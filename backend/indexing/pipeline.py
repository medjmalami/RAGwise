"""
RAGwise — PDF -> DoclingDocument conversion pipeline
=====================================================

Goal: convert ~3,069 born-digital arXiv PDFs into DoclingDocument JSON with
as little data loss as possible, on a single RTX 3050 (4GB VRAM), and be
resumable if the run crashes or OOMs partway through.

Output architecture:

    data/docling/
    ├── 1601.07124v1/
    │   ├── document.json
    │   ├── document.md
    │   └── artifacts/
    │       ├── image_000000.png
    │       └── image_000001.png
    │
    ├── 1601.07125v1/
    │   ├── document.json
    │   ├── document.md
    │   └── artifacts/
    │       └── ...
    │
    └── ...

Each paper is completely self-contained.

Design choices (why, not just what):

- OCR stays ON but not forced. Docling only runs OCR on bitmap regions by
  default. Born-digital arXiv PDFs already have a real text layer, so OCR
  mostly sits idle.

- TableFormer in ACCURATE mode + do_cell_matching=True. ACCURATE costs more
  compute than FAST, but fidelity is more important than throughput for
  research-paper tables.

- Formula + code enrichment ON. arXiv papers are dense with math and often
  contain code listings.

- Picture description via SmolVLM-256M, restricted to informative crops
  via picture_area_threshold.

- generate_page_images=False by default. Full-page rasters are unnecessary
  for this RAG pipeline. Figure/table crops are kept.

- Output is DoclingDocument JSON (image_mode=REFERENCED), not markdown.
  Markdown is exported too for quick human reading.

- Each paper gets its own output directory. JSON, Markdown and referenced
  picture artifacts all live inside that paper directory.

- Resumable by construction: if
      data/docling/<paper_id>/document.json
  already exists, that paper is skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from docling.datamodel.accelerator_options import (
    AcceleratorDevice,
    AcceleratorOptions,
)
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionVlmOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

log = logging.getLogger("ragwise.docling_pipeline")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    manifest_path: Path

    # Fidelity
    table_mode: TableFormerMode = TableFormerMode.FAST
    do_formula_enrichment: bool = False
    do_code_enrichment: bool = False
    do_picture_description: bool = True

    # Only describe sufficiently large/informative figures.
    picture_area_threshold: float = 0.02

    # Image generation
    images_scale: float = 2.0
    generate_page_images: bool = False

    # Hardware
    device: AcceleratorDevice = AcceleratorDevice.CUDA
    num_threads: int = 4
    use_flash_attention2: bool = False

    # Resume
    skip_existing: bool = True
    limit: int | None = None


# --------------------------------------------------------------------------- #
# Converter construction
# --------------------------------------------------------------------------- #


def build_converter(cfg: PipelineConfig) -> DocumentConverter:
    """
    Build one long-lived DocumentConverter.

    The converter is reused for the entire batch so Docling's models are
    loaded once instead of being reloaded for every PDF.
    """

    pipeline_options = PdfPipelineOptions()

    # ------------------------------------------------------------------ #
    # Hardware
    # ------------------------------------------------------------------ #

    pipeline_options.accelerator_options = AcceleratorOptions(
        device=cfg.device,
        num_threads=cfg.num_threads,
        cuda_use_flash_attention2=cfg.use_flash_attention2,
    )

    # ------------------------------------------------------------------ #
    # OCR
    # ------------------------------------------------------------------ #

    # Keep OCR enabled as a safety net.

    # Docling will use the PDF text layer when available and OCR bitmap
    # regions when necessary. We deliberately do not force full-page OCR.
    pipeline_options.do_ocr = True

    # ------------------------------------------------------------------ #
    # Tables
    # ------------------------------------------------------------------ #

    pipeline_options.do_table_structure = True

    pipeline_options.table_structure_options = TableStructureOptions(
        mode=cfg.table_mode,
        do_cell_matching=True,
    )

    # ------------------------------------------------------------------ #
    # Formula / code enrichment
    # ------------------------------------------------------------------ #

    pipeline_options.do_formula_enrichment = cfg.do_formula_enrichment

    pipeline_options.do_code_enrichment = cfg.do_code_enrichment

    # ------------------------------------------------------------------ #
    # Figures
    # ------------------------------------------------------------------ #

    pipeline_options.do_picture_classification = True

    pipeline_options.do_picture_description = cfg.do_picture_description

    pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
        repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        prompt=(
            "Describe this figure in two to three sentences. "
            "Focus on what it shows (chart type, axes, trend, "
            "diagram components) rather than aesthetics."
        ),
        picture_area_threshold=cfg.picture_area_threshold,
    )

    # Generate figure/table crops.
    #
    # These will later be referenced by document.json and document.md.
    pipeline_options.generate_picture_images = True

    # Do not generate full-page rasters.
    pipeline_options.generate_page_images = cfg.generate_page_images

    pipeline_options.images_scale = cfg.images_scale

    # ------------------------------------------------------------------ #
    # Converter
    # ------------------------------------------------------------------ #

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


# --------------------------------------------------------------------------- #
# Paper directory helpers
# --------------------------------------------------------------------------- #


def paper_dir(
    output_dir: Path,
    paper_id: str,
) -> Path:
    """
    Return the canonical directory for one paper.

    Example:

        data/docling/1601.07124v1/
    """

    return output_dir / paper_id


def document_json_path(
    output_dir: Path,
    paper_id: str,
) -> Path:
    return paper_dir(output_dir, paper_id) / "document.json"


def document_markdown_path(
    output_dir: Path,
    paper_id: str,
) -> Path:
    return paper_dir(output_dir, paper_id) / "document.md"


def artifacts_dir(
    output_dir: Path,
    paper_id: str,
) -> Path:
    return paper_dir(output_dir, paper_id) / "artifacts"


# --------------------------------------------------------------------------- #
# Manifest / resume helpers
# --------------------------------------------------------------------------- #


def already_done(
    pdf_path: Path,
    output_dir: Path,
) -> bool:
    """
    A paper is considered converted when document.json exists.

    Expected:

        data/docling/<paper_id>/document.json
    """

    paper_id = pdf_path.stem

    return document_json_path(
        output_dir,
        paper_id,
    ).exists()


def append_manifest(
    manifest_path: Path,
    record: Mapping[str, object],
) -> None:
    """
    Append one JSON record to the JSONL manifest.
    """

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with manifest_path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


# --------------------------------------------------------------------------- #
# Per-document export
# --------------------------------------------------------------------------- #


def export_result(
    conv_res,
    output_dir: Path,
) -> dict[str, object]:
    """
    Export one converted document.

    Final structure:

        data/docling/<paper_id>/
        ├── document.json
        ├── document.md
        └── artifacts/
            ├── image_000000_....png
            └── image_000001_....png

    The important part is that artifacts_dir is explicitly passed to
    Docling's serializers. This prevents Docling from deriving its own
    nested document_artifacts directory.
    """

    stem = conv_res.input.file.stem

    record: dict[str, object] = {
        "source": str(conv_res.input.file),
        "paper_id": stem,
        "status": str(conv_res.status),
        "timestamp": time.time(),
    }

    # ------------------------------------------------------------------ #
    # Failed conversion
    # ------------------------------------------------------------------ #

    if conv_res.status not in (
        ConversionStatus.SUCCESS,
        ConversionStatus.PARTIAL_SUCCESS,
    ):
        record["errors"] = [error.error_message for error in conv_res.errors]

        return record

    # ------------------------------------------------------------------ #
    # Paper directories
    # ------------------------------------------------------------------ #

    paper_output_dir = paper_dir(
        output_dir,
        stem,
    ).resolve()

    paper_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paper_artifacts_dir = artifacts_dir(
        output_dir,
        stem,
    )

    paper_artifacts_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------ #
    # Docling document
    # ------------------------------------------------------------------ #

    doc = conv_res.document

    # ------------------------------------------------------------------ #
    # JSON — source of truth
    # ------------------------------------------------------------------ #

    json_path = document_json_path(
        output_dir,
        stem,
    )

    doc.save_as_json(
        json_path,
        artifacts_dir=paper_artifacts_dir,
        image_mode=ImageRefMode.REFERENCED,
    )

    # ------------------------------------------------------------------ #
    # Markdown — human-readable representation
    # ------------------------------------------------------------------ #

    md_path = document_markdown_path(
        output_dir,
        stem,
    )

    doc.save_as_markdown(
        md_path,
        artifacts_dir=paper_artifacts_dir,
        image_mode=ImageRefMode.REFERENCED,
    )

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #

    record["output_dir"] = str(paper_output_dir)

    record["output_json"] = str(json_path)

    record["output_markdown"] = str(md_path)

    record["artifacts_dir"] = str(paper_artifacts_dir)

    record["num_pages"] = len(doc.pages)
    record["num_tables"] = len(doc.tables)
    record["num_pictures"] = len(doc.pictures)

    # ------------------------------------------------------------------ #
    # Partial-success errors
    # ------------------------------------------------------------------ #

    if conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
        record["errors"] = [error.error_message for error in conv_res.errors]

    # ------------------------------------------------------------------ #
    # Sanity check
    # ------------------------------------------------------------------ #

    text_len = len(doc.export_to_markdown())

    num_pages = max(
        record["num_pages"],
        1,
    )

    chars_per_page = text_len / num_pages

    if chars_per_page < 200:
        record["warning"] = (
            f"low text density: {chars_per_page:.0f} chars/page — review manually"
        )

        log.warning(
            "%s: %s",
            stem,
            record["warning"],
        )

    return record


# --------------------------------------------------------------------------- #
# Batch runner
# --------------------------------------------------------------------------- #


def collect_pending(
    input_dir: Path,
    output_dir: Path,
    cfg: PipelineConfig,
) -> list[Path]:
    """
    Find PDFs that still need conversion.
    """

    pdfs = sorted(input_dir.glob("*.pdf"))

    if cfg.skip_existing:
        pending = [
            pdf
            for pdf in pdfs
            if not already_done(
                pdf,
                output_dir,
            )
        ]

        skipped = len(pdfs) - len(pending)

        if skipped:
            log.info(
                "Skipping %d already-converted files",
                skipped,
            )

    else:
        pending = pdfs

    if cfg.limit is not None:
        pending = pending[: cfg.limit]

    return pending


def run_batch(
    cfg: PipelineConfig,
) -> None:
    """
    Run the complete PDF -> Docling pipeline.
    """

    # Make all paths absolute.
    # Docling resolves referenced artifacts relative to the output document
    # when given relative paths, which causes nested data/docling/... paths.
    cfg.input_dir = cfg.input_dir.resolve()
    cfg.output_dir = cfg.output_dir.resolve()
    cfg.manifest_path = cfg.manifest_path.resolve()

    cfg.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pending = collect_pending(
        cfg.input_dir,
        cfg.output_dir,
        cfg,
    )

    log.info(
        "%d PDFs to convert",
        len(pending),
    )

    if not pending:
        return

    # ------------------------------------------------------------------ #
    # Build one long-lived converter
    # ------------------------------------------------------------------ #

    converter = build_converter(cfg)

    n_ok = 0
    n_partial = 0
    n_fail = 0

    start = time.time()

    # Keep the converter alive for the entire batch so models are loaded
    # once instead of once per PDF.
    results = converter.convert_all(
        pending,
        raises_on_error=False,
    )

    # ------------------------------------------------------------------ #
    # Export results
    # ------------------------------------------------------------------ #

    for i, conv_res in enumerate(
        results,
        start=1,
    ):
        stem = conv_res.input.file.stem

        try:
            record = export_result(
                conv_res,
                cfg.output_dir,
            )

        except Exception:
            log.error(
                "Export failed for %s:\n%s",
                stem,
                traceback.format_exc(),
            )

            record = {
                "source": str(conv_res.input.file),
                "paper_id": stem,
                "status": "EXPORT_FAILURE",
                "timestamp": time.time(),
            }

        append_manifest(
            cfg.manifest_path,
            record,
        )

        # -------------------------------------------------------------- #
        # Statistics
        # -------------------------------------------------------------- #

        status = record["status"]

        if status == str(ConversionStatus.SUCCESS):
            n_ok += 1

        elif status == str(ConversionStatus.PARTIAL_SUCCESS):
            n_partial += 1

            log.warning(
                "%s converted with issues: %s",
                stem,
                record.get("errors"),
            )

        else:
            n_fail += 1

            log.error(
                "%s failed: %s",
                stem,
                record.get(
                    "errors",
                    status,
                ),
            )

        # -------------------------------------------------------------- #
        # Progress
        # -------------------------------------------------------------- #

        if i % 25 == 0 or i == len(pending):
            elapsed = time.time() - start

            log.info(
                "[%d/%d] ok=%d partial=%d fail=%d (%.1fs/doc avg)",
                i,
                len(pending),
                n_ok,
                n_partial,
                n_fail,
                elapsed / i,
            )

    # ------------------------------------------------------------------ #
    # Final summary
    # ------------------------------------------------------------------ #

    log.info(
        "Done. ok=%d partial=%d fail=%d out of %d",
        n_ok,
        n_partial,
        n_fail,
        len(pending),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("arxiv_rag_papers"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/docling"),
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest.jsonl"),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke-test on N files",
    )

    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-process papers even if document.json exists",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU (debugging only)",
    )

    args = parser.parse_args()

    cfg = PipelineConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        skip_existing=not args.no_skip_existing,
        limit=args.limit,
        device=(AcceleratorDevice.CPU if args.cpu else AcceleratorDevice.CUDA),
    )

    run_batch(cfg)


if __name__ == "__main__":
    main()
