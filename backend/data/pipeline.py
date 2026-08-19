"""
RAGwise — PDF -> DoclingDocument conversion pipeline
=====================================================

Goal: convert ~3,069 born-digital arXiv PDFs into DoclingDocument JSON with
as little data loss as possible, on a single RTX 3050 (4GB VRAM), and be
resumable if the run crashes or OOMs partway through.

Design choices (why, not just what):

- OCR stays ON but not forced. Docling only runs OCR on bitmap regions by
  default (force_full_page_ocr=False). Born-digital arXiv PDFs already have
  a real text layer, so OCR mostly sits idle — but it's a free safety net
  for the rare scanned figure or rasterized page. Forcing full-page OCR on
  a digital PDF is pure downside: slower, and OCR text is strictly worse
  than the PDF's own text layer.

- TableFormer in ACCURATE mode + do_cell_matching=True. ACCURATE costs more
  compute than FAST, but you're optimizing for fidelity, not throughput,
  and research-paper tables are exactly the content you don't want mangled.
  do_cell_matching=True binds recognized cells back to the PDF's real text
  instead of letting the model regenerate cell text — more faithful for
  digital PDFs (regeneration is meant for scanned/noisy tables).

- Formula + code enrichment ON. arXiv papers are dense with math and often
  contain code listings; without these, both collapse into unstructured
  image/text noise instead of LaTeX / recognized code blocks.

- Picture description via SmolVLM-256M, restricted to "informative" crops
  via picture_area_threshold. This skips tiny logos/icons/decorative marks
  and only spends the (VRAM-constrained) VLM budget on real figures and
  diagrams — matches the earlier decision to caption informative crops
  only, not every bitmap on the page.

- generate_page_images=False by default. Full-page rasters at 2x scale
  across 3,069 papers is a lot of disk for something you likely don't need
  downstream (you're not doing ColPali-style whole-page retrieval here).
  Figure/table crops (generate_picture_images=True) are kept because
  they're small and are what a multimodal step would actually want.

- Output is DoclingDocument JSON (image_mode=REFERENCED), not markdown, as
  the source of truth. Markdown is exported too, for quick human reading,
  but it throws away bounding boxes, provenance, and table cell structure.
  Any downstream chunker (e.g. Docling's HybridChunker) should load the
  JSON / DoclingDocument object directly, not re-parse the markdown.

- Conversion is sequential, single process, using one long-lived
  DocumentConverter via convert_all(). Loading TableFormer + SmolVLM +
  CodeFormula concurrently on a 4GB card is already close to the ceiling;
  running multiple worker processes against the same GPU would just
  thrash. Throughput here comes from not reloading models per file, not
  from parallelism.

- Resumable by construction: before conversion, any PDF whose output JSON
  already exists is skipped. A JSONL manifest logs every attempt (success,
  partial, failure, OOM) so a crash mid-run costs you nothing but the
  in-flight document.
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

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionVlmOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode

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

    # Fidelity toggles — see module docstring for rationale
    table_mode: TableFormerMode = TableFormerMode.ACCURATE
    do_formula_enrichment: bool = True
    do_code_enrichment: bool = True
    do_picture_description: bool = True
    picture_area_threshold: float = 0.02  # skip crops under 2% of page area
    images_scale: float = 2.0
    generate_page_images: bool = False

    # Hardware
    device: AcceleratorDevice = AcceleratorDevice.CUDA
    num_threads: int = 8
    use_flash_attention2: bool = False  # ignored if flash-attn isn't installed

    # Resume behaviour
    skip_existing: bool = True
    limit: int | None = None  # cap number of files this run, for a smoke test


# --------------------------------------------------------------------------- #
# Converter construction
# --------------------------------------------------------------------------- #


def build_converter(cfg: PipelineConfig) -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()

    pipeline_options.accelerator_options = AcceleratorOptions(
        device=cfg.device,
        num_threads=cfg.num_threads,
        cuda_use_flash_attention2=cfg.use_flash_attention2,
    )

    # Text layer + OCR safety net (bitmap regions only, not forced full-page)
    pipeline_options.do_ocr = True

    # Tables
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = TableStructureOptions(
        mode=cfg.table_mode,
        do_cell_matching=True,
    )

    # Formulas / code
    pipeline_options.do_formula_enrichment = cfg.do_formula_enrichment
    pipeline_options.do_code_enrichment = cfg.do_code_enrichment

    # Figures
    pipeline_options.do_picture_classification = True
    pipeline_options.do_picture_description = cfg.do_picture_description
    pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
        repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        prompt=(
            "Describe this figure in two to three sentences. Focus on what "
            "it shows (chart type, axes, trend, diagram components) rather "
            "than aesthetics."
        ),
        picture_area_threshold=cfg.picture_area_threshold,
    )
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_page_images = cfg.generate_page_images
    pipeline_options.images_scale = cfg.images_scale

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


# --------------------------------------------------------------------------- #
# Manifest / resume helpers
# --------------------------------------------------------------------------- #


def already_done(pdf_path: Path, output_dir: Path) -> bool:
    return (output_dir / f"{pdf_path.stem}.json").exists()


def append_manifest(
    manifest_path: Path,
    record: Mapping[str, object],
) -> None:
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Per-document export + sanity check
# --------------------------------------------------------------------------- #


def export_result(conv_res, output_dir: Path) -> dict[str, object]:
    stem = conv_res.input.file.stem
    record: dict[str, object] = {
        "source": str(conv_res.input.file),
        "status": str(conv_res.status),
        "timestamp": time.time(),
    }

    if conv_res.status not in (
        ConversionStatus.SUCCESS,
        ConversionStatus.PARTIAL_SUCCESS,
    ):
        record["errors"] = [e.error_message for e in conv_res.errors]
        return record

    doc = conv_res.document

    # Source of truth: full DoclingDocument, images kept as referenced files
    # (not embedded base64) so the JSON stays manageable across 3k+ papers.
    json_path = output_dir / f"{stem}.json"
    doc.save_as_json(json_path, image_mode=ImageRefMode.REFERENCED)

    # Human-readable / quick-look export. Not the source of truth — it drops
    # bounding boxes, provenance and table cell structure. Downstream
    # chunking should read the JSON / DoclingDocument, not this file.
    md_path = output_dir / f"{stem}.md"
    doc.save_as_markdown(md_path, image_mode=ImageRefMode.REFERENCED)

    record["output_json"] = str(json_path)
    record["num_pages"] = len(doc.pages)
    record["num_tables"] = len(doc.tables)
    record["num_pictures"] = len(doc.pictures)

    if conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
        record["errors"] = [e.error_message for e in conv_res.errors]

    # Cheap sanity check: flag documents that parsed "successfully" but
    # produced suspiciously little text — usually means the PDF is mostly
    # image content that neither the text layer nor OCR caught cleanly.
    # This won't catch everything, but it's a free signal to spot-check.
    text_len = len(doc.export_to_markdown())
    chars_per_page = text_len / max(record["num_pages"], 1)
    if chars_per_page < 200:
        record["warning"] = (
            f"low text density: {chars_per_page:.0f} chars/page — review manually"
        )
        log.warning("%s: %s", stem, record["warning"])

    return record


# --------------------------------------------------------------------------- #
# Batch runner
# --------------------------------------------------------------------------- #


def collect_pending(
    input_dir: Path, output_dir: Path, cfg: PipelineConfig
) -> list[Path]:
    pdfs = sorted(input_dir.glob("*.pdf"))
    if cfg.skip_existing:
        pending = [p for p in pdfs if not already_done(p, output_dir)]
        skipped = len(pdfs) - len(pending)
        if skipped:
            log.info("Skipping %d already-converted files", skipped)
    else:
        pending = pdfs
    if cfg.limit is not None:
        pending = pending[: cfg.limit]
    return pending


def run_batch(cfg: PipelineConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    pending = collect_pending(cfg.input_dir, cfg.output_dir, cfg)
    log.info("%d PDFs to convert", len(pending))
    if not pending:
        return

    converter = build_converter(cfg)

    n_ok = n_partial = n_fail = 0
    start = time.time()

    # convert_all keeps the pipeline (and its loaded models) alive across
    # the whole batch — building a new DocumentConverter per file would
    # reload TableFormer/SmolVLM/CodeFormula every single time.
    results = converter.convert_all(pending, raises_on_error=False)

    for i, conv_res in enumerate(results, start=1):
        stem = conv_res.input.file.stem
        try:
            record = export_result(conv_res, cfg.output_dir)
        except Exception:  # noqa: BLE001 — log and keep going, don't kill the batch
            log.error("Export failed for %s:\n%s", stem, traceback.format_exc())
            record = {
                "source": str(conv_res.input.file),
                "status": "EXPORT_FAILURE",
                "timestamp": time.time(),
            }

        append_manifest(cfg.manifest_path, record)

        status = record["status"]
        if status == str(ConversionStatus.SUCCESS):
            n_ok += 1
        elif status == str(ConversionStatus.PARTIAL_SUCCESS):
            n_partial += 1
            log.warning("%s converted with issues: %s", stem, record.get("errors"))
        else:
            n_fail += 1
            log.error("%s failed: %s", stem, record.get("errors", status))

        if i % 25 == 0 or i == len(pending):
            elapsed = time.time() - start
            log.info(
                "[%d/%d] ok=%d partial=%d fail=%d  (%.1fs/doc avg)",
                i,
                len(pending),
                n_ok,
                n_partial,
                n_fail,
                elapsed / i,
            )

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
    parser.add_argument("--input-dir", type=Path, default=Path("data/pdfs"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/docling_json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test on N files")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (debugging only)")
    args = parser.parse_args()

    cfg = PipelineConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        skip_existing=not args.no_skip_existing,
        limit=args.limit,
        device=AcceleratorDevice.CPU if args.cpu else AcceleratorDevice.CUDA,
    )
    run_batch(cfg)


if __name__ == "__main__":
    main()
