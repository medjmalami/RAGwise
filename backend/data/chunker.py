"""
Stage 2: Chunk parsed DoclingDocuments (produced by the Stage 1 HTML parser)
using Docling's HybridChunker, and write one JSONL file of chunks per paper —
mirroring Stage 1's per-paper output layout.

Expects the Stage 1 input layout:
    <input>/<paper_id>/<paper_id>.docling.json
    <input>/<paper_id>/<paper_id>.md            (not read, just documents the layout)
    <input>/<paper_id>/artifacts/...            (referenced images — resolved and
                                                  linked into chunks that use them)

Produces:
    <output>/<paper_id>/<paper_id>.chunks.jsonl

Usage:
    python chunk_docling.py --input parsed/ --output chunks/ \
        --tokenizer BAAI/bge-m3 --max-tokens 512
"""

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path


def build_chunker(tokenizer_name: str, max_tokens: int, merge_peers: bool):
    """Build a HybridChunker bound to the embedding model's tokenizer, so
    chunk sizes are measured in the same tokens that will hit BGE-M3 at
    embedding time (rather than the chunker's default tokenizer).

    Handles both the newer docling_core API (explicit BaseTokenizer wrapper
    object) and the older API (tokenizer passed as a plain HF model-name
    string), since the exact signature has moved between docling_core
    versions.
    """
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

    tokenizer_obj = None
    try:
        # Newer API: docling_core.transforms.chunker.tokenizer.huggingface.HuggingFaceTokenizer
        from docling_core.transforms.chunker.tokenizer.huggingface import (
            HuggingFaceTokenizer,
        )
        from transformers import AutoTokenizer

        hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer_obj = HuggingFaceTokenizer(
            tokenizer=hf_tokenizer, max_tokens=max_tokens
        )
    except ImportError:
        # Older docling_core: HybridChunker accepts a plain HF model name
        # and max_tokens directly.
        tokenizer_obj = None

    chunker_kwargs = {}
    sig_params = inspect.signature(HybridChunker).parameters

    if tokenizer_obj is not None and "tokenizer" in sig_params:
        chunker_kwargs["tokenizer"] = tokenizer_obj
    elif "tokenizer" in sig_params:
        chunker_kwargs["tokenizer"] = tokenizer_name
        if "max_tokens" in sig_params:
            chunker_kwargs["max_tokens"] = max_tokens

    if "merge_peers" in sig_params:
        chunker_kwargs["merge_peers"] = merge_peers

    return HybridChunker(**chunker_kwargs)


def get_chunk_text(chunker, chunk):
    """Prefer the chunker's contextualized serialization (prepends section
    headings/captions to the chunk body — this is what should actually be
    embedded), falling back to the raw chunk text on older docling_core
    versions that lack `contextualize`.
    """
    if hasattr(chunker, "contextualize"):
        try:
            return chunker.contextualize(chunk=chunk)
        except Exception:
            pass
    return chunk.text


def count_tokens(chunker, text: str):
    tokenizer = getattr(chunker, "tokenizer", None)
    if tokenizer is None:
        return None
    try:
        # Newer BaseTokenizer wrapper
        if hasattr(tokenizer, "count_tokens"):
            return tokenizer.count_tokens(text)
        # Raw HF tokenizer
        return len(tokenizer.encode(text))
    except Exception:
        return None


def extract_headings(chunk):
    meta = getattr(chunk, "meta", None)
    if meta is None:
        return []
    headings = getattr(meta, "headings", None)
    return list(headings) if headings else []


def build_picture_uri_map(json_path: Path):
    """Read the paper's raw .docling.json directly and map each picture's
    self_ref to its stored image URI. Reading the raw JSON (rather than the
    loaded DoclingDocument's .image attribute) sidesteps a validation quirk
    where the `image` field wasn't reliably surviving `load_from_json()` on
    this docling_core version, even though it's clearly present in the file.
    """
    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    uri_by_ref = {}
    for pic in raw.get("pictures", []) or []:
        self_ref = pic.get("self_ref")
        uri = (pic.get("image") or {}).get("uri")
        if self_ref and uri:
            uri_by_ref[self_ref] = uri
    return uri_by_ref


def resolve_image_path(uri_str, paper_dir: Path, input_dir: Path):
    """Resolve a stored image URI to a real file, relative to --input.

    The URI Stage 1 wrote can be an absolute path from whatever machine/run
    produced it (fragile — breaks if the dataset moves). So beyond trying it
    as-is, this also falls back to just the filename joined against this
    paper's own artifacts/ directory, which is where Stage 1 actually put
    the image and is stable regardless of what path got baked into the JSON.
    """
    if not uri_str:
        return None
    uri_str = str(uri_str)
    if uri_str.startswith("file://"):
        uri_str = uri_str[len("file://") :]

    filename = Path(uri_str).name
    candidates = [
        Path(uri_str) if Path(uri_str).is_absolute() else (paper_dir / uri_str),
        paper_dir / "artifacts" / filename,
    ]

    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except Exception:
            continue
        if candidate.exists():
            try:
                return str(candidate.relative_to(input_dir))
            except ValueError:
                return str(candidate)
    return None


def extract_doc_items(
    chunk, paper_dir: Path, input_dir: Path, picture_uri_by_ref: dict
):
    """Lightweight provenance: item type, page numbers, a stable ref back
    into the source DoclingDocument, and (for pictures) the resolved image
    file path — enough to trace a retrieved chunk back to its exact node,
    and to load any figure it references, without re-parsing the paper's
    .docling.json at retrieval time.
    """
    meta = getattr(chunk, "meta", None)
    items = getattr(meta, "doc_items", None) if meta is not None else None
    if not items:
        return []

    out = []
    for item in items:
        entry = {"type": getattr(item, "label", None) or type(item).__name__}

        self_ref = getattr(item, "self_ref", None)
        if self_ref is None and hasattr(item, "get_ref"):
            try:
                ref_obj = item.get_ref()
                self_ref = getattr(ref_obj, "cref", None) or str(ref_obj)
            except Exception:
                self_ref = None
        if self_ref:
            entry["self_ref"] = self_ref

        prov = getattr(item, "prov", None)
        if prov:
            pages = sorted(
                {p.page_no for p in prov if getattr(p, "page_no", None) is not None}
            )
            if pages:
                entry["pages"] = pages

        if str(entry["type"]).lower() in ("picture", "figure"):
            uri_str = picture_uri_by_ref.get(self_ref) if self_ref else None
            image_path = resolve_image_path(uri_str, paper_dir, input_dir)
            if image_path:
                entry["image_path"] = image_path

        out.append(entry)
    return out


def collect_image_paths(doc_items):
    """Dedup list of resolved image paths across a chunk's doc_items, for a
    single top-level `image_paths` field that's ready to load and pass to a
    vision LLM alongside the chunk's text — no need to filter doc_items."""
    paths = []
    for entry in doc_items:
        p = entry.get("image_path")
        if p and p not in paths:
            paths.append(p)
    return paths


def has_picture(doc_items):
    """True if any doc item backing this chunk is a picture/figure — lets
    you filter/retrieve chunks that include a Gemma-described figure."""
    picture_labels = {"picture", "figure"}
    return any(
        str(entry.get("type", "")).lower() in picture_labels for entry in doc_items
    )


def main():
    parser = argparse.ArgumentParser(
        description="Chunk Stage 1 DoclingDocument JSON files with HybridChunker "
        "and write one JSONL file of chunks per paper."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Stage 1 output directory (contains <paper_id>/<paper_id>.docling.json).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory. Chunks for each paper are written to "
        "<output>/<paper_id>/<paper_id>.chunks.jsonl (mirrors the Stage 1 "
        "--output layout).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="BAAI/bge-m3",
        help="HF tokenizer name used to size chunks (should match the embedding model).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens per chunk, measured with --tokenizer.",
    )
    parser.add_argument(
        "--no-merge-peers",
        action="store_true",
        help="Disable HybridChunker's merging of small adjacent chunks that share headings.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of papers to process.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess and overwrite papers that already have a chunks file.",
    )
    parser.add_argument(
        "--min-chunk-chars",
        type=int,
        default=0,
        help="Drop chunks whose contextualized text is shorter than this many "
        "characters (filters out near-empty fragments like bare headings).",
    )

    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.is_dir():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*/*.docling.json"))
    if args.limit is not None:
        json_files = json_files[: args.limit]

    print(f"Found {len(json_files)} parsed papers to chunk.")

    try:
        from docling_core.types.doc.document import DoclingDocument
    except ImportError as e:
        print(f"Error importing docling_core: {e}")
        sys.exit(1)

    print(f"Loading tokenizer '{args.tokenizer}' and building HybridChunker...")
    try:
        chunker = build_chunker(
            tokenizer_name=args.tokenizer,
            max_tokens=args.max_tokens,
            merge_peers=not args.no_merge_peers,
        )
    except Exception as e:
        print(f"Error building HybridChunker: {e}")
        sys.exit(1)

    failures = []
    skipped = 0
    total_chunks = 0
    papers_written = 0

    for json_path in json_files:
        paper_id = json_path.stem.removesuffix(".docling")
        if not paper_id or paper_id == json_path.stem:
            # Fallback for Python <3.9 or unexpected stem shape.
            paper_id = json_path.parent.name

        paper_dir = json_path.parent
        paper_out_dir = output_dir / paper_id
        out_path = paper_out_dir / f"{paper_id}.chunks.jsonl"

        # --- RESUME CAPABILITY (mirrors Stage 1's per-paper skip) ---
        if out_path.exists() and not args.force:
            skipped += 1
            continue

        print(f"\nChunking {paper_id}...")

        try:
            doc = DoclingDocument.load_from_json(json_path)
        except Exception as e:
            print(f"  Failed to load {json_path.name}: {e}")
            failures.append(paper_id)
            continue

        picture_uri_by_ref = build_picture_uri_map(json_path)

        try:
            chunk_iter = chunker.chunk(dl_doc=doc)
            records = []
            for chunk in chunk_iter:
                text = get_chunk_text(chunker, chunk)
                if not text or len(text.strip()) < args.min_chunk_chars:
                    continue
                doc_items = extract_doc_items(
                    chunk, paper_dir, input_dir, picture_uri_by_ref
                )
                records.append(
                    {
                        # chunk_id/chunk_index/num_chunks are filled in below,
                        # once the final kept-chunk count is known.
                        "paper_id": paper_id,
                        "text": text,
                        "raw_text": chunk.text,
                        "content_hash": hashlib.sha256(
                            chunk.text.encode("utf-8")
                        ).hexdigest(),
                        "token_count": count_tokens(chunker, text),
                        "headings": extract_headings(chunk),
                        "doc_items": doc_items,
                        "has_picture": has_picture(doc_items),
                        "image_paths": collect_image_paths(doc_items),
                        "source_json": str(json_path.relative_to(input_dir)),
                    }
                )

            num_chunks = len(records)
            for i, record in enumerate(records):
                record["chunk_id"] = f"{paper_id}::chunk_{i:04d}"
                record["chunk_index"] = i
                record["num_chunks"] = num_chunks
        except Exception as e:
            print(f"  Failed to chunk {paper_id}: {e}")
            import traceback

            traceback.print_exc()
            failures.append(paper_id)
            continue

        try:
            paper_out_dir.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as out_f:
                for record in records:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"  Failed to write chunks for {paper_id}: {e}")
            failures.append(paper_id)
            continue

        total_chunks += len(records)
        papers_written += 1
        print(f"  Wrote {len(records)} chunks -> {out_path}")

    if skipped:
        print(f"\nSkipped {skipped} already-chunked papers.")
    print(
        f"Done. {papers_written}/{len(json_files) - skipped} newly-processed "
        f"papers succeeded, {total_chunks} chunks written under {output_dir}."
    )
    if failures:
        print(f"Failed ({len(failures)}): {failures}")


if __name__ == "__main__":
    main()
