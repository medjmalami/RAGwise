"""
Stage 3: Embed chunked papers (produced by Stage 2's chunk_docling.py) with
BAAI/bge-m3 and upsert them into a Qdrant collection.

Uses BGE-M3's dense + sparse (lexical) outputs and stores them as two named
vectors on the same point ("dense" and "sparse"), so retrieval can do hybrid
search later (dense for semantic recall, sparse for exact term/acronym
matches that dense embeddings tend to blur). Pass --no-sparse to store
dense-only instead.

Expects the Stage 2 output layout:
    <input>/<paper_id>/<paper_id>.chunks.jsonl

Writes a per-paper marker file under --state-dir once a paper's chunks are
embedded and upserted, so re-running the script resumes instead of
re-embedding everything already done (mirrors Stage 1/2's resume behaviour).
Point IDs are deterministic (uuid5 of chunk_id), so re-running with --force
overwrites the same points instead of duplicating them.

Usage:
    python embed_to_qdrant.py --input chunks/ --collection ragwise_arxiv
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

# Fixed namespace so uuid5(CHUNK_ID_NAMESPACE, chunk_id) is stable across runs/machines.
CHUNK_ID_NAMESPACE = uuid.UUID("d6e1b3c2-8f2a-4b7a-9c1d-2f6a7e9b0c3d")

DENSE_DIM = 1024  # BGE-M3's dense output size.


def resolve_device(requested, require_cuda):
    """Resolve the device and print a clear, unambiguous confirmation of what
    will actually be used — never silently fall back without saying so."""
    if requested:
        device = requested
    else:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "torch reports CUDA is not available on this machine."
                )
            name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"CUDA confirmed — using {name} ({vram_gb:.1f} GB VRAM).")
        except Exception as e:
            print(f"Warning: device resolved to 'cuda' but torch can't confirm it: {e}")
            if require_cuda:
                print("Aborting: --require-cuda was set.")
                sys.exit(1)
            print("Falling back to CPU.")
            device = "cpu"
    else:
        if require_cuda:
            print(
                "Error: --require-cuda was set but no usable CUDA device was found. "
                "Check `nvidia-smi` and that torch was installed with CUDA support."
            )
            sys.exit(1)
        print(
            "Warning: running on CPU — this will be dramatically slower for BGE-M3 at this corpus size."
        )

    return device


def build_embedder(model_name, device, use_fp16):
    from FlagEmbedding import BGEM3FlagModel

    print(f"Loading {model_name} on {device} (fp16={use_fp16})...")
    return BGEM3FlagModel(model_name, use_fp16=use_fp16, device=device)


def embed_texts(embedder, texts, batch_size, max_length, use_sparse):
    """Run BGE-M3 over `texts`. FlagEmbedding batches internally according to
    `batch_size`, so it's safe to pass a whole paper's chunks at once even on
    a small GPU — only one batch of activations is held on the GPU at a time.

    Returns (dense_vecs, sparse_weights_or_None):
      dense_vecs: list of 1024-dim float lists, one per text.
      sparse_weights: list of {token_id_str: weight} dicts, or None.
    """
    output = embedder.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=use_sparse,
        return_colbert_vecs=False,
    )
    dense_vecs = [
        v.tolist() if hasattr(v, "tolist") else list(v) for v in output["dense_vecs"]
    ]
    sparse_weights = output.get("lexical_weights") if use_sparse else None
    return dense_vecs, sparse_weights


def to_sparse_vector(lexical_weights):
    """Convert BGE-M3's {token_id_str: weight} dict into a Qdrant SparseVector,
    with indices sorted ascending (some Qdrant versions expect this)."""
    from qdrant_client import models

    if not lexical_weights:
        return models.SparseVector(indices=[], values=[])
    pairs = sorted(
        (
            (int(token_id), float(weight))
            for token_id, weight in lexical_weights.items()
        ),
        key=lambda p: p[0],
    )
    return models.SparseVector(
        indices=[p[0] for p in pairs], values=[p[1] for p in pairs]
    )


def collection_exists(client, collection_name):
    try:
        return client.collection_exists(collection_name)
    except AttributeError:
        # Older qdrant-client without collection_exists().
        existing = {c.name for c in client.get_collections().collections}
        return collection_name in existing


def ensure_collection(client, collection_name, use_sparse, recreate, on_disk):
    from qdrant_client import models

    exists = collection_exists(client, collection_name)

    if exists and recreate:
        print(
            f"Deleting existing collection '{collection_name}' (--recreate-collection)..."
        )
        client.delete_collection(collection_name)
        exists = False

    if exists:
        print(f"Collection '{collection_name}' already exists, reusing it.")
        return

    print(
        f"Creating collection '{collection_name}' (dense={DENSE_DIM}d, sparse={use_sparse})..."
    )
    vectors_config = {
        "dense": models.VectorParams(
            size=DENSE_DIM, distance=models.Distance.COSINE, on_disk=on_disk
        ),
    }
    sparse_vectors_config = None
    if use_sparse:
        sparse_vectors_config = {
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=on_disk)
            ),
        }

    client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )

    # Payload indexes for the filters you'll actually use at retrieval time.
    client.create_payload_index(
        collection_name=collection_name,
        field_name="paper_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="has_picture",
        field_schema=models.PayloadSchemaType.BOOL,
    )


def load_records(jsonl_path):
    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def chunks_of(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main():
    parser = argparse.ArgumentParser(
        description="Embed Stage 2 chunks with BGE-M3 and upsert them into Qdrant."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Stage 2 output directory (contains <paper_id>/<paper_id>.chunks.jsonl).",
    )
    parser.add_argument(
        "--state-dir",
        type=str,
        default="embed_state",
        help="Directory for per-paper '<paper_id>.done' marker files (resume support).",
    )
    parser.add_argument("--collection", type=str, default="ragwise_arxiv")
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=os.environ.get("QDRANT_URL", "http://localhost:6333"),
    )
    parser.add_argument(
        "--qdrant-api-key", type=str, default=os.environ.get("QDRANT_API_KEY")
    )
    parser.add_argument(
        "--no-grpc",
        action="store_true",
        help="Use Qdrant's REST API instead of gRPC. gRPC is used by default (faster bulk upserts).",
    )
    parser.add_argument(
        "--qdrant-grpc-port",
        type=int,
        default=6334,
        help="gRPC port — matches the docker-compose mapping.",
    )
    parser.add_argument("--model", type=str, default="BAAI/bge-m3")
    parser.add_argument(
        "--device", type=str, default=None, help="cuda / cpu. Auto-detected if omitted."
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Abort instead of silently falling back to CPU if CUDA isn't available.",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable fp16 (fp16 is used by default on cuda; ignored on cpu).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Embedding forward batch size. Keep small on a 4GB-VRAM GPU.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=640,
        help="Should match (or exceed) Stage 2's --max-tokens.",
    )
    parser.add_argument(
        "--no-sparse",
        action="store_true",
        help="Skip BGE-M3's sparse (lexical) vectors and store dense-only.",
    )
    parser.add_argument(
        "--upsert-batch-size",
        type=int,
        default=64,
        help="Points per Qdrant upsert call. Well within max_request_size_mb=64 at this size.",
    )
    parser.add_argument(
        "--vectors-in-ram",
        action="store_true",
        help="Store vectors in RAM instead of on-disk (faster, uses more memory).",
    )
    parser.add_argument("--recreate-collection", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed papers that already have a marker file.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=0,
        help="Sleep this many seconds after every --cooldown-every paper(s), so the GPU gets "
        "real idle time between bursts instead of running continuously. 0 disables it.",
    )
    parser.add_argument(
        "--cooldown-every",
        type=int,
        default=1,
        help="How many papers to process between cooldown sleeps (default: every paper).",
    )

    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    state_dir = Path(args.state_dir).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        print(f"Error: input directory '{input_dir}' does not exist.")
        sys.exit(1)

    jsonl_files = sorted(input_dir.glob("*/*.chunks.jsonl"))
    if args.limit is not None:
        jsonl_files = jsonl_files[: args.limit]
    print(f"Found {len(jsonl_files)} chunked papers.")

    device = resolve_device(args.device, args.require_cuda)
    use_fp16 = (not args.no_fp16) and device == "cuda"
    use_sparse = not args.no_sparse
    on_disk = not args.vectors_in_ram

    try:
        embedder = build_embedder(args.model, device, use_fp16)
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        sys.exit(1)

    from qdrant_client import QdrantClient, models

    prefer_grpc = not args.no_grpc
    print(
        f"Connecting to Qdrant at {args.qdrant_url} via "
        f"{'gRPC (port ' + str(args.qdrant_grpc_port) + ')' if prefer_grpc else 'REST'}..."
    )
    client = QdrantClient(
        url=args.qdrant_url,
        api_key=args.qdrant_api_key,
        prefer_grpc=prefer_grpc,
        grpc_port=args.qdrant_grpc_port,
    )
    try:
        client.get_collections()  # cheap round-trip that confirms the connection actually works
    except Exception as e:
        print(
            f"Error: couldn't reach Qdrant ({'gRPC' if prefer_grpc else 'REST'}): {e}"
        )
        if prefer_grpc:
            print(
                "Try --no-grpc to fall back to REST and see if that connects instead."
            )
        sys.exit(1)
    try:
        ensure_collection(
            client,
            args.collection,
            use_sparse=use_sparse,
            recreate=args.recreate_collection,
            on_disk=on_disk,
        )
    except Exception as e:
        print(f"Error preparing collection: {e}")
        sys.exit(1)

    failures = []
    skipped = 0
    total_points = 0
    papers_done = 0

    for file_index, jsonl_path in enumerate(jsonl_files):
        paper_id = jsonl_path.stem.removesuffix(".chunks")
        if not paper_id or paper_id == jsonl_path.stem:
            paper_id = jsonl_path.parent.name

        marker_path = state_dir / f"{paper_id}.done"
        if marker_path.exists() and not args.force:
            skipped += 1
            continue

        print(f"\nEmbedding {paper_id}...")
        try:
            records = load_records(jsonl_path)
        except Exception as e:
            print(f"  Failed to read {jsonl_path.name}: {e}")
            failures.append(paper_id)
            continue

        if not records:
            marker_path.write_text("empty\n")
            continue

        try:
            texts = [r["text"] for r in records]
            dense_vecs, sparse_weights = embed_texts(
                embedder, texts, args.batch_size, args.max_length, use_sparse
            )
            points = []
            for i, record in enumerate(records):
                vector = {"dense": dense_vecs[i]}
                if use_sparse:
                    vector["sparse"] = to_sparse_vector(sparse_weights[i])
                point_id = str(uuid.uuid5(CHUNK_ID_NAMESPACE, record["chunk_id"]))
                points.append(
                    models.PointStruct(id=point_id, vector=vector, payload=record)
                )
        except Exception as e:
            print(f"  Failed to embed {paper_id}: {e}")
            import traceback

            traceback.print_exc()
            failures.append(paper_id)
            continue

        try:
            for batch in chunks_of(points, args.upsert_batch_size):
                client.upsert(collection_name=args.collection, points=batch, wait=True)
        except Exception as e:
            print(f"  Failed to upsert {paper_id}: {e}")
            failures.append(paper_id)
            continue

        marker_path.write_text(f"{len(points)} points\n")
        total_points += len(points)
        papers_done += 1
        print(f"  Upserted {len(points)} points.")

        is_last_file = file_index == len(jsonl_files) - 1
        if (
            args.cooldown_seconds > 0
            and not is_last_file
            and papers_done % args.cooldown_every == 0
        ):
            print(f"  Cooling down for {args.cooldown_seconds:.0f}s...")
            time.sleep(args.cooldown_seconds)

    if skipped:
        print(f"\nSkipped {skipped} already-embedded papers.")
    print(
        f"Done. {papers_done}/{len(jsonl_files) - skipped} newly-processed papers "
        f"succeeded, {total_points} points upserted into '{args.collection}'."
    )
    if failures:
        print(f"Failed ({len(failures)}): {failures}")


if __name__ == "__main__":
    main()
