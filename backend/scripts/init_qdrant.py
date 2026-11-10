"""
Create the Qdrant collections RAGwise needs, skipping any that already exist.

    ragwise_arxiv  main chunk collection. Mirrors embed_to_qdrant.py exactly:
                   "dense" (1024, cosine) + "sparse" named vectors, payload
                   indexes on paper_id (keyword) and has_picture (bool).
    ragwise_cache  semantic cache. "dense" (1024, cosine), kept in RAM, payload
                   indexes on cache_version (keyword), top_k (integer),
                   expires_at (integer, unix seconds).

An existing collection is never modified or recreated.

Usage:
    python scripts/init_qdrant.py
    python scripts/init_qdrant.py --qdrant-url http://localhost:6333 --no-grpc
    python scripts/init_qdrant.py --purge-expired     # delete expired cache entries

Connection settings come from the flags, falling back to the QDRANT_URL and
QDRANT_API_KEY environment variables (same as embed_to_qdrant.py). This script
does not read .env, so export them or pass the flags.
"""

import argparse
import os
import sys
import time

from qdrant_client import QdrantClient, models

DENSE_DIM = 1024  # BGE-M3 dense output size.
DEFAULT_MAIN_COLLECTION = "ragwise_arxiv"
DEFAULT_CACHE_COLLECTION = "ragwise_cache"


def ensure_main_collection(
    client: QdrantClient, name: str, use_sparse: bool, on_disk: bool
) -> bool:
    """Create the chunk collection. Returns True if created, False if it existed."""
    if client.collection_exists(name):
        print(f"[skip]   '{name}' already exists.")
        return False

    print(
        f"[create] '{name}' (dense={DENSE_DIM}d, sparse={use_sparse}, on_disk={on_disk})"
    )
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(
                size=DENSE_DIM, distance=models.Distance.COSINE, on_disk=on_disk
            ),
        },
        sparse_vectors_config=(
            {
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=on_disk)
                ),
            }
            if use_sparse
            else None
        ),
    )
    client.create_payload_index(
        collection_name=name,
        field_name="paper_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=name,
        field_name="has_picture",
        field_schema=models.PayloadSchemaType.BOOL,
    )
    return True


def ensure_cache_collection(client: QdrantClient, name: str) -> bool:
    """Create the semantic-cache collection. Returns True if created."""
    if client.collection_exists(name):
        print(f"[skip]   '{name}' already exists.")
        return False

    print(f"[create] '{name}' (dense={DENSE_DIM}d, in RAM)")
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(
                size=DENSE_DIM, distance=models.Distance.COSINE, on_disk=False
            ),
        },
    )
    client.create_payload_index(
        collection_name=name,
        field_name="cache_version",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=name,
        field_name="top_k",
        field_schema=models.PayloadSchemaType.INTEGER,
    )
    client.create_payload_index(
        collection_name=name,
        field_name="expires_at",
        field_schema=models.PayloadSchemaType.INTEGER,
    )
    return True


def purge_expired(client: QdrantClient, name: str) -> None:
    if not client.collection_exists(name):
        print(f"[skip]   '{name}' does not exist, nothing to purge.")
        return
    now = int(time.time())
    expired = models.Filter(
        must=[
            models.FieldCondition(key="expires_at", range=models.Range(lte=now)),
        ]
    )
    n = client.count(collection_name=name, count_filter=expired, exact=True).count
    if n:
        client.delete(
            collection_name=name,
            points_selector=models.FilterSelector(filter=expired),
            wait=True,
        )
    print(f"[purge]  deleted {n} expired entries from '{name}'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the RAGwise Qdrant collections (skips existing ones)."
    )
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
        help="Use REST instead of gRPC (gRPC is the default, like the app).",
    )
    parser.add_argument("--qdrant-grpc-port", type=int, default=6334)
    parser.add_argument("--collection", type=str, default=DEFAULT_MAIN_COLLECTION)
    parser.add_argument(
        "--cache-collection", type=str, default=DEFAULT_CACHE_COLLECTION
    )
    parser.add_argument(
        "--no-sparse",
        action="store_true",
        help="Main collection: dense-only (must match how you embed).",
    )
    parser.add_argument(
        "--vectors-in-ram",
        action="store_true",
        help="Main collection: keep vectors in RAM instead of on disk.",
    )
    parser.add_argument(
        "--skip-main", action="store_true", help="Don't touch the main collection."
    )
    parser.add_argument(
        "--skip-cache", action="store_true", help="Don't touch the cache collection."
    )
    parser.add_argument(
        "--purge-expired",
        action="store_true",
        help="Delete expired entries from the cache collection, then exit.",
    )
    args = parser.parse_args()

    prefer_grpc = not args.no_grpc
    client = QdrantClient(
        url=args.qdrant_url,
        api_key=args.qdrant_api_key,
        prefer_grpc=prefer_grpc,
        grpc_port=args.qdrant_grpc_port,
    )
    try:
        client.get_collections()  # confirm the connection actually works
    except Exception as e:
        print(
            f"Error: couldn't reach Qdrant ({'gRPC' if prefer_grpc else 'REST'}): {e}"
        )
        if prefer_grpc:
            print("Try --no-grpc to use REST instead.")
        sys.exit(1)

    try:
        if args.purge_expired:
            purge_expired(client, args.cache_collection)
            return
        if not args.skip_main:
            ensure_main_collection(
                client,
                args.collection,
                use_sparse=not args.no_sparse,
                on_disk=not args.vectors_in_ram,
            )
        if not args.skip_cache:
            ensure_cache_collection(client, args.cache_collection)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
