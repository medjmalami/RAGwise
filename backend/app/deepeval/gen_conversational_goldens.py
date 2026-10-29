"""
Step 1 - Generate multi-turn *scenarios* (DeepEval ConversationalGoldens) from
chunks stored in Qdrant, where each scenario needs information from SEVERAL papers.

How it works
------------
1. Scan the collection (payload only, no vectors/text) and keep chunks that look
   like usable prose: token_count in range, heading not References/Acknowledgements/...
2. Pick one random anchor chunk per paper (max paper coverage).
3. For each anchor, ask Qdrant for its nearest chunks in OTHER papers (by the stored
   dense vector - nothing is re-embedded) and keep the closest few from distinct papers.
   anchor + neighbours = one "context" (a list of chunk texts from different papers).
4. Hand the contexts to DeepEval's Synthesizer.generate_conversational_goldens_from_contexts,
   telling it which paper each chunk came from, so scenarios must combine papers.
5. Keep only goldens whose scenario actually used >= --min-papers-used papers.
6. Write CSV (+ contexts.jsonl for inspection).

Nothing is written to Langfuse or to your database. Output is a CSV.

Example
-------
    # inspect the contexts first (no LLM calls at all):
    python gen_conversational_goldens.py --dry-run --n-goldens 20

    # smoke test the LLM path on 3 goldens:
    python gen_conversational_goldens.py --n-goldens 3

    # the real run:
    python gen_conversational_goldens.py --n-goldens 100
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient, models

# --------------------------------------------------------------------------- #
# Payload fields written by your Stage 2 chunker (chunk_docling.py)
# --------------------------------------------------------------------------- #
LIGHT_FIELDS = ["paper_id", "chunk_id", "token_count", "headings", "has_picture"]
TEXT_FIELDS = LIGHT_FIELDS + ["text", "content_hash"]

# Chunks whose heading path matches this are skipped. Adjust to taste.
DEFAULT_JUNK_HEADINGS = (
    r"references|bibliograph|acknowledg|author contribution|"
    r"reproducibility|checklist|ethics statement|appendix"
)

# --------------------------------------------------------------------------- #
# Prompt hints for the scenario generator (edit freely)
# --------------------------------------------------------------------------- #
SCENARIO_CONTEXT = (
    "A user is chatting with a research assistant that answers questions over a "
    "corpus of arXiv papers (retrieval-augmented, with access to the conversation so far)."
)
CONVERSATIONAL_TASK = (
    "The user opens with a concrete question, then asks follow-up questions that "
    "depend on the earlier answers (pronouns like 'it' or 'that method', 'what about X?', "
    "comparisons, asking to go deeper). The overall goal requires combining information "
    "from several different papers."
)
PARTICIPANT_ROLES = "A researcher or student (user) and an AI research assistant."
SCENARIO_FORMAT = (
    "2-4 sentences, third person: who the user is, their opening question, how they "
    "follow up, and what they want to understand by the end."
)
EXPECTED_OUTCOME_FORMAT = (
    "1-3 sentences describing what a correct, grounded conversation should have "
    "established, mentioning which papers' content it should draw on."
)


# --------------------------------------------------------------------------- #
# Qdrant side: scanning, anchors, neighbours
# --------------------------------------------------------------------------- #
def chunk_is_eligible(payload: Dict[str, Any], args, junk_re) -> bool:
    tc = payload.get("token_count")
    if tc is not None and not (args.min_tokens <= tc <= args.max_tokens):
        return False
    if args.no_pictures and payload.get("has_picture"):
        return False
    headings = " / ".join(payload.get("headings") or [])
    if junk_re is not None and junk_re.search(headings):
        return False
    return True


def text_looks_like_prose(text: str, min_alpha_ratio: float) -> bool:
    """Cheap filter for tables / equations / number soup."""
    if not text:
        return False
    alpha = sum(ch.isalpha() for ch in text)
    return (alpha / max(len(text), 1)) >= min_alpha_ratio


def scan_candidates(client: QdrantClient, args, junk_re) -> Dict[str, List[str]]:
    """paper_id -> [point ids of eligible chunks]. Payload-light scroll."""
    by_paper: Dict[str, List[str]] = defaultdict(list)
    offset = None
    seen = 0
    while True:
        points, offset = client.scroll(
            collection_name=args.collection,
            limit=args.scroll_batch,
            offset=offset,
            with_payload=LIGHT_FIELDS,
            with_vectors=False,
        )
        for p in points:
            seen += 1
            payload = p.payload or {}
            pid = payload.get("paper_id")
            if pid and chunk_is_eligible(payload, args, junk_re):
                by_paper[pid].append(str(p.id))
        if offset is None:
            break
    n_elig = sum(len(v) for v in by_paper.values())
    print(f"Scanned {seen} chunks; {n_elig} eligible across {len(by_paper)} papers.")
    return by_paper


def fetch_point(client: QdrantClient, collection: str, point_id: str):
    res = client.retrieve(
        collection_name=collection,
        ids=[point_id],
        with_payload=TEXT_FIELDS,
        with_vectors=False,
    )
    return res[0] if res else None


def find_neighbours(client: QdrantClient, anchor, args, junk_re, anchor_hash):
    """Closest eligible chunks from OTHER papers, one per paper."""
    anchor_paper = anchor.payload["paper_id"]
    resp = client.query_points(
        collection_name=args.collection,
        query=str(anchor.id),  # query by stored vector of this point id
        using="dense",
        limit=args.neighbour_pool,
        score_threshold=args.min_score,
        query_filter=models.Filter(
            must_not=[
                models.FieldCondition(
                    key="paper_id", match=models.MatchValue(value=anchor_paper)
                )
            ]
        ),
        with_payload=TEXT_FIELDS,
        with_vectors=False,
    )
    chosen, papers = [], {anchor_paper}
    for hit in resp.points:
        pl = hit.payload or {}
        pid = pl.get("paper_id")
        if not pid or pid in papers:
            continue
        if hit.score > args.max_score:  # near-duplicate boilerplate
            continue
        if pl.get("content_hash") and pl.get("content_hash") == anchor_hash:
            continue
        if not chunk_is_eligible(pl, args, junk_re):
            continue
        if not text_looks_like_prose(pl.get("text", ""), args.min_alpha_ratio):
            continue
        chosen.append(hit)
        papers.add(pid)
        if len(chosen) >= args.papers_per_context - 1:
            break
    return chosen


def build_contexts(client: QdrantClient, args) -> List[Dict[str, Any]]:
    junk_re = re.compile(args.junk_headings, re.I) if args.junk_headings else None
    rng = random.Random(args.seed)

    by_paper = scan_candidates(client, args, junk_re)
    papers = sorted(by_paper)
    rng.shuffle(papers)

    target = int(round(args.n_goldens * args.oversample / args.scenarios_per_context))
    contexts: List[Dict[str, Any]] = []
    skipped = Counter()

    for paper in papers:
        if len(contexts) >= target:
            break
        anchor_id = rng.choice(by_paper[paper])
        anchor = fetch_point(client, args.collection, anchor_id)
        if anchor is None:
            skipped["anchor_missing"] += 1
            continue
        a_pl = anchor.payload or {}
        if not text_looks_like_prose(a_pl.get("text", ""), args.min_alpha_ratio):
            skipped["anchor_not_prose"] += 1
            continue

        neighbours = find_neighbours(
            client, anchor, args, junk_re, a_pl.get("content_hash")
        )
        n_papers = 1 + len(neighbours)
        if n_papers < args.min_context_papers:
            skipped["too_few_papers_in_context"] += 1
            continue

        members = [(str(anchor.id), a_pl, None)] + [
            (str(h.id), h.payload, h.score) for h in neighbours
        ]
        contexts.append(
            {
                "context_idx": len(contexts),
                "point_ids": [m[0] for m in members],
                "chunk_ids": [m[1].get("chunk_id") for m in members],
                "paper_ids": [m[1]["paper_id"] for m in members],
                "scores": [m[2] for m in members],  # None for the anchor
                "texts": [m[1]["text"] for m in members],
            }
        )

    print(f"Built {len(contexts)} contexts (target {target}). Skipped: {dict(skipped)}")
    sizes = Counter(len(set(c["paper_ids"])) for c in contexts)
    print(f"Papers per context: {dict(sorted(sizes.items()))}")
    scores = [s for c in contexts for s in c["scores"] if s is not None]
    if scores:
        print(
            "Neighbour cosine scores: "
            f"min={min(scores):.3f} median={statistics.median(scores):.3f} max={max(scores):.3f}"
            "  (tune --min-score/--max-score if contexts look unrelated or duplicated)"
        )
    return contexts


# --------------------------------------------------------------------------- #
# DeepEval side
# --------------------------------------------------------------------------- #
def build_llm(args):
    from deepeval.models import OllamaModel

    kwargs: Dict[str, Any] = {}
    api_key = os.environ.get("OLLAMA_API_KEY")
    if api_key:  # only needed when talking to https://ollama.com directly
        kwargs["headers"] = {"Authorization": f"Bearer {api_key}"}
    return OllamaModel(
        model=args.model,
        base_url=args.ollama_host,
        temperature=args.temperature,
        **kwargs,
    )


def synthesize(contexts: List[Dict[str, Any]], args, llm=None):
    from deepeval.synthesizer import Synthesizer
    from deepeval.synthesizer.config import (
        ConversationalStylingConfig,
        EvolutionConfig,
        FiltrationConfig,
    )

    llm = llm or build_llm(args)
    synthesizer = Synthesizer(
        model=llm,
        async_mode=True,
        max_concurrent=args.max_concurrent,
        filtration_config=FiltrationConfig(
            critic_model=llm,
            synthetic_input_quality_threshold=args.quality_threshold,
            max_quality_retries=args.max_quality_retries,
        ),
        evolution_config=EvolutionConfig(num_evolutions=args.num_evolutions),
        conversational_styling_config=ConversationalStylingConfig(
            scenario_context=SCENARIO_CONTEXT,
            conversational_task=CONVERSATIONAL_TASK,
            participant_roles=PARTICIPANT_ROLES,
            scenario_format=SCENARIO_FORMAT,
            expected_outcome_format=EXPECTED_OUTCOME_FORMAT,
        ),
    )
    return synthesizer.generate_conversational_goldens_from_contexts(
        contexts=[c["texts"] for c in contexts],
        include_expected_outcome=True,
        max_goldens_per_context=args.scenarios_per_context,
        source_files=[sorted(set(c["paper_ids"])) for c in contexts],
        context_chunk_source_files=[c["paper_ids"] for c in contexts],
        target_files_per_context=args.target_files,
    )


def goldens_to_rows(goldens, contexts, args) -> List[Dict[str, Any]]:
    by_texts = {tuple(c["texts"]): c for c in contexts}
    rows, dropped = [], Counter()
    for g in goldens:
        ctx = by_texts.get(tuple(g.context or []))
        if ctx is None:
            dropped["context_not_matched"] += 1
            continue
        meta = g.additional_metadata or {}
        valid = set(ctx["paper_ids"])
        used = sorted(set(meta.get("used_source_files") or []) & valid)
        if len(used) < args.min_papers_used:
            dropped["scenario_used_too_few_papers"] += 1
            continue
        rows.append(
            {
                "scenario": g.scenario,
                "expected_outcome": g.expected_outcome or "",
                "n_papers_used": len(used),
                "used_source_files": json.dumps(used),
                "context_paper_ids": json.dumps(ctx["paper_ids"]),
                "context_chunk_ids": json.dumps(ctx["chunk_ids"]),
                "context_scores": json.dumps(ctx["scores"]),
                "context_chunks": json.dumps(ctx["texts"], ensure_ascii=False),
                "scenario_quality": meta.get("synthetic_scenario_quality"),
                "evolutions": json.dumps(meta.get("evolutions") or []),
            }
        )
    print(
        f"Goldens generated: {len(goldens)}; kept: {len(rows)}; dropped: {dict(dropped)}"
    )
    return rows


CSV_COLUMNS = [
    "golden_id",
    "scenario",
    "expected_outcome",
    "n_papers_used",
    "used_source_files",
    "context_paper_ids",
    "context_chunk_ids",
    "context_scores",
    "context_chunks",
    "scenario_quality",
    "evolutions",
]


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow({"golden_id": f"conv_{i:03d}", **r})
    print(f"Wrote {len(rows)} goldens -> {path}")


def write_contexts(contexts, path):
    with open(path, "w", encoding="utf-8") as f:
        for c in contexts:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Wrote {len(contexts)} contexts -> {path}")


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = p.add_argument_group("Qdrant")
    g.add_argument(
        "--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333")
    )
    g.add_argument("--qdrant-api-key", default=os.environ.get("QDRANT_API_KEY"))
    g.add_argument("--collection", default="ragwise_arxiv")
    g.add_argument("--scroll-batch", type=int, default=2000)

    g = p.add_argument_group("What to generate")
    g.add_argument(
        "--n-goldens",
        type=int,
        default=100,
        help="Final number of conversations wanted.",
    )
    g.add_argument(
        "--oversample",
        type=float,
        default=1.3,
        help="Build this many times more contexts than needed, since some scenarios get filtered out.",
    )
    g.add_argument("--scenarios-per-context", type=int, default=1)
    g.add_argument("--papers-per-context", type=int, default=3)
    g.add_argument(
        "--min-context-papers",
        type=int,
        default=2,
        help="Skip anchors that can't find at least this many distinct papers.",
    )
    g.add_argument(
        "--target-files",
        type=int,
        default=2,
        help="Tell the LLM the scenario must combine at least this many papers.",
    )
    g.add_argument(
        "--min-papers-used",
        type=int,
        default=2,
        help="Drop goldens whose scenario used fewer papers than this (use 1 to keep everything).",
    )
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("Chunk filters")
    g.add_argument("--min-tokens", type=int, default=120)
    g.add_argument("--max-tokens", type=int, default=512)
    g.add_argument("--min-alpha-ratio", type=float, default=0.6)
    g.add_argument(
        "--junk-headings", default=DEFAULT_JUNK_HEADINGS, help="Regex; '' disables."
    )
    g.add_argument(
        "--no-pictures", action="store_true", help="Skip chunks with has_picture=true."
    )
    g.add_argument("--neighbour-pool", type=int, default=60)
    g.add_argument(
        "--min-score",
        type=float,
        default=0.50,
        help="Min cosine similarity to the anchor.",
    )
    g.add_argument(
        "--max-score", type=float, default=0.92, help="Drop near-duplicates above this."
    )

    g = p.add_argument_group("LLM (DeepEval OllamaModel)")
    g.add_argument(
        "--model", default=os.environ.get("OLLAMA_MODEL", "gemma4:31b-cloud")
    )
    g.add_argument(
        "--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    )
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--max-concurrent", type=int, default=4)
    g.add_argument("--num-evolutions", type=int, default=1)
    g.add_argument("--quality-threshold", type=float, default=0.5)
    g.add_argument("--max-quality-retries", type=int, default=3)

    g = p.add_argument_group("Output")
    g.add_argument("--out", default="conversational_goldens.csv")
    g.add_argument("--contexts-out", default="contexts.jsonl")
    g.add_argument(
        "--dry-run",
        action="store_true",
        help="Build + save contexts only; no LLM calls.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)
    contexts = build_contexts(client, args)
    if not contexts:
        sys.exit(
            "No contexts could be built - check --collection and the chunk filters."
        )
    write_contexts(contexts, args.contexts_out)
    if args.dry_run:
        return

    goldens = synthesize(contexts, args)
    rows = goldens_to_rows(goldens, contexts, args)
    rows = rows[: args.n_goldens]
    if len(rows) < args.n_goldens:
        print(
            f"WARNING: only {len(rows)}/{args.n_goldens} goldens survived. "
            "Re-run with a higher --oversample or a different --seed."
        )
    write_csv(rows, args.out)
    papers = {p for r in rows for p in json.loads(r["used_source_files"])}
    print(f"Distinct papers actually used across goldens: {len(papers)}")


if __name__ == "__main__":
    main()
