"""
Step 1 - Generate multi-turn *scenarios* (DeepEval ConversationalGoldens) from
chunks stored in Qdrant. Contexts are a MIX of single-paper and multi-paper.

How it works
------------
1. Scan the collection (payload only, no vectors/text) and keep chunks that look
   like usable prose: token_count in range, heading not References/Acknowledgements/...
2. Pick one random anchor chunk per paper (max paper coverage).
3. For each anchor, decide (based on --single-paper-fraction) whether to build:
   - a SINGLE-PAPER context: anchor + N nearest chunks from the SAME paper
   - a MULTI-PAPER context: anchor + nearest chunks from OTHER papers (one per paper)
4. Hand the contexts to DeepEval's Synthesizer.generate_conversational_goldens_from_contexts,
   telling it which paper each chunk came from.
5. Keep only goldens whose scenario actually used >= --min-papers-used papers
   (this filter is applied per-context-type: single-paper contexts only need 1).
6. Write CSV (+ contexts.jsonl for inspection).

Nothing is written to Langfuse or to your database. Output is a CSV.

Example
-------
    # inspect the contexts first (no LLM calls at all):
    python gen_conversational_goldens.py --dry-run --n-goldens 20

    # 70% multi-paper, 30% single-paper:
    python gen_conversational_goldens.py --n-goldens 100 --single-paper-fraction 0.3

    # all multi-paper (original behavior):
    python gen_conversational_goldens.py --n-goldens 100 --single-paper-fraction 0.0
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
CONVERSATIONAL_TASK_MULTI = (
    "The user opens with a concrete question, then asks follow-up questions that "
    "DEPEND on the assistant's previous answers: at least one follow-up MUST refer "
    "back to what the assistant just said using a pronoun ('it', 'that method', "
    "'this approach') or build on a specific claim the assistant made (e.g. "
    "'you said X - does Y work the same way?'). Follow-ups should be hard to answer "
    "without the earlier turns. The overall goal requires combining information "
    "from several different papers (compare, contrast, reconcile)."
    " Vary the number of follow-ups across scenarios: occasionally end after "
    "just one follow-up (deep single drill-down), occasionally add a third "
    "follow-up where the user pushes back on the assistant's framing or asks "
    "for a concrete example. Avoid always producing exactly two follow-ups."
)
CONVERSATIONAL_TASK_SINGLE = (
    "The user opens with a concrete question about one paper, then asks follow-up "
    "questions that DEPEND on the assistant's previous answers: at least one "
    "follow-up MUST refer back to what the assistant just said ('go deeper on "
    "that', 'you claimed X - how does that square with the ablation?'). Follow-ups "
    "should drill into method, results, limitations, or how a specific claim is "
    "supported, and be hard to answer without the earlier turns."
)
# Personas are rotated per BATCH in _synth (each scenario call is independent,
# so 'vary the user' inside one prompt string just yields the same default
# persona every time). The persona carries the goal variety too: engineer ->
# deploy decision, developer -> implementation, reviewer -> critique.
PERSONAS = [
    # PhD student variants — rotated, no single template dominates
    "A PhD student running a literature review for their qualifying exam and an AI research assistant.",
    "A PhD student drafting a related-work section and an AI research assistant.",
    "A PhD student preparing for a committee meeting and an AI research assistant.",
    "A PhD student scoping a thesis chapter and an AI research assistant.",
    "A PhD student looking for research gaps to address and an AI research assistant.",
    # Non-student roles
    "A senior ML engineer deciding whether to deploy a technique in production and an AI research assistant.",
    "A developer implementing a method from the papers and an AI research assistant.",
    "A skeptical reviewer probing weak claims and an AI research assistant.",
    "A domain expert (e.g. medicine, law, finance) checking applicability to their field and an AI research assistant.",
    "A product manager comparing approaches for a buying decision and an AI research assistant.",
    "A research scientist designing a follow-up experiment and an AI research assistant.",
    "An ML platform engineer sizing the infra needed to serve a method at scale and an AI research assistant.",
    "An SRE assessing failure modes and operational risks of a technique before adopting it and an AI research assistant.",
    "A technical writer preparing a deep-dive blog post on a method and an AI research assistant.",
]
SCENARIO_FORMAT = (
    "2-4 sentences, third person: who the user is, their opening question, how they "
    "follow up, and what they want to understand by the end. At least one follow-up "
    "MUST depend on the assistant's previous answer; make that dependency explicit "
    "in the scenario text (e.g. quote or paraphrase what it builds on). Name the "
    "specific paper(s) or system(s) asked about. Never say 'the corpus', 'arXiv papers' "
    "or 'provided text'."
)
EXPECTED_OUTCOME_FORMAT = (
    "1-3 sentences starting with 'The conversation must establish that...', listing "
    "2-4 concrete checkable claims (numbers, named mechanisms, specific comparisons) "
    "from the passages. Include only what the passages actually say; do not "
    "extrapolate or invent mechanisms."
    " When the passage uses specific symbols or notation (e.g., y_inner, y_ref, k_top), "
    "name them in the rubric. If the passages do not contain a specific number, named "
    "mechanism, or named comparison, state that explicitly rather than paraphrasing the abstract."
)


# --------------------------------------------------------------------------- #
# Qdrant side: scanning, anchors, neighbours
# --------------------------------------------------------------------------- #
import re as _re

_CAPTION_RE = _re.compile(r"^(Figure|Fig\.?|Table|Tab\.?)\s*\d+", _re.IGNORECASE)


def chunk_is_eligible(payload, args, junk_re) -> bool:
    # Cheap checks always apply (no text needed — works in scan_candidates
    # which fetches LIGHT_FIELDS only).
    tc = payload.get("token_count")
    if tc is not None and not (args.min_tokens <= tc <= args.max_tokens):
        return False
    if args.no_pictures and payload.get("has_picture"):
        return False
    headings = " / ".join(payload.get("headings") or [])
    if junk_re is not None and junk_re.search(headings):
        return False

    # Content-floor checks: only when text is present (find_neighbours path,
    # which fetches TEXT_FIELDS; scan_candidates path skips these).
    text = (payload.get("text") or "").strip()
    if not text:
        return True  # let scan_candidates keep it; find_neighbours will filter later

    if len(text) < 400:
        return False
    if _CAPTION_RE.match(text) and len(text) < 600:
        return False
    tokens = text.split()
    if tokens:
        proper = sum(1 for t in tokens if t[:1].isupper())
        if proper / len(tokens) > 0.6 and not any(
            kw in text.lower()
            for kw in ("abstract", "introduction", "method", "result", "we propose")
        ):
            return False
    return True


_SENT_RE = re.compile(r"[a-z0-9)\]]\.\s+[A-Z]")


def text_looks_like_prose(text: str, min_alpha_ratio: float) -> bool:
    """Cheap filter for tables / equations / code / number soup."""
    if not text:
        return False
    alpha = sum(ch.isalpha() for ch in text)
    if (alpha / max(len(text), 1)) < min_alpha_ratio:
        return False
    if text.count(" = ") > 4:  # flattened tables ("X, 1 = 0.42. ...")
        return False
    if len(_SENT_RE.findall(text)) < 2:  # code, CLI, pseudo-code, bare captions
        return False
    return True


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


def find_neighbours(
    client: QdrantClient, anchor, args, junk_re, anchor_hash, same_paper: bool = False
):
    """Nearest eligible chunks from OTHER papers (multi-paper mode) or the SAME
    paper (single-paper mode).

    - same_paper=False: one chunk per distinct OTHER paper (for multi-paper contexts)
    - same_paper=True:  N nearest chunks from the anchor's own paper (deduped by hash)
    """
    anchor_paper = anchor.payload["paper_id"]
    target_count = (
        args.single_paper_chunks - 1 if same_paper else args.papers_per_context - 1
    )

    if same_paper:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="paper_id", match=models.MatchValue(value=anchor_paper)
                )
            ]
        )
    else:
        query_filter = models.Filter(
            must_not=[
                models.FieldCondition(
                    key="paper_id", match=models.MatchValue(value=anchor_paper)
                )
            ]
        )

    resp = client.query_points(
        collection_name=args.collection,
        query=str(anchor.id),  # query by stored vector of this point id
        using="dense",
        limit=args.neighbour_pool,
        score_threshold=args.min_score,
        query_filter=query_filter,
        with_payload=TEXT_FIELDS,
        with_vectors=False,
    )

    chosen: List[Any] = []
    seen_hashes = {anchor_hash}
    papers_seen = {anchor_paper}
    for hit in resp.points:
        # Skip the anchor itself (it shows up in same-paper results).
        if str(hit.id) == str(anchor.id):
            continue
        pl = hit.payload or {}
        pid = pl.get("paper_id")
        if not pid:
            continue
        # Multi-paper mode: only one chunk per distinct paper.
        if not same_paper and pid in papers_seen:
            continue
        if hit.score > args.max_score:  # near-duplicate boilerplate
            continue
        h_hash = pl.get("content_hash")
        if h_hash and h_hash in seen_hashes:
            continue
        seen_hashes.add(h_hash)
        if not chunk_is_eligible(pl, args, junk_re):
            continue
        if not text_looks_like_prose(pl.get("text", ""), args.min_alpha_ratio):
            continue
        chosen.append(hit)
        papers_seen.add(pid)
        if len(chosen) >= target_count:
            break
    return chosen


def build_contexts(client: QdrantClient, args) -> List[Dict[str, Any]]:
    junk_re = re.compile(args.junk_headings, re.I) if args.junk_headings else None
    rng = random.Random(args.seed)

    by_paper = scan_candidates(client, args, junk_re)
    papers = sorted(by_paper)
    rng.shuffle(papers)

    target = int(round(args.n_goldens * args.oversample / args.scenarios_per_context))
    n_single_target = int(target * args.single_paper_fraction)
    n_multi_target = target - n_single_target

    contexts: List[Dict[str, Any]] = []
    n_single = n_multi = 0
    skipped = Counter()

    for paper in papers:
        if len(contexts) >= target:
            break

        # Decide: single-paper or multi-paper for this anchor?
        want_single = n_single < n_single_target and (
            n_multi >= n_multi_target or rng.random() < args.single_paper_fraction
        )

        anchor_id = rng.choice(by_paper[paper])
        anchor = fetch_point(client, args.collection, anchor_id)
        if anchor is None:
            skipped["anchor_missing"] += 1
            continue
        a_pl = anchor.payload or {}
        if not text_looks_like_prose(a_pl.get("text", ""), args.min_alpha_ratio):
            skipped["anchor_not_prose"] += 1
            continue

        if want_single:
            # Single-paper: anchor + nearby chunks from the SAME paper.
            neighbours = find_neighbours(
                client,
                anchor,
                args,
                junk_re,
                a_pl.get("content_hash"),
                same_paper=True,
            )
            members = [(str(anchor.id), a_pl, None)] + [
                (str(h.id), h.payload, h.score) for h in neighbours
            ]
            if len(members) < args.min_single_chunks:
                skipped["single_paper_too_few_chunks"] += 1
                continue
            contexts.append(
                {
                    "context_idx": len(contexts),
                    "point_ids": [m[0] for m in members],
                    "chunk_ids": [m[1].get("chunk_id") for m in members],
                    "paper_ids": [m[1]["paper_id"] for m in members],
                    "scores": [m[2] for m in members],
                    "texts": [m[1]["text"] for m in members],
                    "is_single_paper": True,
                }
            )
            n_single += 1
        else:
            # Multi-paper: anchor + one chunk per distinct OTHER paper.
            neighbours = find_neighbours(
                client,
                anchor,
                args,
                junk_re,
                a_pl.get("content_hash"),
                same_paper=False,
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
                    "scores": [m[2] for m in members],
                    "texts": [m[1]["text"] for m in members],
                    "is_single_paper": False,
                }
            )
            n_multi += 1

    print(
        f"Built {len(contexts)} contexts: "
        f"{n_single} single-paper, {n_multi} multi-paper "
        f"(target {target}: {n_single_target} single, {n_multi_target} multi). "
        f"Skipped: {dict(skipped)}"
    )
    sizes = Counter(len(set(c["paper_ids"])) for c in contexts)
    print(f"Papers per context: {dict(sorted(sizes.items()))}")
    # Split cosine stats by context type, since same-paper (high-score, often
    # adjacent prose) and cross-paper (lower-score, related topics) scores are
    # tuned with different --min-score/--max-score thresholds.
    single_scores = [
        s
        for c in contexts
        if c.get("is_single_paper")
        for s in c["scores"]
        if s is not None
    ]
    multi_scores = [
        s
        for c in contexts
        if not c.get("is_single_paper")
        for s in c["scores"]
        if s is not None
    ]
    for label, scores in [
        ("single-paper", single_scores),
        ("multi-paper", multi_scores),
    ]:
        if scores:
            print(
                f"  {label} neighbour cosine scores: "
                f"min={min(scores):.3f} median={statistics.median(scores):.3f} max={max(scores):.3f}"
            )
    return contexts


# --------------------------------------------------------------------------- #
# DeepEval side
# --------------------------------------------------------------------------- #
class RobustOllamaModel:
    """Build an OllamaModel subclass that strips markdown code fences from
    LLM responses before JSON-schema validation.

    Why: gemma4:31b-cloud (and some other Ollama cloud models) wrap structured
    output in ```json ... ``` even when format=json_schema is passed. deepeval's
    stock OllamaModel then calls schema.model_validate_json(content) directly,
    which raises ValidationError on the fence. The retry policy can't help
    (ValidationError isn't retryable), so every scenario generation fails.

    Returns a dynamically created subclass of the actual OllamaModel class
    installed in the user's env (so we don't pin a deepeval version).
    """

    @staticmethod
    def _clean_json_text(text: str) -> str:
        import re

        if not text:
            return text
        s = text.strip()

        # Strip ```json ... ``` or ``` ... ``` fences (multi-line aware).
        if s.startswith("```"):
            first_nl = s.find("\n")
            if first_nl != -1:
                first_line = s[:first_nl].strip().lower()
                # Drop the opening fence line if it's just ``` or ```json
                if first_line in ("```", "```json", "```json5", "```jsonc"):
                    s = s[first_nl + 1 :]
            s = s.rstrip()
            if s.endswith("```"):
                s = s[:-3].rstrip()

        # Some models prepend prose like "Here is the JSON:" before the object.
        # Find the outermost { ... } or [ ... ] and use that.
        if s and s[0] not in "{[":
            start_obj = s.find("{")
            start_arr = s.find("[")
            starts = [x for x in (start_obj, start_arr) if x != -1]
            if starts:
                start = min(starts)
                opener = s[start]
                closer = "}" if opener == "{" else "]"
                end = s.rfind(closer)
                if end > start:
                    s = s[start : end + 1]

        # Strip trailing commas before } or ] (LLMs sometimes add them).
        s = re.sub(r",\s*([\]}])", r"\1", s)
        return s

    @staticmethod
    def _repair_partial(content: str, schema_cls) -> Optional[Any]:
        """When the LLM emits valid JSON but with missing required fields
        (e.g. a scenario item with used_source_files but no scenario text),
        try to recover a partial result instead of failing the whole batch.

        Strategy: parse the raw JSON, walk list items, drop the ones missing
        required fields, validate the rest. Returns the schema instance if at
        least one item survives, else None.
        """
        import json as _json

        from pydantic import ValidationError

        try:
            raw = _json.loads(content)
        except Exception:
            return None

        # The schema is a *List wrapper (e.g. ConversationalScenarioList) with
        # a `data: List[Inner]` field, OR a bare inner model.
        inner_cls = None
        inner_list = None

        if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
            # Find the inner class from the schema's field annotations.
            for fname, finfo in schema_cls.model_fields.items():
                ftype = finfo.annotation
                origin = getattr(ftype, "__origin__", None)
                args = getattr(ftype, "__args__", ())
                if origin is list and args:
                    candidate = args[0]
                    # Strip Optional[X]
                    if hasattr(candidate, "__args__"):
                        candidate = candidate.__args__[0]
                    if hasattr(candidate, "model_validate"):
                        inner_cls = candidate
                        inner_list = raw["data"]
                        break
        elif isinstance(raw, list):
            for fname, finfo in schema_cls.model_fields.items():
                ftype = finfo.annotation
                origin = getattr(ftype, "__origin__", None)
                args = getattr(ftype, "__args__", ())
                if origin is list and args and hasattr(args[0], "model_validate"):
                    inner_cls = args[0]
                    inner_list = raw
                    break

        if inner_cls is None or not inner_list:
            return None

        # Validate each item individually; keep the ones that pass.
        kept = []
        for item in inner_list:
            if not isinstance(item, dict):
                continue
            try:
                kept.append(inner_cls(**item))
            except ValidationError:
                continue  # drop malformed item

        if not kept:
            return None

        # Re-wrap in the outer list schema: find the field name that holds the list.
        for fname, finfo in schema_cls.model_fields.items():
            ftype = finfo.annotation
            origin = getattr(ftype, "__origin__", None)
            if origin is list:
                return schema_cls(**{fname: kept})
        return None

    @staticmethod
    def _parse(content: str, schema_cls, clean_fn, repair_fn) -> Optional[Any]:
        """Try to parse LLM output into the schema. Returns None if unparseable.

        Tries in order:
        1. Clean (strip fences, trailing commas) -> model_validate_json
        2. json_repair (if installed) -> model_validate_json
        3. repair_partial (drop malformed list items) on each candidate
        """
        candidates = [clean_fn(content)]
        # json_repair handles truncated / unbalanced JSON the LLM sometimes emits.
        try:
            import json_repair

            repaired_text = json_repair.repair_json(content)
            if repaired_text and repaired_text != content:
                candidates.append(clean_fn(str(repaired_text)))
        except Exception:
            pass  # json_repair not installed or failed

        for c in candidates:
            try:
                return schema_cls.model_validate_json(c)
            except Exception:
                repaired = repair_fn(c, schema_cls)
                if repaired is not None:
                    return repaired
        return None

    @classmethod
    def create(cls):
        from deepeval.models import OllamaModel

        clean = cls._clean_json_text
        repair_partial = cls._repair_partial
        parse = cls._parse

        class _RobustOllamaModel(OllamaModel):
            # Re-apply the same retry policy the stock class uses.
            # (create_retry_decorator returns a fresh decorator instance with
            # the same config; module-level retry_ollama is private.)
            @staticmethod
            def _retry():
                try:
                    from deepeval.constants import ProviderSlug as PS
                    from deepeval.models.retry_policy import (
                        create_retry_decorator,
                    )

                    return create_retry_decorator(PS.OLLAMA)
                except Exception:
                    # If deepeval internals move, return a no-op decorator
                    # so the model still works (just without retries).
                    def _noop(fn):
                        return fn

                    return _noop

            # Override the async path (used by Synthesizer when async_mode=True).
            # On schema failure, we try json_repair + repair_partial. If that
            # fails, we re-ask the model with a temperature bump (at temp 0 the
            # model would return the same broken output).
            async def a_generate(self, prompt, schema=None, audio=None):
                chat_model = self.load_model(async_mode=True)
                messages = self._prompt_messages(prompt, audio)
                content = ""  # initialized so the post-loop raise can reference it
                for attempt in range(3):
                    response = await chat_model.chat(
                        model=self.name,
                        messages=messages,
                        format=schema.model_json_schema() if schema else None,
                        options={
                            **{
                                "temperature": (self.temperature or 0.0) + 0.2 * attempt
                            },
                            **self.generation_kwargs,
                        },
                    )
                    content = response.message.content
                    if not schema:
                        return content, 0
                    parsed = parse(content, schema, clean, repair_partial)
                    if parsed is not None:
                        return parsed, 0
                raise ValueError(
                    f"LLM returned unparseable JSON after 3 attempts "
                    f"(last content: {content[:200]!r})"
                )

            # Same for the sync path, for completeness.
            def generate(self, prompt, schema=None, audio=None):
                chat_model = self.load_model()
                messages = self._prompt_messages(prompt, audio)
                content = ""  # initialized so the post-loop raise can reference it
                for attempt in range(3):
                    response = chat_model.chat(
                        model=self.name,
                        messages=messages,
                        format=schema.model_json_schema() if schema else None,
                        options={
                            **{
                                "temperature": (self.temperature or 0.0) + 0.2 * attempt
                            },
                            **self.generation_kwargs,
                        },
                    )
                    content = response.message.content
                    if not schema:
                        return content, 0
                    parsed = parse(content, schema, clean, repair_partial)
                    if parsed is not None:
                        return parsed, 0
                raise ValueError(
                    f"LLM returned unparseable JSON after 3 attempts "
                    f"(last content: {content[:200]!r})"
                )

        # Wrap with retry decorator (best-effort; no-op if import fails).
        # cast(Any, ...) silences pyright: the tenacity wrapper returns a
        # _Wrapped object whose inferred return type widens to include None,
        # which doesn't match the bound a_generate signature. The runtime
        # behavior is unchanged.
        from typing import Any as _Any
        from typing import cast as _cast

        retry_decorator = _RobustOllamaModel._retry()
        _RobustOllamaModel.a_generate = _cast(
            _Any, retry_decorator(_RobustOllamaModel.a_generate)
        )
        _RobustOllamaModel.generate = _cast(
            _Any, retry_decorator(_RobustOllamaModel.generate)
        )
        return _RobustOllamaModel


def build_llm(args):
    RobustOllamaModelCls = RobustOllamaModel.create()

    kwargs: Dict[str, Any] = {}
    api_key = os.environ.get("OLLAMA_API_KEY")
    if api_key:  # only needed when talking to https://ollama.com directly
        kwargs["headers"] = {"Authorization": f"Bearer {api_key}"}
    return RobustOllamaModelCls(
        model=args.model,
        base_url=args.ollama_host,
        temperature=args.temperature,
        **kwargs,
    )


def _synth_batch(contexts, args, llm, task, target_files, participant_roles):
    """Run the Synthesizer on one batch of contexts. May raise on LLM failure."""
    from deepeval.synthesizer import Synthesizer
    from deepeval.synthesizer.config import (
        ConversationalStylingConfig,
        EvolutionConfig,
        FiltrationConfig,
    )

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
            conversational_task=task,
            participant_roles=participant_roles,
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
        target_files_per_context=target_files,
    )


def _synth(contexts, args, llm, task, target_files, batch_size=5):
    """Process contexts in batches so one LLM failure only costs that batch,
    not the whole run. Personas rotate per batch (see PERSONAS note above)."""
    out: List[Any] = []
    for b, i in enumerate(range(0, len(contexts), batch_size)):
        batch = contexts[i : i + batch_size]
        role = PERSONAS[b % len(PERSONAS)]
        try:
            out += _synth_batch(
                batch, args, llm, task, target_files, participant_roles=role
            )
        except Exception as e:
            print(
                f"  [batch {i // batch_size} failed: {type(e).__name__}: "
                f"{str(e)[:120]}] skipping {len(batch)} contexts",
                file=sys.stderr,
            )
    return out


def synthesize(contexts: List[Dict[str, Any]], args, llm=None):
    """Split by context type so each slice gets the right prompt and target_files.

    - Multi-paper  -> task='combine several papers', target_files=args.target_files
    - Single-paper -> task='drill into one paper',    target_files=1

    (Verified on deepeval 4.3.2: target_files_per_context and context_chunk_source_files
     both exist on generate_conversational_goldens_from_contexts - line 2308 of
     synthesizer.py. No inspect.signature shim needed.)
    """
    llm = llm or build_llm(args)
    multi = [c for c in contexts if not c.get("is_single_paper")]
    single = [c for c in contexts if c.get("is_single_paper")]
    goldens: List[Any] = []
    if multi:
        print(
            f"Synthesizing {len(multi)} multi-paper contexts (target_files={args.target_files})..."
        )
        goldens += _synth(
            multi, args, llm, CONVERSATIONAL_TASK_MULTI, args.target_files
        )
    if single:
        print(f"Synthesizing {len(single)} single-paper contexts (target_files=1)...")
        goldens += _synth(single, args, llm, CONVERSATIONAL_TASK_SINGLE, 1)
    return goldens


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
        # Single-paper contexts: DeepEval often leaves used_source_files empty
        # when target_files=1 (the LLM doesn't bother filling it since there's
        # only one paper). By construction, the scenario MUST have used the
        # context's single paper, so populate it from the context.
        if ctx.get("is_single_paper") and not used:
            used = sorted(valid)
        # The min_papers filter is per-context-type: single-paper contexts only
        # need 1 paper; multi-paper contexts must use >= --min-papers-used.
        min_papers = 1 if ctx.get("is_single_paper") else args.min_papers_used
        if len(used) < min_papers:
            dropped["scenario_used_too_few_papers"] += 1
            continue
        # Auto-trim: keep only chunks belonging to papers the scenario actually
        # used. Multi-paper contexts often ship a 3rd paper the scenario
        # ignored; trimming it keeps the row's context_chunks aligned with the
        # rubric (and the n_papers_used count honest).
        used_set = set(used)
        keep_idx = [i for i, pid in enumerate(ctx["paper_ids"]) if pid in used_set]
        rows.append(
            {
                "scenario": g.scenario,
                "expected_outcome": g.expected_outcome or "",
                "n_papers_used": len(used),
                "used_source_files": json.dumps(used),
                "context_paper_ids": json.dumps(
                    [ctx["paper_ids"][i] for i in keep_idx]
                ),
                "context_chunk_ids": json.dumps(
                    [ctx["chunk_ids"][i] for i in keep_idx]
                ),
                "context_scores": json.dumps([ctx["scores"][i] for i in keep_idx]),
                "context_chunks": json.dumps(
                    [ctx["texts"][i] for i in keep_idx], ensure_ascii=False
                ),
                "is_single_paper": ctx.get("is_single_paper", False),
                "scenario_quality": meta.get("synthetic_scenario_quality"),
                "evolutions": json.dumps(meta.get("evolutions") or []),
            }
        )
    print(
        f"Goldens generated: {len(goldens)}; kept: {len(rows)}; dropped: {dict(dropped)}"
    )
    return rows


def select_rows(rows, args):
    """Pick final n_goldens preserving the requested single/multi ratio.

    rows[:n_goldens] would skew toward single-paper, because multi-paper goldens
    get filtered out more often (harder to actually use >=2 papers). So we select
    by quota, then top up from leftovers if either slice is short.
    """
    rng = random.Random(args.seed + 1)  # different stream from context-building
    single = [r for r in rows if r["is_single_paper"]]
    multi = [r for r in rows if not r["is_single_paper"]]
    rng.shuffle(single)
    rng.shuffle(multi)
    n_s = round(args.n_goldens * args.single_paper_fraction)
    picked = single[:n_s] + multi[: args.n_goldens - n_s]
    if len(picked) < args.n_goldens:  # top up from leftovers
        rest = single[n_s:] + multi[args.n_goldens - n_s :]
        picked += rest[: args.n_goldens - len(picked)]
    rng.shuffle(picked)
    return picked


CSV_COLUMNS = [
    "golden_id",
    "scenario",
    "expected_outcome",
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
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("Context mix (single-paper vs multi-paper)")
    g.add_argument(
        "--single-paper-fraction",
        type=float,
        default=0.3,
        help="Fraction of contexts built from a SINGLE paper. "
        "0.0 = all multi-paper (original behavior). 0.3 = 30% single, 70% multi.",
    )
    g.add_argument(
        "--single-paper-chunks",
        type=int,
        default=3,
        help="For single-paper contexts: how many chunks to pull from the same paper "
        "(anchor + this-minus-1 neighbours). 3 = anchor + 2 nearby chunks.",
    )
    g.add_argument(
        "--min-single-chunks",
        type=int,
        default=3,
        help="Minimum chunks a single-paper context must have (anchor + neighbours). "
        "Anchors with fewer eligible same-paper neighbours are skipped.",
    )

    g = p.add_argument_group("Multi-paper contexts")
    g.add_argument("--papers-per-context", type=int, default=2)
    g.add_argument(
        "--min-context-papers",
        type=int,
        default=2,
        help="Skip multi-paper anchors that can't find at least this many distinct papers.",
    )
    g.add_argument(
        "--target-files",
        type=int,
        default=2,
        help="Tell the LLM multi-paper scenarios must combine at least this many papers. "
        "Auto-set to None when --single-paper-fraction > 0.",
    )
    g.add_argument(
        "--min-papers-used",
        type=int,
        default=2,
        help="Drop multi-paper goldens whose scenario used fewer papers than this. "
        "Single-paper goldens are always kept (filter is 1 for them).",
    )

    g = p.add_argument_group("Chunk filters")
    g.add_argument("--min-tokens", type=int, default=120)
    g.add_argument("--max-tokens", type=int, default=512)
    g.add_argument("--min-alpha-ratio", type=float, default=0.7)
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
        default=0.72,
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
    rows = select_rows(rows, args)
    if len(rows) < args.n_goldens:
        print(
            f"WARNING: only {len(rows)}/{args.n_goldens} goldens survived. "
            "Re-run with a higher --oversample or a different --seed."
        )
    write_csv(rows, args.out)
    n_single = sum(1 for r in rows if r.get("is_single_paper"))
    print(
        f"Final: {n_single} single-paper, {len(rows) - n_single} multi-paper goldens."
    )


if __name__ == "__main__":
    main()
