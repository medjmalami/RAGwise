"""Rolling summarization of old conversation turns.

The full conversation stays in the DB (the UI shows everything). Only what the
LLM sees is capped: [summary of old turns] + [recent turns verbatim].

`estimate_tokens` and `plan_compaction` are pure (no I/O) so they are easy to
unit-test. `compact_history` is the orchestrator the controller calls.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import uuid
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.config import settings
from app.services.conversation_service import save_summary

logger = logging.getLogger(__name__)

# Conservative on purpose: there is no exact tokenizer for the Ollama cloud
# model, and French/Arabic text tokenizes worse than English. Swap
# estimate_tokens() if you later get real counts.
CHARS_PER_TOKEN = 3
MESSAGE_OVERHEAD_TOKENS = 4

# What we assume a summary costs when planning (the prompt asks for <= 250 words
# and the summary LLM has num_predict=512).
SUMMARY_BUDGET_TOKENS = 500
SUMMARY_MAX_CHARS = 4000

# Limits on what we feed the summarizer (it runs with num_ctx=8192).
SUMMARY_MAX_CHARS_PER_MSG = 2500
SUMMARY_MAX_INPUT_CHARS = 20000

SUMMARY_SYSTEM_PROMPT = """You maintain a running summary of a conversation between a user \
and an AI assistant that answers questions about scientific papers (arXiv).

You receive the previous summary (possibly empty) and new messages. Write an updated \
summary that merges both.

Rules:
- Keep exactly as written: paper IDs, model/dataset/method names, numbers, and any \
constraint, goal or preference the user stated.
- Record what the user asked about, which papers were involved, and the conclusions \
reached or questions left open.
- Drop greetings and long explanations; details can be retrieved from the papers again.
- Do NOT add anything that is not in the conversation. Do NOT answer the user.
- Use the same language as the conversation. At most 250 words.
- Output only the summary: no preamble, no title."""


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def estimate_tokens(text: Optional[str]) -> int:
    if not text:
        return 0
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def message_tokens(m: dict) -> int:
    return estimate_tokens(m.get("content")) + MESSAGE_OVERHEAD_TOKENS


def _turn_starts(history: list[dict]) -> list[int]:
    """Index of the first message of each turn (a turn starts at a user message)."""
    starts = [i for i, m in enumerate(history) if m.get("role") == "user"]
    if not starts or starts[0] != 0:
        starts.insert(0, 0)  # stray leading messages join the first turn
    return starts


def plan_compaction(
    summary: Optional[str],
    history: list[dict],
    *,
    trigger_tokens: int,
    target_tokens: int,
    min_keep_turns: int,
    summary_budget_tokens: int = SUMMARY_BUDGET_TOKENS,
) -> Optional[int]:
    """Return the index where `history` should be split, or None to do nothing.

    history[:split] gets folded into the summary, history[split:] stays verbatim.
    Always cuts on a turn boundary, always keeps >= min_keep_turns recent turns.
    """
    total = estimate_tokens(summary) + sum(message_tokens(m) for m in history)
    if total <= trigger_tokens:
        return None

    starts = _turn_starts(history)
    max_drop = len(starts) - max(1, min_keep_turns)
    if max_drop < 1:
        return None  # everything left is "recent"; nothing we are allowed to cut

    bounds = starts + [len(history)]
    turn_tokens = [
        sum(message_tokens(m) for m in history[bounds[i] : bounds[i + 1]])
        for i in range(len(starts))
    ]
    tail = sum(turn_tokens)

    dropped = 0
    # Drop at least one turn, then keep dropping until summary + tail fits the target.
    while dropped < max_drop and (
        dropped == 0 or summary_budget_tokens + tail > target_tokens
    ):
        tail -= turn_tokens[dropped]
        dropped += 1
    return bounds[dropped]


def _format_one(m: dict, max_chars: int) -> str:
    is_user = m.get("role") == "user"
    content = (m.get("content") or "").strip()
    if len(content) > max_chars:
        content = content[:max_chars].rstrip() + " ..."
    line = f"{'User' if is_user else 'Assistant'}: {content}"
    papers = m.get("paper_ids") or []
    if papers and not is_user:
        line += f"\n[Papers retrieved for this answer: {', '.join(papers)}]"
    return line


def _format_messages(messages: list[dict]) -> str:
    per_msg = SUMMARY_MAX_CHARS_PER_MSG
    while True:
        text = "\n\n".join(_format_one(m, per_msg) for m in messages)
        if len(text) <= SUMMARY_MAX_INPUT_CHARS or per_msg <= 200:
            return text
        per_msg //= 2


def _clean_summary(raw) -> str:
    text = raw if isinstance(raw, str) else ""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:SUMMARY_MAX_CHARS]


# ---------------------------------------------------------------------------
# LLM + orchestration
# ---------------------------------------------------------------------------
async def summarize_turns(
    llm: BaseChatModel,
    prior_summary: Optional[str],
    messages: list[dict],
    *,
    config: Optional[RunnableConfig] = None,
    timeout: float = 45.0,
) -> Optional[str]:
    """Fold `messages` into `prior_summary`. Returns None on failure (never raises,
    except cancellation), so the caller can tell failure apart from an empty result."""
    if not messages:
        return prior_summary
    prompt = (
        f"Previous summary:\n{prior_summary or '(none)'}\n\n"
        f"New messages:\n{_format_messages(messages)}\n\n"
        "Updated summary:"
    )
    try:
        msg = await asyncio.wait_for(
            llm.ainvoke(
                [SystemMessage(SUMMARY_SYSTEM_PROMPT), HumanMessage(prompt)],
                config=config,
            ),
            timeout=timeout,
        )
        text = _clean_summary(msg.content)
        # Same guard as title_service: some Gemma outputs leak "<unused..>" tokens.
        if text and "<unused" not in text:
            return text
        logger.warning("Summary came back empty or malformed")
    except Exception:
        logger.warning("Summarization failed", exc_info=True)
    return None


async def compact_history(
    llm: BaseChatModel,
    db: AsyncSession,
    *,
    conversation_id: uuid.UUID,
    summary: Optional[str],
    summary_upto_id: Optional[int],
    history: list[dict],
    config: Optional[RunnableConfig] = None,
) -> tuple[Optional[str], list[dict]]:
    """Return (summary, history) to give the graph. Never fails the request.

    `history` must be the messages AFTER `summary_upto_id` (as get_history returns).
    If it is over the trigger, the oldest turns are folded into the summary, the
    summary is persisted, and only the recent tail is returned.
    """
    try:
        split = plan_compaction(
            summary,
            history,
            trigger_tokens=settings.history_trigger_tokens,
            target_tokens=settings.history_target_tokens,
            min_keep_turns=settings.history_min_keep_turns,
        )
        if split is None:
            return summary, history

        head, tail = history[:split], history[split:]
        new_summary = await summarize_turns(llm, summary, head, config=config)

        if new_summary is None:
            # History is only slightly over the trigger, so sending it as-is is
            # safe and loses nothing; the next request retries. Only if it has
            # grown far past the trigger do we drop the old turns for this request.
            total = estimate_tokens(summary) + sum(message_tokens(m) for m in history)
            if total > 2 * settings.history_trigger_tokens:
                logger.warning(
                    "Summarization failed and history is ~%d tokens; dropping %d "
                    "old messages for this request (conversation %s)",
                    total,
                    len(head),
                    conversation_id,
                )
                return summary, tail
            logger.warning(
                "Summarization failed; keeping full history (~%d tokens) "
                "(conversation %s)",
                total,
                conversation_id,
            )
            return summary, history

        try:
            saved = await save_summary(
                db,
                conversation_id,
                summary=new_summary,
                upto_message_id=head[-1]["id"],
                expected_upto_id=summary_upto_id,
            )
            await db.commit()
            if not saved:
                logger.info(
                    "Concurrent compaction on conversation %s; using ours for this request",
                    conversation_id,
                )
        except SQLAlchemyError:
            await db.rollback()
            logger.exception(
                "Failed to persist summary for conversation %s", conversation_id
            )
        return new_summary, tail
    except Exception:
        logger.exception("History compaction failed; using uncompacted history")
        return summary, history
