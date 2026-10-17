import asyncio
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.services.conversation_service import TITLE_MAX_LEN, build_title

logger = logging.getLogger(__name__)

TITLE_PROMPT = (
    "Write a short title (max 6 words) for a conversation that starts with "
    "the user's message. Use the same language as the message. "
    "Output only the title: no quotes, no trailing punctuation, no explanation."
)


def _clean(raw) -> str:
    text = raw if isinstance(raw, str) else ""
    text = " ".join(text.split()).strip(" \"'`.")
    return text[:TITLE_MAX_LEN]


async def generate_title(llm: BaseChatModel, query: str, timeout: float = 8.0) -> str:
    """Never raises (except cancellation): falls back to the truncated query."""
    try:
        msg = await asyncio.wait_for(
            llm.ainvoke([SystemMessage(TITLE_PROMPT), HumanMessage(query[:1000])]),
            timeout=timeout,
        )
        title = _clean(msg.content)
        if title and "<unused" not in title:
            return title
    except Exception:
        logger.warning("Title generation failed, using fallback", exc_info=True)
    return build_title(query)
