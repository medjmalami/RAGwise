from dataclasses import dataclass
from typing import Optional, TypeVar

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.config.config import settings
from app.services.vision_llm import PathImageChatOpenAI

T = TypeVar("T", bound=ChatOpenAI)

# What to send to turn thinking off, per server family.
_NO_THINK: dict[str, dict] = {
    "unsupported": {},
    "reasoning_effort": {"reasoning_effort": "none"},
    "reasoning_effort_minimal": {"reasoning_effort": "minimal"},
    "openrouter": {"reasoning": {"effort": "none"}},
    "chat_template_kwargs": {"chat_template_kwargs": {"enable_thinking": False}},
}


def _extra_body(thinking: bool) -> Optional[dict]:
    body = dict(settings.llm_extra_body)
    if not thinking:
        body.update(_NO_THINK[settings.llm_thinking_control])
    return body or None


def _make(
    cls: type[T],
    *,
    model: Optional[str],
    temperature: float,
    max_tokens: Optional[int] = None,
    thinking: bool = True,
) -> T:
    return cls(
        model=model or settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=SecretStr(settings.llm_api_key),
        temperature=temperature,
        max_completion_tokens=max_tokens,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
        stream_usage=settings.llm_stream_usage,
        extra_body=_extra_body(thinking),
    )


@dataclass(frozen=True)
class LLMs:
    main: PathImageChatOpenAI
    summary: ChatOpenAI
    title: ChatOpenAI
    rewrite: ChatOpenAI


def build_llms() -> LLMs:
    return LLMs(
        main=_make(
            PathImageChatOpenAI,
            model=None,
            temperature=0.2,
            thinking=settings.llm_main_thinking,
        ),
        summary=_make(
            ChatOpenAI,
            model=settings.llm_summary_model,
            temperature=0,
            max_tokens=512,
            thinking=False,
        ),
        title=_make(
            ChatOpenAI,
            model=settings.llm_title_model,
            temperature=0.3,
            max_tokens=64,
            thinking=False,
        ),
        rewrite=_make(
            ChatOpenAI,
            model=settings.llm_rewrite_model,
            temperature=0,
            max_tokens=128,
            thinking=False,
        ),
    )
