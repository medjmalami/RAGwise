import asyncio
from typing import Any

from openai import AsyncOpenAI

from app.config.config import settings

client = AsyncOpenAI(
    base_url=settings.llm_base_url, api_key=settings.llm_api_key, max_retries=0
)
MSG: Any = [{"role": "user", "content": "Reply with exactly three words."}]


def _g(tc: dict) -> dict[str, Any]:
    # Google documents Gemini-specific fields under a nested "extra_body" key.
    # The outer one is the OpenAI SDK argument, the inner one is the JSON key.
    return {"extra_body": {"extra_body": {"google": {"thinking_config": tc}}}}


CASES: dict[str, dict[str, Any]] = {
    "bare": {},
    "temp+max_completion_tokens": {"temperature": 0.3, "max_completion_tokens": 256},
    "cap_30": {"max_completion_tokens": 30},
    "effort_minimal": {"reasoning_effort": "minimal"},
    "effort_low": {"reasoning_effort": "low"},
    "google_include_thoughts_false": _g({"include_thoughts": False}),
    "google_level_minimal": _g(
        {"thinking_level": "minimal", "include_thoughts": False}
    ),
}


async def main():
    print("model:", settings.llm_model)
    for name, kw in CASES.items():
        for run in (1, 2):
            try:
                r = await client.chat.completions.create(
                    model=settings.llm_model, messages=MSG, **kw
                )
                text = r.choices[0].message.content or ""
                u = r.usage
                print(
                    f"OK   {name} #{run}: thought_leak={'<thought>' in text} "
                    f"tokens={u.completion_tokens if u else None}/"
                    f"{u.total_tokens if u else None} text_end={text[-40:]!r}"
                )
            except Exception as e:
                print(f"FAIL {name} #{run}: {type(e).__name__}: {str(e)[:120]}")
            await asyncio.sleep(1)


asyncio.run(main())
