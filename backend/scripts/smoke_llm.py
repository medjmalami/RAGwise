import asyncio

from langchain_openai import ChatOpenAI

from app.services.llm_factory import build_llms

HARD_Q = (
    "Compare dense retrieval and sparse retrieval for question answering. "
    "Explain the trade-offs step by step and say when hybrid search helps."
)


async def check(name: str, llm: ChatOpenAI, prompt: str, n: int = 3) -> None:
    for i in range(n):
        try:
            r = await llm.ainvoke(prompt)
            text = r.content if isinstance(r.content, str) else str(r.content)
            print(
                f"OK   {name} #{i + 1}: leak={'<thought>' in text} "
                f"usage={r.usage_metadata} text={text[:70]!r}"
            )
        except Exception as e:
            print(f"FAIL {name} #{i + 1}: {type(e).__name__}: {str(e)[:100]}")


async def main() -> None:
    llms = build_llms()
    await check("title", llms.title, "Reply with exactly three words.")
    await check(
        "rewrite",
        llms.rewrite,
        "Rewrite as a standalone search query: 'what about its limits?' "
        "(previous topic: RAG chunking). Output only the query.",
    )
    await check("main-hard", llms.main, HARD_Q)

    buf, chunks = "", 0
    async for c in llms.main.astream(HARD_Q):
        chunks += 1
        if isinstance(c.content, str):
            buf += c.content
    print(f"stream: chunks={chunks} leak={'<thought>' in buf} text={buf[:70]!r}")


asyncio.run(main())
