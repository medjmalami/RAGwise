import asyncio

from openai import AsyncOpenAI

from app.config.config import settings


async def main():
    client = AsyncOpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    async for m in client.models.list():
        if "gemma" in m.id.lower():
            print(m.id)


asyncio.run(main())
