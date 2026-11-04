from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    gemini_api_key: str
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_base_url: str
    cohere_api_key: str
    cohere_rerank_model: str = "rerank-v3.5"
    otel_service_name: str
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    access_token_ttl_minutes: int = 15
    refresh_token_ttl_days: int = 30
    cookie_secure: bool = True
    history_trigger_tokens: int = 5000
    history_target_tokens: int = 2500
    history_min_keep_turns: int = 3
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_thinking_control: Literal[
        "unsupported",
        "reasoning_effort",
        "reasoning_effort_minimal",
        "openrouter",
        "chat_template_kwargs",
    ] = "unsupported"
    llm_max_retries: int = 6
    llm_extra_body: dict = {}
    llm_main_thinking: bool = True
    llm_timeout: float = 120.0
    llm_stream_usage: bool = True
    llm_summary_model: Optional[str] = None
    llm_title_model: Optional[str] = None
    llm_rewrite_model: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()  # type: ignore[call-arg]
