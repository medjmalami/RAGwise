from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str
    gemini_api_key: str
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_base_url: str
    cohere_api_key: str
    cohere_rerank_model: str = "rerank-v3.5"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()  # type: ignore[call-arg]
