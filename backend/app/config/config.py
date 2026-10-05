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

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()  # type: ignore[call-arg]
