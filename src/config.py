from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    database_url: str
    cognito_user_pool_id: str
    cognito_region: str
    cognito_app_client_id: str = ""
    openai_api_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = "http://localhost:3010"
    nestjs_base_url: str = ""
    webhook_api_key: str = "dev-webhook-secret"


settings = Settings()
