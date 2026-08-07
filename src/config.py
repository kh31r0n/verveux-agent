from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Tolerate env vars the agent doesn't consume (the shared .env carries
        # backend + observability keys like LANGSMITH_API_KEY).
        extra="ignore",
    )

    database_url: str
    cognito_user_pool_id: str
    cognito_region: str
    cognito_app_client_id: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    vertex_service_account_json: str = ""
    vertex_project_id: str = ""
    vertex_location: str = ""
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = "http://localhost:3010"
    nestjs_base_url: str = ""
    webhook_api_key: str = "dev-webhook-secret"

    # ── FinOps spend-file repair ──────────────────────────────────────────────
    # S3 bucket/region the agent reads originals from and writes corrected files
    # to. Same names the backend's UploadService uses, so both point at one bucket.
    aws_s3_bucket: str = ""
    aws_s3_region: str = "us-east-1"
    # Optional HMAC-SHA256 over the raw request body, on top of X-System-Key.
    # Empty disables the check.
    agent_machine_hmac_secret: str = ""
    # Model used for column-mapping inference when repairing spend files. The chat
    # providers resolve their models per-request (see providers/registry.py); this
    # endpoint has no request-scoped model, so it reads its own setting.
    spend_fix_model: str = "gpt-5.6-luna"


settings = Settings()
