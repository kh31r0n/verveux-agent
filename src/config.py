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
    anthropic_api_key: str = ""
    vertex_service_account_json: str = ""
    vertex_project_id: str = ""
    vertex_location: str = ""
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_host: str = "http://localhost:3010"
    nestjs_base_url: str = ""
    webhook_api_key: str = "dev-webhook-secret"
    # ── Multi-message coalescing ─────────────────────────────────────────────
    # WhatsApp users often split one thought across several rapid messages.
    # After a turn acquires its thread's run slot, it waits until no new
    # fragment has arrived for `message_settle_seconds` (checked in windows of
    # that size) before invoking the graph, so the burst is answered as ONE
    # turn. `message_settle_max_seconds` caps the total wait so a steady
    # stream of fragments can't stall the reply past the backend's timeout.
    # Set message_settle_seconds=0 to disable the settle wait (runs are still
    # serialized per thread).
    message_settle_seconds: float = 2.0
    message_settle_max_seconds: float = 10.0
    # ── Query normalization ──────────────────────────────────────────────────
    # Fleet-wide kill switch for the query_normalizer node. Rollout is driven
    # per tenant by the backend (TenantSettings.queryNormalizationEnabled,
    # forwarded on every /chat/stream call); this env var can disable the
    # feature everywhere regardless of tenant flags.
    query_normalization_enabled: bool = True


settings = Settings()
