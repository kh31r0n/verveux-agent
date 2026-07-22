from pydantic import model_validator
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
    # ── Vertex AI rate-limit handling (429 RESOURCE_EXHAUSTED) ───────────────
    # Vertex enforces per-model requests-per-minute quotas that the prospecting
    # fan-out can exhaust. These bound an exponential-backoff retry that lives
    # ONLY in the Vertex provider (openai/anthropic paths are untouched). Set
    # vertex_max_retries=0 to disable retrying.
    vertex_max_retries: int = 5
    vertex_retry_base_seconds: float = 1.0
    vertex_retry_max_seconds: float = 30.0
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
    # ── Prospecting (aurora) ─────────────────────────────────────────────────
    # Autonomous discovery agent. Search is done via Serper (Google SERP REST).
    # `serper_api_key` is REQUIRED — the service refuses to start without it (see
    # the validator below) so a missing key is caught at boot instead of every
    # prospecting run silently completing with 0 results.
    # `prospecting_max_searches` caps SERP calls per run (cost guardrail);
    # `prospecting_fetch_timeout_seconds` bounds each page fetch.
    serper_api_key: str = ""
    prospecting_max_searches: int = 10
    prospecting_fetch_timeout_seconds: float = 8.0
    prospecting_max_results_per_search: int = 10
    # Caps how many extract_and_enrich branches of the Send fan-out run at once.
    # Applied to the prospecting graph ONLY when the tenant's provider is Vertex
    # (its RPM quota is the tight one); other providers keep unbounded fan-out.
    prospecting_vertex_extract_concurrency: int = 5

    @model_validator(mode="after")
    def _require_serper_api_key(self) -> "Settings":
        """Fail fast at startup when the Serper key is absent.

        Without it the prospecting agent's web_search node returns nothing and
        every run completes with 0 found — a silent misconfiguration. We refuse
        to boot instead so the problem surfaces immediately on deploy.
        """
        if not self.serper_api_key.strip():
            raise ValueError(
                "SERPER_API_KEY is not set. The prospecting agent (aurora) "
                "requires it for web search; refusing to start. Set "
                "SERPER_API_KEY in the environment."
            )
        return self


settings = Settings()
