from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Keys in a dotenv FILE that match no field would otherwise abort the
        # boot (pydantic-settings v2 defaults to extra="forbid" for that source;
        # unmatched OS env vars are simply never read). Ignoring them lets a
        # stale local .env — or a Cloud Run revision still sending the retired
        # COGNITO_* placeholders — start cleanly during a config migration.
        extra="ignore",
    )

    database_url: str
    # ── Service-to-service authentication (src/auth/service_auth.py) ─────────
    # `service_auth_audience` is this service's own Cloud Run URL: the `aud`
    # claim callers must request their ID token for. Empty disables OIDC
    # verification entirely, which is the local-dev and test posture — the
    # shared secret is then the only accepted credential.
    #
    # `service_auth_allowed_service_accounts` is a comma-separated allowlist of
    # caller SA emails. Verification fails closed when it is empty but an
    # audience is set, since an unrestricted audience would accept a token from
    # any Google identity able to reach the service.
    #
    # `allow_shared_secret_auth` keeps the legacy WEBHOOK_API_KEY header path
    # alive. True through the migration (and always, off Cloud Run); flip to
    # False in production once every caller sends an ID token.
    service_auth_audience: str = ""
    service_auth_allowed_service_accounts: str = ""
    allow_shared_secret_auth: bool = True
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    # Gemini Enterprise Agent Platform. All three are optional: with none set
    # the provider authenticates by ADC, which on Cloud Run is the attached
    # service account (roles/aiplatform.user). `gemini_location` defaults to
    # "global" inside the provider — Gemini 3 is not published to regions.
    gemini_service_account_json: str = ""
    gemini_project_id: str = ""
    gemini_location: str = ""
    # ── Gemini rate-limit handling (429 RESOURCE_EXHAUSTED) ──────────────────
    # Agent Platform enforces per-model requests-per-minute quotas that the
    # prospecting fan-out can exhaust. These bound an exponential-backoff retry
    # that lives ONLY in the Gemini provider (openai/anthropic paths are
    # untouched). Set gemini_max_retries=0 to disable retrying.
    gemini_max_retries: int = 5
    gemini_retry_base_seconds: float = 1.0
    gemini_retry_max_seconds: float = 30.0
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
    # Applied to the prospecting graph ONLY when the tenant's provider is Gemini
    # (its RPM quota is the tight one); other providers keep unbounded fan-out.
    prospecting_gemini_extract_concurrency: int = 5
    # ── Prospecting self-improvement ─────────────────────────────────────────
    # In-run refinement loop: when a run's quality survivors fall short of
    # `prospecting_min_quality_prospects`, the graph refines its own queries and
    # searches again, up to `prospecting_max_iterations` TOTAL passes (>=1). A
    # value of 1 disables the loop (single pass, today's behaviour). Cross-run
    # strategy memory (best/avoid queries per niche) is read at run start and
    # updated at run end via a LangGraph PostgresStore. Human good/bad feedback
    # is injected into the extraction prompt, capped at
    # `prospecting_feedback_examples` per verdict.
    prospecting_max_iterations: int = 2
    prospecting_min_quality_prospects: int = 5
    prospecting_places_enabled: bool = True
    prospecting_feedback_examples: int = 6
    # Send Serper's `location` param (derived from the city named in each query)
    # so the SERP is geo-biased to the tenant's cities instead of the whole
    # country — `gl`/`hl` only pin the country, which lets national directory
    # pages outrank local ones. A located search that comes back empty (Google
    # doesn't know the municipality) is retried nationally, so this can only add
    # results, never remove them. Set false to fall back to country-only search.
    prospecting_geo_targeting_enabled: bool = True
    # ── Enrichment (sherlock) ────────────────────────────────────────────────
    # One-time website enrichment of a single claimed contact, triggered by the
    # backend scheduler. No search involved — the URL comes from the contact.
    #
    # `sherlock_max_pages` bounds the same-site crawl (landing page + the
    # contact/about pages discovered on it). `sherlock_max_bytes` is a WIRE-byte
    # cap per page (compression is disabled so a gzip bomb cannot expand past it).
    # `sherlock_total_budget_seconds` is the wall-clock ceiling for ALL fetching
    # in one run — httpx timeouts are per-operation, so this is what actually
    # stops a slow-trickle server. `sherlock_max_iterations` bounds the in-run
    # refinement loop (>=1; 1 disables it).
    #
    # The page budget went 4 → 6 when qualification was added. Finding a phone
    # number needs the contact page; judging SIZE needs the services, locations
    # and pricing pages too, and a size estimate the model had no evidence for is
    # worse than none. The time budget moved with it so the extra pages have
    # somewhere to fit. Cost is bounded elsewhere — per-tenant daily limits and
    # the `aiCreditsBalance > 0` claim predicate — not by these numbers.
    sherlock_max_pages: int = 6
    sherlock_max_bytes: int = 512 * 1024
    sherlock_max_page_chars: int = 6000
    sherlock_fetch_timeout_seconds: float = 8.0
    sherlock_total_budget_seconds: float = 60.0
    sherlock_max_iterations: int = 2
    sherlock_feedback_examples: int = 6

    # ── Website discovery (sherlock, contacts that arrive with no website) ───
    # An aurora prospect whose extractor found no domain is unreachable by
    # enrichment, so discovery resolves one from the business name via Serper
    # before the crawl. Cost is bounded here (queries x 2 endpoints per run) on
    # top of the backend's one-claim-per-tenant-per-tick and daily limits.
    #
    # `min_confidence` is the bar the confirming LLM must clear, and `margin` is
    # how far ahead of the runner-up domain the winner must score: raising either
    # buys precision at the cost of prospects left without a website, which is
    # the trade we want — a wrong site is written onto the contact and read as
    # fact by a human.
    sherlock_discovery_enabled: bool = True
    sherlock_discovery_max_queries: int = 2
    sherlock_discovery_max_results: int = 10
    sherlock_discovery_min_confidence: float = 0.7
    sherlock_discovery_margin: float = 0.15

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
