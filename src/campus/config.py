"""Settings of the Campus agents (env prefix ``CAMPUS_``).

Campus calls Gemini as the platform, by Application Default Credentials, with
the models fixed here — never the tenant's credentials or model: Rocky, where
Campus lives, has no per-tenant credentials endpoint, and nothing is billed per
school.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class CampusSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CAMPUS_", env_file=".env", extra="ignore")

    # `gemini` calls Vertex AI; `fake` answers with fixed texts and never leaves
    # the process — tests and local gates.
    llm: str = "gemini"
    # Fixed, not taken from GOOGLE_CLOUD_PROJECT, which on a developer's machine
    # often points at another project.
    gemini_project: str = "vervux-platform-prod"
    # The newest models are only served from the global endpoint.
    gemini_location: str = "global"
    # Two tiers: a fast one for what happens live, a smarter one to design
    # questions and write the report.
    fast_model: str = "gemini-3.5-flash-lite"
    smart_model: str = "gemini-4-argon"
    tts_model: str = "gemini-3.8-flash-lite-tts"
    tts_voice: str = "Kore"
    # Past this, the narration goes out as text only.
    voice_timeout_seconds: float = 4.0

    # Where the Game Master's voice and the host's material go. With a bucket,
    # Cloud Storage; without one, a local folder (development and tests only:
    # Cloud Run instances share no disk).
    bucket: str = ""
    data_dir: str = "/tmp/campus-agents"

    # A session with no news for this long is over.
    idle_hours: float = 2.0
    # Minimum time between two replies of the Companion in the same group.
    mate_cooldown_seconds: float = 15.0
    # A design still "working" after this long died with its instance.
    design_timeout_minutes: float = 15.0
    # Sessions, their events and designs are deleted after this many days.
    retention_days: int = 30


campus_settings = CampusSettings()
