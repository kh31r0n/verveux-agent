from typing import AsyncIterator
import asyncio
import json

from .base import ChatProvider
from langgraph.types import RunnableConfig
from ..config import settings


def resolve_vertex_credentials(config: RunnableConfig) -> dict:
    creds = (config.get("configurable") or {}).get("vertex_credentials")
    if creds:
        return creds if isinstance(creds, dict) else json.loads(creds)
    json_str = settings.vertex_service_account_json
    if not json_str:
        raise ValueError(
            "Vertex AI credentials not found. Set VERTEX_SERVICE_ACCOUNT_JSON in .env "
            "or pass it via configurable."
        )
    return json.loads(json_str)


def resolve_vertex_project_id(config: RunnableConfig) -> str:
    project_id = (config.get("configurable") or {}).get("vertex_project_id") or settings.vertex_project_id
    if not project_id:
        raise ValueError("Vertex AI project id not found.")
    return project_id


def resolve_vertex_location(config: RunnableConfig) -> str:
    location = (config.get("configurable") or {}).get("vertex_location") or settings.vertex_location
    if not location:
        raise ValueError("Vertex AI location not found.")
    return location


def _has_api_key(creds: dict) -> str | None:
    """Return the API key if credentials contain one, else None."""
    return creds.get("apiKey") or creds.get("api_key") or None


class VertexProvider(ChatProvider):
    def __init__(self, config: RunnableConfig):
        self.credentials_dict = resolve_vertex_credentials(config)
        self.project_id = resolve_vertex_project_id(config)
        self.location = resolve_vertex_location(config)

        api_key = _has_api_key(self.credentials_dict)
        if api_key:
            # --- API key mode: use google-generativeai SDK (Gemini API) ---
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._use_genai = True
        else:
            # --- Service account / authorized_user mode: use vertexai SDK ---
            import vertexai
            from google.oauth2 import service_account

            cred_type = self.credentials_dict.get("type")
            if cred_type == "service_account":
                credentials = service_account.Credentials.from_service_account_info(
                    self.credentials_dict,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
            elif cred_type == "authorized_user":
                from google.oauth2.credentials import Credentials as OAuth2Credentials
                credentials = OAuth2Credentials.from_authorized_user_info(
                    self.credentials_dict,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
            else:
                raise ValueError(f"Unsupported credentials type: {cred_type}")

            vertexai.init(
                project=self.project_id,
                location=self.location,
                credentials=credentials,
            )
            self._use_genai = False

    def _build_contents(self, messages: list[dict]) -> tuple[list, str | None]:
        """Convert OpenAI-style messages to Gemini content format."""
        contents = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_instruction = content
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})
            else:
                contents.append({"role": "user", "parts": [{"text": content}]})

        return contents, system_instruction

    async def stream_chat(
        self,
        messages: list[dict],
        model: str,
        **kwargs,
    ) -> AsyncIterator[str]:
        contents, system_instruction = self._build_contents(messages)
        loop = asyncio.get_event_loop()

        if self._use_genai:
            # --- google-generativeai SDK (API key) ---
            import google.generativeai as genai
            gemini_model = genai.GenerativeModel(
                model or "gemini-2.0-flash",
                system_instruction=system_instruction,
            )
            responses = await loop.run_in_executor(
                None,
                lambda: list(gemini_model.generate_content(contents, stream=True)),
            )
        else:
            # --- vertexai SDK (service account) ---
            from vertexai.generative_models import GenerativeModel
            gemini_model = GenerativeModel(
                model or "gemini-2.0-flash",
                system_instruction=system_instruction,
            )
            responses = await loop.run_in_executor(
                None,
                lambda: list(gemini_model.generate_content(contents, stream=True)),
            )

        for response in responses:
            if response.text:
                yield response.text
