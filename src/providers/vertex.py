from typing import AsyncIterator
import asyncio
import json

import google.auth.credentials
import vertexai
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account

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


class VertexProvider(ChatProvider):
    def __init__(self, config: RunnableConfig):
        self.credentials_dict = resolve_vertex_credentials(config)
        self.project_id = resolve_vertex_project_id(config)
        self.location = resolve_vertex_location(config)

        credentials = service_account.Credentials.from_service_account_info(
            self.credentials_dict,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

        vertexai.init(
            project=self.project_id,
            location=self.location,
            credentials=credentials,
        )

    async def stream_chat(
        self,
        messages: list[dict],
        model: str,
        **kwargs,
    ) -> AsyncIterator[str]:
        # Convert OpenAI-style messages to Gemini content format
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

        gemini_model = GenerativeModel(
            model,
            system_instruction=system_instruction,
        )

        # Vertex SDK is sync; run in thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        responses = await loop.run_in_executor(
            None,
            lambda: list(gemini_model.generate_content(contents, stream=True)),
        )

        for response in responses:
            if response.text:
                yield response.text
