from typing import AsyncIterator
import google.auth
from google.cloud import aiplatform
import google.auth.transport.requests
from .base import ChatProvider
from langgraph.types import RunnableConfig
from ..config import settings
import json

def resolve_vertex_credentials(config: RunnableConfig) -> dict:
    """Return the Vertex AI credentials."""
    creds = (config.get("configurable") or {}).get("vertex_credentials")
    if creds:
        return creds
    
    json_str = settings.vertex_service_account_json
    if not json_str:
        raise ValueError(
            "Vertex AI credentials not found. Set VERTEX_SERVICE_ACCOUNT_JSON in .env or pass it via configurable."
        )
    return json.loads(json_str)

def resolve_vertex_project_id(config: RunnableConfig) -> str:
    """Return the Vertex AI project id."""
    project_id = (config.get("configurable") or {}).get("vertex_project_id") or settings.vertex_project_id
    if not project_id:
        raise ValueError(
            "Vertex AI project id not found. Set VERTEX_PROJECT_ID in .env or pass it via configurable."
        )
    return project_id

def resolve_vertex_location(config: RunnableConfig) -> str:
    """Return the Vertex AI location."""
    location = (config.get("configurable") or {}).get("vertex_location") or settings.vertex_location
    if not location:
        raise ValueError(
            "Vertex AI location not found. Set VERTEX_LOCATION in .env or pass it via configurable."
        )
    return location

class VertexProvider(ChatProvider):
    def __init__(self, config: RunnableConfig):
        self.credentials = resolve_vertex_credentials(config)
        self.project_id = resolve_vertex_project_id(config)
        self.location = resolve_vertex_location(config)

        vertexai.init(
            project=self.project_id,
            location=self.location,
            credentials=google.auth.credentials.Credentials.from_service_account_info(
                self.credentials
            ),
        )

from langgraph.types import RunnableConfig
import vertexai
from vertexai.generative_models import GenerativeModel

class VertexProvider(ChatProvider):
    def __init__(self, config: RunnableConfig):
        self.credentials = resolve_vertex_credentials(config)
        self.project_id = resolve_vertex_project_id(config)
        self.location = resolve_vertex_location(config)

        vertexai.init(
            project=self.project_id,
            location=self.location,
            credentials=google.auth.credentials.Credentials.from_service_account_info(
                self.credentials
            ),
        )

    async def stream_chat(
        self,
        messages: list[dict],
        model: str,
        **kwargs,
    ) -> AsyncIterator[str]:
        model = GenerativeModel(model)
        responses = model.generate_content(messages, stream=True, **kwargs)
        for response in responses:
            yield response.text
