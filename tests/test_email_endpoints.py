"""/email/* endpoints: auth, versioning (required, email-only code names), error mapping."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from src.config import settings
from src.graphs.clara_graph import build_clara_graph
from src.graphs.registry import UnknownCodeNameError
from src.main import app
from src.services.gmail import GmailScopeError, GmailUnavailable, MissingMessageIdError

HEADERS = {"x-agent-key": settings.webhook_api_key}
SYNC = {"tenant_id": "t", "mailbox_id": "mb", "agent_code_name": "clara", "sync_claim_token": "c1"}
DRAFT = {
    "tenant_id": "t",
    "mailbox_id": "mb",
    "agent_code_name": "clara",
    "approval_id": "ap",
    "gmail_thread_id": "th",
    "reply_to_gmail_message_id": "m1",
    "final_body": "Gracias.",
}


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def email_graph():
    graph = build_clara_graph(MemorySaver())
    with patch("src.main.get_or_compile_graph", AsyncMock(return_value=graph)):
        yield graph


def test_sync_requires_the_agent_key(client):
    assert client.post("/email/sync", json=SYNC).status_code == 401


def test_sync_requires_an_explicit_code_name(client, email_graph):
    body = {k: v for k, v in SYNC.items() if k != "agent_code_name"}
    assert client.post("/email/sync", json=body, headers=HEADERS).status_code == 422


def test_sync_rejects_unknown_code_names(client):
    with patch("src.main.get_or_compile_graph", AsyncMock(side_effect=UnknownCodeNameError("x", []))):
        assert client.post("/email/sync", json=SYNC, headers=HEADERS).status_code == 400


def test_sync_rejects_code_names_of_non_email_graphs(client):
    chat_graph = type("Graph", (), {"name": "LangGraph"})()
    with patch("src.main.get_or_compile_graph", AsyncMock(return_value=chat_graph)):
        response = client.post("/email/sync", json={**SYNC, "agent_code_name": "helena"}, headers=HEADERS)
    assert response.status_code == 400
    assert "not an email agent" in response.text


def test_sync_is_accepted_and_runs_in_background(client, email_graph):
    with patch("src.main.clara_runner.sync_mailbox", AsyncMock()) as run:
        response = client.post("/email/sync", json=SYNC, headers=HEADERS)
    assert response.status_code == 202
    assert run.await_args.args[0].sync_claim_token == "c1"


def test_follow_up_is_accepted(client, email_graph):
    body = {**SYNC, "email_thread_id": "et", "gmail_thread_id": "th", "follow_up_number": 1}
    body.pop("sync_claim_token")
    with patch("src.main.clara_runner.generate_follow_up", AsyncMock()):
        assert client.post("/email/follow-up", json=body, headers=HEADERS).status_code == 202


def test_draft_returns_the_created_draft(client, email_graph):
    created = {"gmail_draft_id": "d1", "gmail_draft_message_id": "dm", "already_existed": False}
    with patch("src.main.clara_runner.create_approved_draft", AsyncMock(return_value=created)):
        response = client.post("/email/drafts", json=DRAFT, headers=HEADERS)
    assert response.status_code == 200
    assert response.json() == created


@pytest.mark.parametrize(
    "error, status, kind",
    [
        (MissingMessageIdError("no id"), 422, "missing_message_id"),
        (GmailScopeError("scopes"), 409, "gmail_scope"),
        (GmailUnavailable("down"), 503, "gmail_unavailable"),
    ],
)
def test_draft_refusals_are_mapped(client, email_graph, error, status, kind):
    with patch("src.main.clara_runner.create_approved_draft", AsyncMock(side_effect=error)):
        response = client.post("/email/drafts", json=DRAFT, headers=HEADERS)
    assert response.status_code == status
    assert response.json()["detail"]["kind"] == kind


def test_revoked_grant_is_reported_as_reauth(client, email_graph):
    revoked = httpx.HTTPStatusError(
        "409", request=httpx.Request("GET", "http://b"), response=httpx.Response(409, text="gmail_auth_revoked")
    )
    with patch("src.main.clara_runner.create_approved_draft", AsyncMock(side_effect=revoked)):
        response = client.post("/email/drafts", json=DRAFT, headers=HEADERS)
    assert response.status_code == 409
    assert response.json()["detail"]["kind"] == "gmail_auth"


def test_draft_rejects_an_empty_body(client, email_graph):
    assert client.post("/email/drafts", json={**DRAFT, "final_body": ""}, headers=HEADERS).status_code == 422
