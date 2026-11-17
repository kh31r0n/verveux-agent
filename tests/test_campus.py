"""The Campus agents end to end (moved from vervux-campus/services/agents),
with the fixed-answer model: what the Game Core tells comes in over HTTP and
the commands go out over SSE. The store is in memory and the files in a
temporary folder; ``tests/test_campus_store_pg.py`` checks the Postgres store.
"""

import asyncio
import io
import json
import zipfile
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from src.campus.api import router
from src.campus.design import check, read_office, validated
from src.campus.files import LocalFiles
from src.campus.live import build_report
from src.campus.session import NARRATION_DEMO, Campus, calls_mate, configure
from src.campus.store import MemoryStore

KEY = {"X-System-Key": "dev-webhook-secret"}


@pytest.fixture
async def api(tmp_path) -> AsyncIterator[httpx.AsyncClient]:
    campus = Campus(MemoryStore(), LocalFiles(tmp_path))
    configure(campus)
    app = FastAPI()
    app.include_router(router)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://agent/campus", headers=KEY
    ) as client:
        yield client
    await campus.drain()


def parse_sse(body: str) -> list[dict]:
    events = []
    for block in body.replace("\r\n", "\n").strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.split("\n") if ": " in line)
        if "event" in fields:
            raw = fields.get("id")
            events.append(
                {
                    "id": int(raw.rpartition(":")[2]) if raw else None,
                    "raw_id": raw,
                    "event": fields["event"],
                    "data": json.loads(fields["data"]),
                }
            )
    return events


async def read_stream(api: httpx.AsyncClient, path: str, headers: dict | None = None) -> list[dict]:
    """Reads a stream until it ends."""
    response = await api.get(path, headers=headers, timeout=30)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    return parse_sse(response.text)


# --- auth ---------------------------------------------------------------------


async def test_every_campus_route_needs_the_service_credential(api: httpx.AsyncClient) -> None:
    anonymous = httpx.AsyncClient(transport=api._transport, base_url="http://agent/campus")
    async with anonymous:
        assert (await anonymous.post("/sessions/ABC123/feed", json={"feed": "x"})).status_code == 401
        assert (await anonymous.get("/sessions/ABC123/stream")).status_code == 401
        assert (await anonymous.get("/designs/" + "0" * 32)).status_code == 401


# --- the agents of a session --------------------------------------------------

AI = {"game_master": True, "voice": True, "companion": "guide", "tone": "fun"}
TEAMS = [
    {"team_id": "g1", "members": 3, "score": 120, "position": 4},
    {"team_id": "g2", "members": 3, "score": 80, "position": 7},
]


def started(mode: str) -> dict[str, Any]:
    return {
        "feed": "started",
        "lang": "es",
        "mode": mode,
        "ai": AI,
        "duration_min": 60,
        "topics": ["Ciencias"],
        "teams": TEAMS,
    }


def moment(number: int, line: dict[str, Any]) -> dict[str, Any]:
    return {
        "feed": "moment",
        "moment": number,
        "line": line,
        "phase": "roll",
        "item": {"kind": "round", "number": 1, "of": 5},
        "activity": None,
        "teams": TEAMS,
        "outcomes": [],
    }


def attempt(topic: str, correct: bool, **more: Any) -> dict[str, Any]:
    return {
        "topic": topic,
        "prompt": f"Pregunta de {topic}",
        "kind": "multiple_choice",
        "correct": correct,
        "text": None,
        "reference": None,
        "criteria": [],
        "score": None,
        "feedback": None,
        "confidence": None,
        **more,
    }


FINISHED = {
    "feed": "finished",
    "students": [
        {
            "player_id": "p1",
            "team_id": "g1",
            "attempts": [
                attempt("Ciencias", True),
                attempt("Ciencias", False),
                # An open answer the Evaluator did not grade live.
                attempt("Historia", False, kind="open", text="Porque sí", reference="Por la lluvia"),
            ],
        }
    ],
}


async def play(api: httpx.AsyncClient, session: str, feeds: list[dict[str, Any]]) -> list[dict]:
    for feed in [*feeds, FINISHED]:
        response = await api.post(f"/sessions/{session}/feed", json=feed)
        assert response.status_code == 202
    events = await read_stream(api, f"/sessions/{session}/stream")
    assert events[-1]["event"] == "end"
    return events[:-1]


def commands(events: list[dict], tool: str) -> list[dict[str, Any]]:
    return [
        e["data"]["command"]
        for e in events
        if e["event"] == "command" and e["data"]["command"]["tool"] == tool
    ]


async def test_an_individual_game_gets_narration_voice_evaluation_and_a_report(
    api: httpx.AsyncClient,
) -> None:
    concept = {"question_id": "q7", "topic": "Ciencias", "concept": "Júpiter", "source": "…"}
    events = await play(
        api,
        "SOLO01",
        [
            started("individual"),
            moment(1, {"line": "round_start", "round": 1, "rounds": 5}),
            {"feed": "concepts", "concepts": [concept]},
            {"feed": "guess", "concept": concept, "player_id": "p1", "attempt": 2, "text": "Júpites"},
            {
                "feed": "open_answers",
                "question_id": "q9",
                "topic": "Historia",
                "prompt": "¿Por qué?",
                "reference": "Por la lluvia",
                "criteria": ["Menciona la lluvia"],
                "answers": [
                    {"player_id": "p1", "text": "Por la lluvia de la zona"},
                    {"player_id": "p2", "text": "No sé"},
                ],
            },
        ],
    )

    assert commands(events, "narrate") == [{"tool": "narrate", "moment": 1, "text": "[GM] round_start"}]
    assert commands(events, "team_note") == []
    [voice] = [e["data"] for e in events if e["event"] == "gm.voice"]
    assert (voice["moment"], voice["file"]) == (1, "1.wav")
    audio = await api.get("/sessions/SOLO01/audio/1.wav")
    assert audio.status_code == 200 and audio.content[:4] == b"RIFF"
    assert (await api.get("/sessions/SOLO01/audio/9.wav")).status_code == 404

    assert commands(events, "describe")[0]["question_id"] == "q7"
    guess = commands(events, "guess_feedback")[0]
    assert (guess["player_id"], guess["attempt"], guess["correct"]) == ("p1", 2, True)
    results = commands(events, "eval_results")[0]["results"]
    assert [r["player_id"] for r in results] == ["p1", "p2"]
    assert results[0]["score"] > results[1]["score"]

    keys = [e["data"]["key"] for e in events]
    assert len(keys) == len(set(keys))

    [report] = commands(events, "report")
    [student] = report["report"]["students"]
    assert student["open"][0]["score"] is not None
    assert {t["topic"]: t["answered"] for t in student["topics"]} == {"Ciencias": 2, "Historia": 1}
    assert student["summary"].startswith("Nota global")
    # The report is the last thing on the stream.
    assert events[-1]["data"]["command"]["tool"] == "report"


async def test_in_groups_the_game_master_and_the_teammate_write_in_each_room(
    api: httpx.AsyncClient,
) -> None:
    def chat(text: str, team: str = "g1") -> dict[str, Any]:
        return {"feed": "chat", "team_id": team, "player_id": "p3", "text": text, "context": "…"}

    events = await play(
        api,
        "GRUPO1",
        [
            started("groups"),
            moment(1, {"line": "activity_intro", "kind": "quiz", "lightning": False}),
            moment(2, {"line": "round_end", "leaders": ["g1"]}),
            chat("yo creo que es la b"),
            chat("¿alguien sabe?"),
            # Right after, in the same group: the Companion does not answer twice.
            chat("ia, ayúdanos"),
            chat("@ia danos una pista", team="g2"),
        ],
    )

    assert sorted(c["moment"] for c in commands(events, "narrate")) == [1, 2]
    notes = commands(events, "team_note")
    assert sorted(n["team_id"] for n in notes) == ["g1", "g2"]
    assert all(n["moment"] == 2 for n in notes)
    assert not [e for e in events if e["event"] == "gm.voice"]
    mate = commands(events, "mate_say")
    assert sorted(m["team_id"] for m in mate) == ["g1", "g1", "g2", "g2"]


async def test_a_new_feed_after_the_end_starts_a_new_epoch(api: httpx.AsyncClient) -> None:
    first = await play(api, "OTRA01", [started("individual")])
    again = await play(api, "OTRA01", [started("individual")])
    epoch = lambda events: events[0]["raw_id"].partition(":")[0]  # noqa: E731
    assert epoch(first) != epoch(again)
    # The new epoch starts from offset 0 with only its own events.
    assert [e["id"] for e in again] == list(range(len(again)))


def test_the_teammate_answers_calls_and_questions() -> None:
    for text in ["¿alguien sabe?", "IA, ayúdanos", "oye @ia", "compañero, una pista", "what is it?"]:
        assert calls_mate(text), text
    for text in ["yo creo que es la b", "había una vez", "vamos equipo"]:
        assert not calls_mate(text), text


def test_grades_count_only_each_students_own_answers() -> None:
    report = build_report(
        [
            {
                "player_id": "p1",
                "team_id": "g1",
                "attempts": [
                    attempt("Ciencias", True),
                    attempt("Ciencias", False),
                    attempt("Historia", True),
                    attempt("Historia", True, kind="open", text="…", score=40, confidence=30),
                    attempt("Historia", False, kind="open", text="otra"),
                ],
            },
            {"player_id": "p2", "team_id": "g1", "attempts": []},
        ]
    )
    first, second = report["students"]
    assert first["grade"] == round((100 + 0 + 100 + 40) / 4)
    assert first["topics"] == [
        {"topic": "Ciencias", "grade": 50, "answered": 2},
        {"topic": "Historia", "grade": 70, "answered": 2},
    ]
    assert first["missed"] == ["Pregunta de Ciencias", "Pregunta de Historia"]
    assert [(o["score"], o["low_confidence"]) for o in first["open"]] == [(40, True), (None, False)]
    assert (second["grade"], second["topics"]) == (0, [])


# --- phase 0 narration demo ----------------------------------------------------

DEMO = "/sessions/ABC123/narration-demo/stream"


async def test_the_narration_arrives_in_order_and_ends(api: httpx.AsyncClient) -> None:
    assert (await api.post("/sessions/ABC123/narration-demo")).status_code == 202
    events = await read_stream(api, DEMO)
    assert [e["event"] for e in events] == (
        ["gm.narration.delta"] * len(NARRATION_DEMO) + ["gm.narration.done", "end"]
    )
    narration = events[:-1]
    assert [e["id"] for e in narration] == list(range(len(narration)))
    assert narration[-1]["data"]["text"] == "".join(NARRATION_DEMO)


async def test_a_reconnection_only_receives_what_it_missed(api: httpx.AsyncClient) -> None:
    await api.post("/sessions/ABC123/narration-demo")
    everything = await read_stream(api, DEMO)
    resumed = await read_stream(api, DEMO, headers={"Last-Event-ID": everything[3]["raw_id"]})
    assert resumed == [e for e in everything if e["id"] is None or e["id"] > 3]
    other = await read_stream(api, DEMO, headers={"Last-Event-ID": "otra-ejecucion:3"})
    assert other == everything


async def test_a_session_without_a_flow_has_no_stream(api: httpx.AsyncClient) -> None:
    assert (await api.get("/sessions/NOFLOW1/stream")).status_code == 404
    assert (await api.post("/sessions/abc/narration-demo")).status_code == 422


# --- the Designer ---------------------------------------------------------------

FRAGMENTS = {"f1": {"id": "f1", "file": "a.pdf", "location": "página 1", "text": "El agua hierve a 100 grados."}}


def question(**changes: Any) -> dict[str, Any]:
    return {
        "kind": "verdadero_falso",
        "prompt": "¿El agua hierve a 100 grados?",
        "answer": "verdadero",
        "options": [],
        "fragment": "f1",
        "quote": "el agua hierve a 100",
        **changes,
    }


def test_a_question_survives_only_with_a_citation_found_in_its_fragment() -> None:
    assert check(question(), FRAGMENTS) is None
    assert check(question(quote="El AGUA hierve, a 100"), FRAGMENTS) is None
    assert "no existe" in check(question(fragment="f9"), FRAGMENTS)
    assert "no aparece" in check(question(quote="la fotosíntesis ocurre en"), FRAGMENTS)
    assert "no aparece" in check(question(quote="agua"), FRAGMENTS)


def test_each_kind_of_question_must_be_well_formed() -> None:
    multiple = {"kind": "opcion_multiple", "answer": "100", "options": ["50", "100", "200"]}
    assert check(question(**multiple), FRAGMENTS) is None
    assert check(question(**{**multiple, "answer": "150"}), FRAGMENTS) is not None
    assert check(question(answer="quizás"), FRAGMENTS) is not None
    assert check(question(kind="ordenar", options=["a", "b"]), FRAGMENTS) is not None
    assert check(question(kind="abierta", answer=""), FRAGMENTS) is not None
    assert check(question(kind="abierta", answer="Hierve a 100 grados"), FRAGMENTS) is None


def test_validation_keeps_the_citation_and_drops_repeats() -> None:
    result = validated(
        {
            "fragments": list(FRAGMENTS.values()),
            "max_questions": 5,
            "candidates": [question(), question(), question(prompt="Otra", quote="inventada del todo")],
        }
    )
    assert result["discarded"] == 2
    [kept] = result["questions"]
    assert kept["citation"] == {"file": "a.pdf", "location": "página 1", "quote": "el agua hierve a 100"}


def test_office_files_are_read_without_converting_them() -> None:
    def archive(files: dict[str, str]) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as out:
            for name, text in files.items():
                out.writestr(name, text)
        return buffer.getvalue()

    slides = archive(
        {
            "ppt/slides/slide2.xml": "<a:p><a:t>La c&#233;lula</a:t></a:p>",
            "ppt/slides/slide1.xml": "<a:p><a:t>Introducción</a:t></a:p>",
        }
    )
    assert read_office("clase.pptx", slides) == [
        {"location": "diapositiva 1", "text": "Introducción"},
        {"location": "diapositiva 2", "text": "La célula"},
    ]


async def wait_for(api: httpx.AsyncClient, design_id: str, status: str) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for _ in range(100):
        state = (await api.get(f"/designs/{design_id}")).json()
        if state["status"] == status:
            return state
        await asyncio.sleep(0.05)
    raise AssertionError(f"the design did not reach {status}: {state}")


async def test_the_designer_proposes_and_nothing_enters_without_approval(api: httpx.AsyncClient) -> None:
    files = [
        ("files", ("clase.pdf", b"%PDF-1.4 contenido", "application/pdf")),
        ("files", ("notas.txt", b"Los planetas giran alrededor del Sol.", "text/plain")),
        ("files", ("programa.exe", b"MZ", "application/octet-stream")),
    ]
    started = await api.post("/designs", files=files, data={"topic": "Ciencias", "max_questions": "4"})
    assert started.status_code == 202
    design_id = started.json()["design_id"]

    proposal = await wait_for(api, design_id, "waiting_approval")
    statuses = {f["name"][3:]: f["status"] for f in proposal["files"]}
    assert statuses == {"clase.pdf": "processed", "notas.txt": "processed", "programa.exe": "rejected"}
    questions = proposal["questions"]
    assert len(questions) >= 2 and proposal["discarded"] >= 1
    assert all(q["citation"]["quote"] and q["citation"]["location"] for q in questions)
    assert all(q["prompt"] != "Inventada" for q in questions)

    approved = await api.post(f"/designs/{design_id}/approval", json={"approved": [questions[0]["id"]]})
    assert approved.status_code == 200
    assert (await wait_for(api, design_id, "done"))["status"] == "done"


async def test_a_design_needs_files_and_an_existing_id(api: httpx.AsyncClient) -> None:
    assert (await api.post("/designs", data={"topic": "x"})).status_code == 422
    assert (await api.get("/designs/" + "0" * 32)).status_code == 404
    assert (await api.post("/designs/" + "0" * 32 + "/approval", json={"approved": []})).status_code == 404
    assert (await api.get("/designs/no-es-un-id")).status_code == 422


async def test_the_game_master_names_the_leader_and_the_streak_by_id_and_says_their_alias(
    api: httpx.AsyncClient, monkeypatch
) -> None:
    from src.campus.llm import FakeLlm

    spoken: list[str] = []
    original = FakeLlm.speech

    async def record(self, text: str, lang: str) -> bytes:
        spoken.append(text)
        return await original(self, text, lang)

    monkeypatch.setattr(FakeLlm, "speech", record)
    standings = [
        {"player_id": "p2", "alias": "Ana", "score": 420, "streak": 4},
        {"player_id": "p1", "alias": "Luis", "score": 300, "streak": 0},
    ]
    events = await play(
        api,
        "LIDER1",
        [
            started("individual"),
            {**moment(3, {"line": "round_end", "leaders": ["t2"]}), "standings": standings},
        ],
    )
    [narration] = commands(events, "narrate")
    # Lo escrito nombra por identificador: la pantalla pone el nombre.
    assert narration["text"] == "[GM] round_end · primero {p2} · racha {p2}"
    # Lo hablado usa el alias, que nunca viaja por el stream.
    assert spoken == ["Game Master: round_end, primero Ana, Ana en racha"]
    assert "Ana" not in json.dumps([e["data"] for e in events], ensure_ascii=False)


def test_an_alias_written_by_the_model_goes_back_to_its_id() -> None:
    from src.campus.live import _unnamed

    standings = [{"player_id": "p2", "alias": "Ana María", "score": 1, "streak": 3}]
    assert _unnamed("¡Bravo, Ana María! Anabel no.", standings) == "¡Bravo, {p2}! Anabel no."
