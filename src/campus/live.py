"""Los agentes que actúan durante la partida: Game Master, Evaluador,
Compañero y Calificador. Cada uno es una función que recibe la novedad y
publica en el stream de la partida (``emit``); ``session.py`` les pone plazo y
reintentos.

Reciben lo que el Game Core cuenta de la partida, con identificadores
seudónimos y nunca alias, y proponen comandos. La única excepción son los
alias de `standings` (quien va primero o está en racha), solo para la frase
hablada del Game Master: en lo escrito se cambian por su identificador.

No escriben estado de juego: cada comando sale por el stream, el gateway lo
lleva al Game Core y este lo valida antes de aplicarlo.
"""

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .config import campus_settings as config
from .files import Files
from .llm import get_llm, language

logger = logging.getLogger(__name__)

GAME_MASTER = "game_master"
EVALUATOR = "evaluator"
TEAMMATE = "teammate"
GRADER = "grader"
AWARDER = "awarder"

# Calificación desde la que una respuesta abierta cuenta como acierto.
PASSING = 60
# Con menos confianza que esta, la nota queda marcada para el host.
LOW_CONFIDENCE = 60
# Frases tras las que el Game Master escribe además en la sala de cada grupo.
ROOM_LINES = {"round_end", "coop_end", "game_over"}


Emit = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class LiveState:
    session_id: str
    # Lo que se supo de la partida al empezar: idioma, modo, IA y equipos.
    session: dict[str, Any]
    # Lo que el Game Core acaba de contar.
    feed: dict[str, Any]
    # Publica un evento en el stream de la partida (``{"event": …, …}``).
    emit: Emit
    files: Files
    # Lo que el agente cuenta de su trabajo para los registros
    # (``campus_agent_done``): por ejemplo, qué pasó con la voz.
    outcome: dict[str, Any] = field(default_factory=dict)


def _command(key: str, command: dict[str, Any]) -> dict[str, Any]:
    """Un comando para el Game Core. ``key`` lo identifica: si el agente se
    reintenta y lo publica dos veces, el Game Core lo aplica una."""
    return {"event": "command", "key": key, "command": command}


def _str(properties: dict[str, Any], *required: str) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(required)}


TEXT = {"type": "string"}
ROOMS = {
    "type": "array",
    "items": _str({"team_id": TEXT, "text": TEXT}, "team_id", "text"),
}

# --- Game Master -----------------------------------------------------------

TONES = {"neutral": "neutro y claro", "fun": "divertido y cercano", "formal": "formal y sobrio"}

# El personaje del Game Master. Con cualquiera, nunca se burla de un estudiante.
STYLES = {
    "family": "Narras con calidez, para todas las edades.",
    "sportscaster": "Narras como un comentarista deportivo: emoción, ritmo y jugadas destacadas.",
    "playful": "Narras con humor: puedes bromear sobre una tirada, una casilla o una mala racha, "
    "nunca sobre una persona, su nombre ni sus respuestas.",
}

BOARD_RULES = """
También diriges el tablero para que la partida siga reñida. `board.ahead` trae, del equipo que
va primero al último, las casillas que tiene delante (`node`, `kind`). Puedes cambiar hasta
`board.budget` casillas a `question`, `bonus` o `trap`: premios delante de quien va atrás,
trampas delante de quien va muy adelante. Si la partida va pareja, no cambies nada. Escribe los
cambios en `board` y no los anuncies en `text`."""

GM_SYSTEM = """Eres el Game Master de un juego de tablero educativo que se juega en clase.
Narras lo que acaba de pasar. No decides nada del juego: ni el dado, ni los puntos, ni las
respuestas. Nunca reveles ni insinúes la respuesta de una pregunta.

Recibes en `line` lo que acaba de pasar:
- round_start: empieza la ronda `round` de `rounds`; todos tiran el dado. En la ronda 1
  empieza la partida: da la bienvenida.
- last_round: empieza la última ronda.
- squares: las fichas ya avanzaron y todos responden la misma pregunta; la casilla de cada
  equipo decide cuánto gana o pierde.
- activity_intro: empieza una actividad; `activity` dice cómo se juega: cuéntalo con tus
  palabras, sin copiarlo. Con `lightning`, es una ronda relámpago extra, porque la partida
  va adelantada.
- round_end: terminó la ronda; `leaders` son los equipos que van primero y `focus` dice de
  qué hablar esta vez, para no repetir siempre quién va primero:
  · `top`: el podio, `teams` del primero al tercero.
  · `streak`: `team_id` lleva `streak` aciertos seguidos.
  · `climb`: `team_id` subió del puesto `from` al `to` en esta ronda.
  · `fastest`: `team_id` dio el acierto más rápido de la ronda, en `elapsed_ms` milisegundos
    (dilo en segundos con dos decimales).
  Habla de ese dato. Si `focus` no es `top`, no digas quién va primero.
- coop_end: terminó una misión cooperativa; `success` dice si la lograron.
- game_over: terminó la partida; `winners` son los ganadores.
`teams` trae los puntos y la casilla de cada equipo; `outcomes`, lo que le pasó a cada uno
en su casilla. `standings` trae a los jugadores que van primero, de más a menos puntos, y a
los que están en racha (`streak`: aciertos seguidos, desde 3). Cuenta solo lo que está en los
datos: no inventes reglas, premios ni cifras.
{who}

Al cierre de ronda, sigue `focus`. En las demás frases puedes nombrar a quien va primero o
felicitar a quien esté en racha, pero no lo hagas en cada frase: varía. Nunca digas quién
falló ni quién va último.

{style} Nunca te burles de un estudiante.{board}

Escribe en {language}, con tono {tone}. Máximo 40 palabras por texto.
- `text`: la narración para todos. Para nombrar a un jugador o a un equipo escribe su
  identificador entre llaves, por ejemplo {{p3}}, {{t2}} o {{g3}}; la pantalla lo cambia por su
  nombre. Nunca escribas el `alias` aquí ni inventes nombres.
- `spoken`: la misma narración para leer en voz alta, sin identificadores ni llaves. A los
  jugadores de `standings` llámalos por su `alias`; a cualquier otro, por lo que hizo
  («quien va primero», «quien lleva la racha», «quien más subió», «quien respondió más
  rápido»).
- `rooms`: {rooms}"""

WHO_GROUPS = "Se juega por grupos: cada identificador de `teams` es un grupo de estudiantes."
WHO_SOLO = (
    "Se juega de forma individual: cada identificador de `teams` es una sola persona. "
    "No digas «equipo» ni «grupo»: escribe solo su identificador."
)

# Cómo se juega cada actividad, para que el Game Master no lo invente.
ACTIVITIES = {
    "quiz": "Quiz relámpago: preguntas seguidas para todos; suma más quien acierta más rápido.",
    "guess_word": "Adivina la palabra: se describe un concepto y cada quien lo escribe; "
    "hay pistas y vale más acertar en menos intentos.",
    "explain": "Explícalo con tus palabras: una pregunta abierta que cada quien responde "
    "por escrito; se califica la idea, no la ortografía.",
    "true_false": "Verdadero o falso: quien falla o no responde a tiempo queda fuera; "
    "ganan quienes sigan en pie.",
    "sequence": "Ordena la secuencia: hay que poner los elementos en su orden correcto.",
    "safe": "Caja fuerte, misión cooperativa: cada acierto revela un dígito de la "
    "combinación y nadie puede abrirla solo.",
    "boss": "Jefe final, misión cooperativa: cada acierto le quita energía al jefe y cada "
    "error se la devuelve; se gana entre todos.",
}

GM_SCHEMA = _str({"text": TEXT, "spoken": TEXT, "rooms": ROOMS}, "text", "spoken", "rooms")
SQUARE_KINDS = ("question", "bonus", "trap")
BOARD = {
    "type": "array",
    "items": _str(
        {"node": {"type": "integer"}, "kind": {"type": "string", "enum": list(SQUARE_KINDS)}},
        "node",
        "kind",
    ),
}
GM_BOARD_SCHEMA = _str(
    {"text": TEXT, "spoken": TEXT, "rooms": ROOMS, "board": BOARD}, "text", "spoken", "rooms"
)


def _board_changes(
    proposed: Any, brief: dict[str, Any], teams: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Lo que propuso el modelo, dentro de lo que el Game Core admite: casillas
    del resumen, sin fichas encima, con otro tipo y como mucho ``budget``."""
    current = {
        square["node"]: square["kind"]
        for team in brief.get("ahead", [])
        for square in team["squares"]
    }
    occupied = {team.get("position") for team in teams}
    changes: list[dict[str, Any]] = []
    for change in proposed if isinstance(proposed, list) else []:
        node, kind = change.get("node"), change.get("kind")
        if (
            node in current
            and node not in occupied
            and kind in SQUARE_KINDS
            and kind != current[node]
            and all(c["node"] != node for c in changes)
        ):
            changes.append({"node": node, "kind": kind})
    return changes[: brief.get("budget", 0)]


def _unnamed(text: str, standings: list[dict[str, Any]]) -> str:
    """Lo escrito nombra por identificador: si el modelo puso un alias, se
    cambia por `{p3}` y la pantalla vuelve a poner el nombre. Así el alias
    nunca viaja por el stream ni llega al Game Core."""
    for standing in standings:
        alias, player = standing.get("alias") or "", standing.get("player_id") or ""
        if alias and player:
            text = re.sub(rf"(?<!\w){re.escape(alias)}(?!\w)", f"{{{player}}}", text)
    return text


async def narrate(state: LiveState) -> dict[str, Any]:
    write = state.emit
    session, feed = state.session, state.feed
    ai = session.get("ai", {})
    lang = session.get("lang", "es")
    groups = session.get("mode") == "groups"
    moment = feed["moment"]
    # Según la frecuencia que eligió el host, este momento puede no narrarse;
    # al cerrar una ronda, igual se dirige el tablero.
    narrates = feed.get("narrate", True)
    brief = feed.get("board")
    if not narrates and not brief:
        state.outcome["voice"] = "silent"
        return {}
    # En partidas por grupos escribe también en la sala de cada grupo, en una
    # sola llamada para no multiplicar las solicitudes.
    rooms_for = feed["teams"] if groups and feed["line"]["line"] in ROOM_LINES else []
    rooms = (
        "una frase para la sala de cada equipo de `rooms_for`, sobre cómo le fue a ese equipo."
        if rooms_for
        else "una lista vacía."
    )
    reply = await get_llm().json(
        "gm",
        system=GM_SYSTEM.format(
            language=language(lang),
            tone=TONES.get(ai.get("tone"), TONES["fun"]),
            rooms=rooms,
            who=WHO_GROUPS if groups else WHO_SOLO,
            style=STYLES.get(ai.get("style"), STYLES["family"]),
            board=BOARD_RULES if brief else "",
        ),
        data={
            "line": feed["line"],
            "activity": ACTIVITIES.get(feed["line"].get("kind")),
            "item": feed.get("item"),
            "teams": feed["teams"],
            "outcomes": feed.get("outcomes", []),
            "standings": feed.get("standings", []),
            "rooms_for": rooms_for,
            **({"board": brief} if brief else {}),
        },
        schema=GM_BOARD_SCHEMA if brief else GM_SCHEMA,
    )
    if brief:
        changes = _board_changes(reply.get("board"), brief, feed["teams"])
        if changes:
            command = {"tool": "reshape_board", "moment": moment, "changes": changes}
            await write(_command(f"b:{moment}", command))
        state.outcome["board"] = len(changes)
    if not narrates:
        state.outcome["voice"] = "silent"
        return {}
    text = _unnamed((reply.get("text") or "").strip(), feed.get("standings", []))
    if text:
        await write(_command(f"n:{moment}", {"tool": "narrate", "moment": moment, "text": text}))
    known = {team["team_id"] for team in rooms_for}
    for room in reply.get("rooms") or []:
        if room.get("team_id") in known and (room.get("text") or "").strip():
            team = room["team_id"]
            note = {"tool": "team_note", "moment": moment, "team_id": team, "text": room["text"]}
            await write(_command(f"t:{moment}:{team}", note))

    # La voz existe solo en partidas individuales, donde todos comparten sala.
    spoken = (reply.get("spoken") or "").strip()
    if not ai.get("voice"):
        state.outcome["voice"] = "off"
    elif groups:
        state.outcome["voice"] = "groups"
    elif not spoken:
        state.outcome["voice"] = "nothing_to_say"
    else:
        try:
            audio = await asyncio.wait_for(
                get_llm().speech(spoken, lang), timeout=config.voice_timeout_seconds
            )
        except Exception as error:  # noqa: BLE001 - la voz nunca detiene la narración
            # Sin audio a tiempo, la narración queda solo en texto.
            logger.warning("Sin voz para el momento %s: %s", moment, error)
            state.outcome["voice"] = (
                "timeout" if isinstance(error, TimeoutError) else type(error).__name__
            )
            return {}
        await state.files.put(f"audio/{state.session_id}/{moment}.wav", audio, "audio/wav")
        await write(
            {"event": "gm.voice", "key": f"v:{moment}", "moment": moment, "file": f"{moment}.wav"}
        )
        state.outcome["voice"] = "generated"
    return {}


# --- Evaluador -------------------------------------------------------------

DESCRIBE_SYSTEM = """Eres el Evaluador de un juego educativo. En «Adivina la palabra» los
jugadores deben adivinar un concepto. Para cada concepto de `concepts` escribe en {language}
una descripción de una frase, de máximo 25 palabras, usando solo lo que dice `source`.
No escribas el concepto, ni palabras de su misma familia, ni su traducción."""

GUESS_SYSTEM = """Eres el Evaluador de un juego educativo. Un jugador intenta adivinar
`concept`. `attempt` es lo que escribió.
- `correct`: verdadero solo si es el mismo concepto: un sinónimo exacto, otra forma de
  escribirlo o un error de tecleo. Ante la duda, falso.
- `closeness`: «hot» si está muy cerca, «warm» si va por buen camino, «cold» si no.
- `hint`: una pista nueva en {language}, de máximo 15 palabras, tomada de `source`. No
  escribas el concepto ni palabras de su misma familia."""

EVALUATE_SYSTEM = """Eres el Evaluador de un juego educativo. Calificas respuestas abiertas de
estudiantes a `prompt`, contra la respuesta de referencia `reference` y los criterios
`criteria` del docente. Califica solo el contenido: no penalices la ortografía ni la
brevedad si la idea está. Para cada respuesta de `answers` devuelve:
- `score`: de 0 a 100. Respuestas iguales reciben la misma nota.
- `feedback`: dos o tres frases en {language}, dirigidas al estudiante de tú, que digan qué
  acertó y qué le faltó. No nombres a nadie.
- `confidence`: de 0 a 100, qué tan segura es tu nota. Baja si la respuesta es ambigua, está
  fuera de tema o no la puedes juzgar con la referencia."""

GUESS_SCHEMA = _str(
    {
        "correct": {"type": "boolean"},
        "closeness": {"type": "string", "enum": ["cold", "warm", "hot"]},
        "hint": TEXT,
    },
    "correct",
    "closeness",
    "hint",
)
DESCRIBE_SCHEMA = _str(
    {
        "descriptions": {
            "type": "array",
            "items": _str({"question_id": TEXT, "text": TEXT}, "question_id", "text"),
        }
    },
    "descriptions",
)
RESULT = _str(
    {
        "player_id": TEXT,
        "score": {"type": "integer"},
        "feedback": TEXT,
        "confidence": {"type": "integer"},
    },
    "player_id",
    "score",
    "feedback",
    "confidence",
)
EVALUATE_SCHEMA = _str({"results": {"type": "array", "items": RESULT}}, "results")


def _clamp(value: Any) -> int:
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return 0


async def grade_answers(lang: str, question: dict[str, Any]) -> list[dict[str, Any]]:
    """Califica en una sola llamada todas las respuestas a una pregunta abierta."""
    reply = await get_llm().json(
        "evaluate",
        system=EVALUATE_SYSTEM.format(language=language(lang)),
        data={
            "prompt": question["prompt"],
            "reference": question["reference"],
            "criteria": question.get("criteria", []),
            "answers": question["answers"],
        },
        schema=EVALUATE_SCHEMA,
    )
    asked = {answer["player_id"] for answer in question["answers"]}
    model = get_llm().model()
    return [
        {
            "player_id": result["player_id"],
            "score": _clamp(result.get("score")),
            "feedback": (result.get("feedback") or "").strip() or "—",
            "confidence": _clamp(result.get("confidence")),
            "model": model,
        }
        for result in reply.get("results") or []
        if result.get("player_id") in asked
    ]


async def evaluate(state: LiveState) -> dict[str, Any]:
    write = state.emit
    feed = state.feed
    lang = state.session.get("lang", "es")
    kind = feed["feed"]
    if kind == "concepts":
        reply = await get_llm().json(
            "describe",
            system=DESCRIBE_SYSTEM.format(language=language(lang)),
            data={"concepts": feed["concepts"]},
            schema=DESCRIBE_SCHEMA,
        )
        known = {concept["question_id"] for concept in feed["concepts"]}
        for description in reply.get("descriptions") or []:
            question = description.get("question_id")
            if question in known and (description.get("text") or "").strip():
                command = {"tool": "describe", "question_id": question, "text": description["text"]}
                await write(_command(f"d:{question}", command))
    elif kind == "guess":
        concept = feed["concept"]
        reply = await get_llm().json(
            "guess",
            system=GUESS_SYSTEM.format(language=language(lang)),
            data={
                "concept": concept["concept"],
                "source": concept["source"],
                "attempt": feed["text"],
                "attempt_number": feed["attempt"],
            },
            schema=GUESS_SCHEMA,
        )
        closeness = reply.get("closeness")
        command = {
            "tool": "guess_feedback",
            "question_id": concept["question_id"],
            "player_id": feed["player_id"],
            "attempt": feed["attempt"],
            "correct": bool(reply.get("correct")),
            "closeness": closeness if closeness in ("cold", "warm", "hot") else "cold",
            "hint": (reply.get("hint") or "").strip() or None,
        }
        key = f"g:{concept['question_id']}:{feed['player_id']}:{feed['attempt']}"
        await write(_command(key, command))
    elif kind == "open_answers":
        results = await grade_answers(lang, feed)
        command = {"tool": "eval_results", "question_id": feed["question_id"], "results": results}
        await write(_command(f"e:{feed['question_id']}", command))
    return {}


# --- Compañero -------------------------------------------------------------

MATE_SYSTEM = """Eres el Compañero de IA de un grupo de estudiantes que juega un juego educativo.
Hablas en el chat del grupo, en {language}, en un mensaje de máximo 30 palabras. Te presentas
como IA si aún no lo has hecho. {mode}
No conoces la respuesta correcta y nunca la das: ayudas a pensar. Si te piden la respuesta,
devuelve una pregunta que ayude a encontrarla. No pidas ni repitas datos personales.
`context` es lo que el grupo tiene delante y `message` lo que acaba de escribir alguien."""

MATE_MODES = {
    "guide": "Eres guía: das pistas y haces preguntas, sin aportar la solución.",
    "collaborator": "Eres colaborador: además de pistas, aportas tus propias ideas al grupo.",
}

NUDGE_SYSTEM = """Eres el Compañero de IA de varios grupos de estudiantes que juegan un juego
educativo. Empieza la actividad `activity`. Escribe en {language}, para el chat de cada equipo
de `teams`, un mensaje de máximo 20 palabras que lo anime a organizarse. Di que eres una IA.
No des respuestas."""

MATE_SCHEMA = _str({"reply": TEXT}, "reply")
NUDGE_SCHEMA = _str({"rooms": ROOMS}, "rooms")


async def reply(state: LiveState) -> dict[str, Any]:
    write = state.emit
    session, feed = state.session, state.feed
    lang = session.get("lang", "es")
    if feed["feed"] == "nudge":
        answer = await get_llm().json(
            "nudge",
            system=NUDGE_SYSTEM.format(language=language(lang)),
            data={"activity": feed["activity"], "teams": feed["teams"]},
            schema=NUDGE_SCHEMA,
        )
        for room in answer.get("rooms") or []:
            if room.get("team_id") in feed["teams"] and (room.get("text") or "").strip():
                team = room["team_id"]
                command = {"tool": "mate_say", "team_id": team, "text": room["text"]}
                await write(_command(f"m:{feed['moment']}:{team}", command))
        return {}
    mode = MATE_MODES.get(session.get("ai", {}).get("companion"), MATE_MODES["guide"])
    answer = await get_llm().json(
        "mate",
        system=MATE_SYSTEM.format(language=language(lang), mode=mode),
        data={"context": feed.get("context", ""), "message": feed["text"]},
        schema=MATE_SCHEMA,
    )
    text = (answer.get("reply") or "").strip()
    if text:
        command = {"tool": "mate_say", "team_id": feed["team_id"], "text": text}
        await write(_command(f"m:{feed['seq']}", command))
    return {}


# --- Calificador -----------------------------------------------------------

SUMMARY_SYSTEM = """Eres el Calificador de un juego educativo. Para cada estudiante de `students`
redacta en {language} un resumen de dos frases para su docente: sus fortalezas y sus brechas,
según sus notas por tema (`topics`) y las preguntas que falló (`missed`). Refiérete a él como
«el estudiante»; no inventes datos que no estén."""

SUMMARY_SCHEMA = _str(
    {
        "summaries": {
            "type": "array",
            "items": _str({"player_id": TEXT, "summary": TEXT}, "player_id", "summary"),
        }
    },
    "summaries",
)


def _is_open(attempt: dict[str, Any]) -> bool:
    return attempt.get("kind") == "open" and attempt.get("text") is not None


def _mark(attempt: dict[str, Any]) -> int | None:
    """Lo que vale una respuesta, de 0 a 100. ``None`` si quedó sin calificar."""
    if _is_open(attempt):
        return attempt.get("score")
    return 100 if attempt.get("correct") else 0


def build_report(students: list[dict[str, Any]]) -> dict[str, Any]:
    """Las notas, con reglas fijas: solo cuentan las respuestas propias de cada
    estudiante. Ni los puntos de juego, ni el dado, ni el resultado del grupo."""
    report = []
    for student in students:
        marks: list[int] = []
        topics: dict[str, list[int]] = {}
        missed: list[str] = []
        reviews = []
        for attempt in student["attempts"]:
            mark = _mark(attempt)
            if _is_open(attempt):
                confidence = attempt.get("confidence")
                reviews.append(
                    {
                        "topic": attempt["topic"],
                        "prompt": attempt["prompt"],
                        "answer": attempt["text"],
                        "score": mark,
                        "feedback": attempt.get("feedback"),
                        "low_confidence": confidence is not None and confidence < LOW_CONFIDENCE,
                    }
                )
            if mark is None:
                continue
            marks.append(mark)
            topics.setdefault(attempt["topic"], []).append(mark)
            if mark < PASSING and attempt["prompt"] not in missed:
                missed.append(attempt["prompt"])
        report.append(
            {
                "player_id": student["player_id"],
                "grade": round(sum(marks) / len(marks)) if marks else 0,
                "topics": [
                    {"topic": topic, "grade": round(sum(of) / len(of)), "answered": len(of)}
                    for topic, of in topics.items()
                ],
                "open": reviews,
                "missed": missed,
                "summary": "",
            }
        )
    return {"students": report}


async def grade(state: LiveState) -> dict[str, Any]:
    write = state.emit
    lang = state.session.get("lang", "es")
    students = state.feed["students"]

    # Lo que el Evaluador no alcanzó a calificar en vivo se califica ahora.
    pending: dict[str, dict[str, Any]] = {}
    for student in students:
        for attempt in student["attempts"]:
            if _is_open(attempt) and attempt.get("score") is None and attempt.get("reference"):
                question = pending.setdefault(
                    attempt["prompt"],
                    {
                        "prompt": attempt["prompt"],
                        "reference": attempt["reference"],
                        "criteria": attempt.get("criteria", []),
                        "answers": [],
                    },
                )
                question["answers"].append(
                    {"player_id": student["player_id"], "text": attempt["text"]}
                )
    for question in pending.values():
        try:
            results = {r["player_id"]: r for r in await grade_answers(lang, question)}
        except Exception as error:  # noqa: BLE001 - el informe sale aunque falle el modelo
            # Sin modelo, esas respuestas siguen pendientes para el host.
            logger.warning("No se pudieron calificar las respuestas pendientes: %s", error)
            continue
        for student in students:
            result = results.get(student["player_id"])
            for attempt in student["attempts"]:
                mine = result and attempt["prompt"] == question["prompt"] and _is_open(attempt)
                if mine and attempt.get("score") is None:
                    attempt.update(
                        score=result["score"],
                        feedback=result["feedback"],
                        confidence=result["confidence"],
                    )

    report = build_report(students)
    # El resumen es lo único que redacta el modelo. Si falla, el informe sale
    # igual, con las notas.
    for start in range(0, len(report["students"]), 10):
        batch = report["students"][start : start + 10]
        try:
            answer = await get_llm().json(
                "summaries",
                system=SUMMARY_SYSTEM.format(language=language(lang)),
                data={
                    "students": [
                        {
                            "player_id": s["player_id"],
                            "grade": s["grade"],
                            "topics": s["topics"],
                            "missed": s["missed"][:5],
                        }
                        for s in batch
                    ]
                },
                schema=SUMMARY_SCHEMA,
                smart=True,
            )
        except Exception as error:  # noqa: BLE001 - el informe sale aunque falle el modelo
            logger.warning("No se pudieron redactar los resúmenes: %s", error)
            continue
        summaries = {s.get("player_id"): s.get("summary") for s in answer.get("summaries") or []}
        for student in batch:
            student["summary"] = (summaries.get(student["player_id"]) or "").strip()
    await write(_command("report", {"tool": "report", "report": report}))
    return {}


# --- Premios del final -----------------------------------------------------

AWARDS_SYSTEM = """Eres el Game Master de un juego educativo que acaba de terminar. Pones nombre
a los premios del final: reconocimientos, no puntos. `candidates` trae cada premio con quién lo
gana (`player_id`) y el dato que lo justifica (`fact`). Para cada uno escribe en {language}:
- `title`: de 2 a 4 palabras, máximo 40 caracteres.
- `reason`: una frase de máximo 140 caracteres que diga el dato, con el identificador del
  jugador entre llaves, por ejemplo {{p3}}; la pantalla lo cambia por su nombre.
{style} Celebra lo que hicieron. Nunca te burles de nadie ni digas quién falló."""

AWARDS_SCHEMA = _str(
    {
        "awards": {
            "type": "array",
            "items": _str(
                {"player_id": TEXT, "title": TEXT, "reason": TEXT}, "player_id", "title", "reason"
            ),
        }
    },
    "awards",
)
MAX_AWARDS = 5


def award_candidates(students: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Quién gana cada premio, con datos de la partida: la racha más larga, la
    rapidez, el dominio de un tema y la mejor explicación. El modelo solo les
    pone nombre."""
    candidates: list[dict[str, Any]] = []

    def run(attempts: list[dict[str, Any]]) -> int:
        best = current = 0
        for attempt in attempts:
            current = current + 1 if attempt.get("correct") else 0
            best = max(best, current)
        return best

    streaks = sorted(((run(s["attempts"]), s["player_id"]) for s in students), reverse=True)
    if streaks and streaks[0][0] >= 3:
        candidates.append(
            {
                "award": "streak",
                "player_id": streaks[0][1],
                "fact": f"{streaks[0][0]} aciertos seguidos",
            }
        )

    fast = []
    for student in students:
        times = sorted(
            a["elapsed_ms"] for a in student["attempts"] if a.get("correct") and a.get("elapsed_ms")
        )
        if len(times) >= 3:
            fast.append((times[len(times) // 2], student["player_id"]))
    if fast:
        ms, player = min(fast)
        candidates.append(
            {
                "award": "fast",
                "player_id": player,
                "fact": f"acertó en {ms / 1000:.1f} s de mediana",
            }
        )

    topics: dict[str, list[tuple[float, int, str]]] = {}
    for student in students:
        by_topic: dict[str, list[bool]] = {}
        for attempt in student["attempts"]:
            by_topic.setdefault(attempt.get("topic", ""), []).append(bool(attempt.get("correct")))
        for topic, results in by_topic.items():
            if topic and len(results) >= 3:
                topics.setdefault(topic, []).append(
                    (sum(results) / len(results), len(results), student["player_id"])
                )
    experts = sorted(((max(rows), topic) for topic, rows in topics.items()), reverse=True)
    if experts and experts[0][0][0] >= 0.75:
        (share, _, player), topic = experts[0]
        candidates.append(
            {
                "award": "topic",
                "player_id": player,
                "fact": f"{round(share * 100)} % de aciertos en {topic}",
            }
        )

    explained = [
        (max(a["score"] for a in s["attempts"] if a.get("score") is not None), s["player_id"])
        for s in students
        if any(a.get("score") is not None for a in s["attempts"])
    ]
    if explained:
        score, player = max(explained)
        if score >= 80:
            candidates.append(
                {
                    "award": "explainer",
                    "player_id": player,
                    "fact": f"la mejor explicación, con {score} de 100",
                }
            )

    # Que no se los lleve todos la misma persona.
    given: dict[str, int] = {}
    fair = []
    for candidate in candidates:
        if given.get(candidate["player_id"], 0) < 2:
            given[candidate["player_id"]] = given.get(candidate["player_id"], 0) + 1
            fair.append(candidate)
    return fair[:MAX_AWARDS]


async def award(state: LiveState) -> dict[str, Any]:
    ai = state.session.get("ai", {})
    candidates = award_candidates(state.feed.get("students", []))
    state.outcome["awards"] = len(candidates)
    if not candidates:
        return {}
    reply = await get_llm().json(
        "awards",
        system=AWARDS_SYSTEM.format(
            language=language(state.session.get("lang", "es")),
            style=STYLES.get(ai.get("style"), STYLES["family"]),
        ),
        data={"candidates": candidates},
        schema=AWARDS_SCHEMA,
    )
    winners = {c["player_id"] for c in candidates}
    awards = [
        {
            "player_id": a["player_id"],
            "title": a["title"].strip()[:40],
            "reason": a["reason"].strip()[:140],
        }
        for a in reply.get("awards") or []
        if a.get("player_id") in winners
        and (a.get("title") or "").strip()
        and (a.get("reason") or "").strip()
    ][:MAX_AWARDS]
    if awards:
        await state.emit(_command("awards", {"tool": "awards", "awards": awards}))
    return {}


@dataclass(frozen=True)
class Agent:
    run: Callable[[LiveState], Awaitable[dict[str, Any]]]
    # Plazo de cada intento. Lo que no llega a tiempo no se espera: la
    # partida sigue con el guion.
    seconds: float
    attempts: int


AGENTS = {
    # Una narración que llega después de su momento ya no sirve: el Game Core
    # la descarta. Mejor no esperarla más.
    GAME_MASTER: Agent(narrate, seconds=10, attempts=1),
    AWARDER: Agent(award, seconds=20, attempts=1),
    EVALUATOR: Agent(evaluate, seconds=25, attempts=2),
    TEAMMATE: Agent(reply, seconds=20, attempts=1),
    GRADER: Agent(grade, seconds=300, attempts=2),
}
