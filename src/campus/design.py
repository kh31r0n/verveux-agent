"""El Diseñador: antes de la partida, lee el material del host y propone
preguntas nuevas. Cada pregunta cita el fragmento del material que la
respalda, y ninguna entra al banco sin la aprobación del host.

El grafo extrae fragmentos de cada archivo, genera preguntas y las valida con
reglas fijas: la que no cita un fragmento que exista, o cuya cita no aparece
en él, se descarta. Si quedan pocas, vuelve a generar, hasta tres veces.

Corre en segundo plano en la instancia que recibió el material (``session.py``);
el material está en el almacén de archivos (``designs/{id}/``), que llega en
``config["configurable"]["files"]``.
"""

import io
import mimetypes
import re
import unicodedata
import zipfile
from html import unescape
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from .files import Files
from .llm import Attachment, get_llm, language

DESIGNER = "designer"

MAX_FILES = 5
MAX_FILE_BYTES = 15 * 1024 * 1024
MAX_FRAGMENTS = 60
MAX_FRAGMENT_CHARS = 600
# Un párrafo más corto que esto se junta con el siguiente.
MIN_FRAGMENT_CHARS = 120
MAX_ATTEMPTS = 3
# Formatos que Gemini lee tal cual.
NATIVE = ("application/pdf", "image/", "audio/", "video/")
KINDS = ("opcion_multiple", "verdadero_falso", "ordenar", "abierta")


class DesignState(TypedDict, total=False):
    design_id: str
    lang: str
    topic: str
    max_questions: int
    # Lo que pasó con cada archivo: procesado, con advertencias o rechazado.
    files: list[dict[str, Any]]
    fragments: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    questions: list[dict[str, Any]]
    discarded: int
    attempts: int


def design_prefix(design_id: str) -> str:
    return f"designs/{design_id}"


def _plain(text: str) -> str:
    """Minúsculas, sin tildes ni signos, para comparar una cita con su fragmento."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _xml_text(xml: bytes) -> str:
    """El texto de un XML de oficina, con un salto por párrafo."""
    text = re.sub(rb"</(w:p|a:p|si)>", b"\n", xml)
    text = re.sub(rb"<[^>]+>", b"", text)
    return unescape(text.decode("utf-8", errors="ignore"))


def _chunks(text: str, location: str) -> list[dict[str, str]]:
    """Parte un texto en fragmentos por párrafos, de un tamaño manejable."""
    fragments: list[dict[str, str]] = []
    current = ""
    for paragraph in (p.strip() for p in re.split(r"\n\s*\n|\n", text)):
        if not paragraph:
            continue
        full = len(current) + len(paragraph) > MAX_FRAGMENT_CHARS
        if current and (full or len(current) >= MIN_FRAGMENT_CHARS):
            fragments.append({"location": location, "text": current})
            current = ""
        current = f"{current} {paragraph}".strip()[: MAX_FRAGMENT_CHARS * 2]
    if current:
        fragments.append({"location": location, "text": current})
    return fragments


def read_office(name: str, data: bytes) -> list[dict[str, str]]:
    """Fragmentos de un documento, una presentación o una hoja de cálculo.
    Son archivos comprimidos con XML dentro: se lee su texto sin convertirlos."""
    suffix = Path(name).suffix.lower()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if suffix == ".docx":
            return _chunks(_xml_text(archive.read("word/document.xml")), "documento")
        if suffix == ".pptx":
            slides = sorted(
                (n for n in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                key=lambda n: int(re.findall(r"\d+", n)[-1]),
            )
            fragments = []
            for number, slide in enumerate(slides, start=1):
                fragments += _chunks(_xml_text(archive.read(slide)), f"diapositiva {number}")
            return fragments
        if suffix == ".xlsx" and "xl/sharedStrings.xml" in names:
            return _chunks(_xml_text(archive.read("xl/sharedStrings.xml")), "hoja de cálculo")
    return []


EXTRACT_SYSTEM = """Eres el Diseñador de un juego educativo. Lees el material de un docente para
preparar preguntas. Divide el contenido del archivo en fragmentos que se sostengan solos: una
idea, un dato o una definición por fragmento. Para cada uno devuelve:
- `location`: dónde está, en {language}: «página 3», «diapositiva 2», «minuto 1:30», «imagen».
- `text`: lo que dice, fiel al original, en máximo 80 palabras. No añadas nada que no esté.
Como mucho {limit} fragmentos. Si el archivo no trae contenido útil, devuelve una lista vacía."""

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "fragments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"location": {"type": "string"}, "text": {"type": "string"}},
                "required": ["location", "text"],
            },
        }
    },
    "required": ["fragments"],
}


async def extract(state: DesignState, config: RunnableConfig) -> dict[str, Any]:
    """Convierte cada archivo del host en fragmentos de texto con su ubicación.
    Un archivo que no se puede leer no bloquea a los demás."""
    store: Files = config["configurable"]["files"]
    lang = state.get("lang", "es")
    files, fragments = [], []
    for object_name in await store.names(design_prefix(state["design_id"])):
        path = Path(object_name)
        name = path.name
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        status = {"name": name, "status": "processed", "note": ""}
        found: list[dict[str, str]] = []
        try:
            data = await store.get(object_name) or b""
            if len(data) > MAX_FILE_BYTES:
                raise ValueError("el archivo supera el tamaño máximo")
            if mime.startswith("text/") or path.suffix.lower() in (".md", ".csv", ".txt"):
                found = _chunks(data.decode("utf-8", errors="ignore"), "texto")
            elif path.suffix.lower() in (".docx", ".pptx", ".xlsx"):
                found = read_office(name, data)
            elif mime.startswith(NATIVE):
                reply = await get_llm().json(
                    "extract",
                    system=EXTRACT_SYSTEM.format(language=language(lang), limit=MAX_FRAGMENTS),
                    data={"file": name},
                    schema=EXTRACT_SCHEMA,
                    smart=True,
                    files=[Attachment(data, mime)],
                )
                found = [
                    {"location": f.get("location") or "", "text": (f.get("text") or "").strip()}
                    for f in reply.get("fragments") or []
                    if (f.get("text") or "").strip()
                ]
            else:
                raise ValueError("formato no admitido")
            if not found:
                status.update(status="warnings", note="no se encontró contenido útil")
        except Exception as error:  # noqa: BLE001 - un archivo ilegible no detiene a los demás
            status.update(status="rejected", note=str(error)[:200])
        for fragment in found[: MAX_FRAGMENTS - len(fragments)]:
            fragments.append(
                {
                    "id": f"f{len(fragments) + 1}",
                    "file": name,
                    "location": fragment["location"],
                    "text": fragment["text"][: MAX_FRAGMENT_CHARS * 2],
                }
            )
        files.append(status)
    return {"files": files, "fragments": fragments, "attempts": 0, "questions": []}


DESIGN_SYSTEM = """Eres el Diseñador de un juego educativo. Con los fragmentos del material del
docente (`fragments`) escribe hasta {count} preguntas nuevas en {language} sobre el tema `topic`,
de dificultad variada y de estos tipos:
- opcion_multiple: `options` con 4 opciones y `answer` igual a la opción correcta.
- verdadero_falso: `answer` es «verdadero» o «falso»; `options` vacío.
- ordenar: `options` con 3 a 5 elementos en su orden correcto; `answer` vacío.
- abierta: `answer` es la respuesta de referencia y `criteria`, 2 criterios de evaluación.
Reglas:
- Cada pregunta se responde solo con lo que dice un fragmento. `fragment` es el identificador de
  ese fragmento y `quote` una cita literal suya, de 4 a 12 palabras seguidas, copiada tal cual.
- No uses conocimiento externo ni mezcles fragmentos. No repitas preguntas de `avoid`.
- `explanation`: una frase que explique la respuesta con lo que dice el fragmento."""

DESIGN_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "kind": {"type": "string", "enum": list(KINDS)},
                    "prompt": {"type": "string"},
                    "answer": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "explanation": {"type": "string"},
                    "criteria": {"type": "array", "items": {"type": "string"}},
                    "fragment": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["kind", "prompt", "answer", "options", "fragment", "quote"],
            },
        }
    },
    "required": ["questions"],
}


async def generate(state: DesignState) -> dict[str, Any]:
    attempt = state.get("attempts", 0) + 1
    kept = state.get("questions", [])
    wanted = state.get("max_questions", 10) - len(kept)
    if not state.get("fragments") or wanted <= 0:
        return {"candidates": [], "attempts": attempt}
    reply = await get_llm().json(
        "design",
        system=DESIGN_SYSTEM.format(language=language(state.get("lang", "es")), count=wanted),
        data={
            "topic": state.get("topic", ""),
            "fragments": [
                {"id": f["id"], "location": f["location"], "text": f["text"]}
                for f in state["fragments"]
            ],
            "avoid": [q["prompt"] for q in kept],
            "max_questions": wanted,
        },
        schema=DESIGN_SCHEMA,
        smart=True,
    )
    return {"candidates": reply.get("questions") or [], "attempts": attempt}


def check(question: dict[str, Any], fragments: dict[str, dict[str, Any]]) -> str | None:
    """Por qué una pregunta generada no sirve, o ``None`` si sirve."""
    fragment = fragments.get(question.get("fragment", ""))
    if fragment is None:
        return "cita un fragmento que no existe"
    quote = _plain(question.get("quote") or "")
    if len(quote.split()) < 3 or quote not in _plain(fragment["text"]):
        return "su cita no aparece en el fragmento"
    prompt = (question.get("prompt") or "").strip()
    answer = (question.get("answer") or "").strip()
    options = [o.strip() for o in question.get("options") or [] if o and o.strip()]
    kind = question.get("kind")
    if not prompt or kind not in KINDS:
        return "le falta el enunciado o el tipo"
    if kind == "opcion_multiple":
        same = [o for o in options if o.lower() == answer.lower()]
        if not 2 <= len(options) <= 6 or len(same) != 1:
            return "la respuesta no es una de sus opciones"
    if kind == "verdadero_falso" and answer.lower() not in ("verdadero", "falso"):
        return "la respuesta no es verdadero ni falso"
    if kind == "ordenar" and not 3 <= len(options) <= 6:
        return "no trae de 3 a 6 elementos"
    if kind == "abierta" and not answer:
        return "le falta la respuesta de referencia"
    return None


def validated(state: DesignState) -> dict[str, Any]:
    """Reglas fijas, sin modelo: solo sobreviven las preguntas con respaldo."""
    fragments = {f["id"]: f for f in state.get("fragments", [])}
    questions = list(state.get("questions", []))
    seen = {_plain(q["prompt"]) for q in questions}
    discarded = state.get("discarded", 0)
    for candidate in state.get("candidates", []):
        prompt = _plain(candidate.get("prompt") or "")
        if check(candidate, fragments) is not None or prompt in seen:
            discarded += 1
            continue
        seen.add(prompt)
        fragment = fragments[candidate["fragment"]]
        kind = candidate["kind"]
        options = [o.strip() for o in candidate.get("options") or [] if o and o.strip()]
        questions.append(
            {
                "id": f"g{len(questions) + 1}",
                "topic": (candidate.get("topic") or state.get("topic") or "General").strip(),
                "kind": kind,
                "prompt": candidate["prompt"].strip(),
                "answer": (candidate.get("answer") or "").strip(),
                "options": options if kind in ("opcion_multiple", "ordenar") else [],
                "explanation": (candidate.get("explanation") or "").strip(),
                "criteria": candidate.get("criteria") or [] if kind == "abierta" else [],
                # La cita que el host ve al aprobar.
                "citation": {
                    "file": fragment["file"],
                    "location": fragment["location"],
                    "quote": candidate["quote"].strip(),
                },
            }
        )
    return {"questions": questions[: state.get("max_questions", 10)], "discarded": discarded}


async def validate(state: DesignState) -> dict[str, Any]:
    return validated(state)


async def enough(state: DesignState) -> str:
    """Si quedaron pocas preguntas y aún hay intentos, se vuelve a generar."""
    wanted = state.get("max_questions", 10)
    short = len(state.get("questions", [])) < max(1, wanted // 2)
    retry = short and state.get("attempts", 0) < MAX_ATTEMPTS and state.get("fragments")
    return "generate" if retry else END


def build_graph() -> StateGraph:
    builder = StateGraph(DesignState)
    builder.add_node("extract", extract)
    builder.add_node("generate", generate)
    # Validar es determinista: reglas fijas, sin modelo.
    builder.add_node("validate", validate)
    builder.add_edge(START, "extract")
    builder.add_edge("extract", "generate")
    builder.add_edge("generate", "validate")
    builder.add_conditional_edges("validate", enough, {"generate": "generate", END: END})
    return builder


designer = build_graph().compile()
