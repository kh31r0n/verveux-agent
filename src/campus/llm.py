"""Acceso al modelo de los agentes de Campus. Los agentes no llaman a Gemini
directamente: piden un JSON con un esquema, o voz, y esta capa decide quién
contesta. Es Gemini de plataforma (ADC), no las credenciales del tenant: ver
``campus.config``.

Con ``CAMPUS_LLM=fake`` nadie sale a la red: las respuestas son fijas y
dependen solo de los datos. Así las pruebas y las puertas recorren todo el
camino de los agentes sin cuota ni credenciales.
"""

import asyncio
import io
import json
import logging
import wave
from dataclasses import dataclass
from typing import Any, Protocol

from .config import campus_settings as config

logger = logging.getLogger(__name__)

# El público es latinoamericano: sin «vosotros», que suena a España.
LANGUAGES = {
    "es": "español latinoamericano neutro (trata al grupo de «ustedes», nunca de «vosotros»)",
    "en": "inglés",
    "pt": "portugués de Brasil",
}

# Esperas en segundos ante un 429. El modelo grande comparte poca cuota en el
# proyecto: si sigue agotado, el rápido contesta en su lugar.
RETRY_WAITS = (0.0, 2.0, 6.0)


def exhausted(error: Exception) -> bool:
    """Si el error es de cuota (429 RESOURCE_EXHAUSTED) y vale la pena reintentar."""
    return getattr(error, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(error)


def log_usage(task: str, model: str, response: Any) -> None:
    """Deja en el registro los tokens de una llamada, para medir el costo de una
    partida: ``uso del modelo tarea=gm modelo=… entrada=… salida=…``."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return
    logger.info(
        "uso del modelo tarea=%s modelo=%s entrada=%d salida=%d razonamiento=%d",
        task,
        model,
        usage.prompt_token_count or 0,
        usage.candidates_token_count or 0,
        getattr(usage, "thoughts_token_count", None) or 0,
    )
SPEECH_LOCALES = {"es": "es-US", "en": "en-US", "pt": "pt-BR"}


@dataclass
class Attachment:
    """Un archivo del host que el modelo lee tal cual: PDF, imagen, audio o video."""

    data: bytes
    mime_type: str


class Llm(Protocol):
    async def json(
        self,
        task: str,
        *,
        system: str,
        data: dict[str, Any],
        schema: dict[str, Any],
        smart: bool = False,
        files: list[Attachment] | None = None,
    ) -> dict[str, Any]:
        """Responde con un objeto que cumple ``schema``."""

    async def speech(self, text: str, lang: str) -> bytes:
        """Audio WAV de ``text``."""

    def model(self, smart: bool = False) -> str:
        """El modelo que responde ``json``: queda con cada nota, para calibrarlas."""


class GeminiLlm:
    """Gemini por Vertex AI, con las credenciales de Google Cloud del entorno."""

    def __init__(self) -> None:
        from google import genai

        self._client = genai.Client(
            vertexai=True, project=config.gemini_project, location=config.gemini_location
        )

    def model(self, smart: bool = False) -> str:
        return config.smart_model if smart else config.fast_model

    async def json(
        self,
        task: str,
        *,
        system: str,
        data: dict[str, Any],
        schema: dict[str, Any],
        smart: bool = False,
        files: list[Attachment] | None = None,
    ) -> dict[str, Any]:
        from google.genai import types

        parts: list[Any] = [
            types.Part.from_bytes(data=file.data, mime_type=file.mime_type) for file in files or []
        ]
        parts.append(types.Part.from_text(text=json.dumps(data, ensure_ascii=False)))
        generation = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            # Baja, para que dos respuestas iguales reciban la misma nota.
            temperature=0.6 if task == "gm" else 0.1,
            max_output_tokens=16000 if smart else 4000,
        )
        tries = [(self.model(smart), wait) for wait in RETRY_WAITS]
        if smart:
            tries.append((self.model(False), 0.0))
        last: Exception | None = None
        for model, wait in tries:
            await asyncio.sleep(wait)
            try:
                response = await self._client.aio.models.generate_content(
                    model=model,
                    contents=[types.Content(role="user", parts=parts)],
                    config=generation,
                )
            except Exception as error:
                if not exhausted(error):
                    raise
                logger.warning("Sin cuota en %s para %s; se reintenta", model, task)
                last = error
                continue
            log_usage(task, model, response)
            return json.loads(response.text or "{}")
        assert last is not None
        raise last

    async def speech(self, text: str, lang: str) -> bytes:
        from google.genai import types

        response = await self._client.aio.models.generate_content(
            model=config.tts_model,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    language_code=SPEECH_LOCALES.get(lang, "es-US"),
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=config.tts_voice)
                    ),
                ),
            ),
        )
        log_usage("voz", config.tts_model, response)
        audio = response.candidates[0].content.parts[0].inline_data
        if "wav" in (audio.mime_type or ""):
            return audio.data
        # Algunos modelos devuelven PCM sin cabecera: 24 kHz, 16 bits, mono.
        return pcm_to_wav(audio.data)


def pcm_to_wav(pcm: bytes, rate: int = 24000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm)
    return buffer.getvalue()


class FakeLlm:
    """Respuestas fijas, sin red. Lo que contesta depende solo de los datos."""

    def model(self, smart: bool = False) -> str:
        return "fake-smart" if smart else "fake"

    async def json(
        self,
        task: str,
        *,
        system: str,
        data: dict[str, Any],
        schema: dict[str, Any],
        smart: bool = False,
        files: list[Attachment] | None = None,
    ) -> dict[str, Any]:
        if task == "gm":
            line = data["line"]["line"]
            # Quien va primero y quien está en racha: por su identificador en
            # lo escrito y por su alias en lo hablado.
            first = (data.get("standings") or [None])[0]
            streaks = [s for s in data.get("standings") or [] if s.get("streak", 0) >= 3]
            named = f" · primero {{{first['player_id']}}}" if first else ""
            named += "".join(f" · racha {{{s['player_id']}}}" for s in streaks)
            said = f", primero {first['alias']}" if first else ""
            said += "".join(f", {s['alias']} en racha" for s in streaks)
            reply: dict[str, Any] = {
                "text": f"[GM] {line}{named}",
                "spoken": f"Game Master: {line}{said}",
                "rooms": [
                    {"team_id": team["team_id"], "text": f"[GM] {team['team_id']}: {line}"}
                    for team in data.get("rooms_for", [])
                ],
            }
            # Como director: un premio en la primera pregunta delante del último.
            board = data.get("board")
            if board and board.get("ahead"):
                last = board["ahead"][-1]["squares"]
                first = next((s for s in last if s["kind"] == "question"), None)
                reply["board"] = [{"node": first["node"], "kind": "bonus"}] if first else []
            return reply
        if task == "awards":
            return {
                "awards": [
                    {
                        "player_id": c["player_id"],
                        "title": f"Premio {c['award']}",
                        "reason": f"{{{c['player_id']}}}: {c['fact']}.",
                    }
                    for c in data["candidates"]
                ]
            }
        if task == "describe":
            return {
                "descriptions": [
                    {
                        "question_id": concept["question_id"],
                        "text": f"Concepto de {concept['topic']}, "
                        f"de {len(concept['concept'])} letras.",
                    }
                    for concept in data["concepts"]
                ]
            }
        if task == "guess":
            # Da por bueno un intento que solo difiere en la última letra.
            concept, attempt = data["concept"].lower(), data["attempt"].lower()
            close = len(concept) == len(attempt) and concept[:-1] == attempt[:-1]
            return {
                "correct": close,
                "closeness": "hot" if close else "warm",
                "hint": f"Empieza por «{data['concept'][:1]}».",
            }
        if task == "evaluate":
            return {
                "results": [
                    {
                        "player_id": answer["player_id"],
                        # Más largo, mejor: basta para probar el camino.
                        "score": min(100, 40 + 2 * len(answer["text"])),
                        "feedback": "Buen intento. Revisa la respuesta de referencia.",
                        "confidence": 80,
                    }
                    for answer in data["answers"]
                ]
            }
        if task == "mate":
            return {"reply": "Soy la IA del grupo: ¿qué opción pueden descartar primero?"}
        if task == "nudge":
            return {
                "rooms": [
                    {"team_id": team, "text": "Soy la IA del grupo. ¡Repártanse las ideas!"}
                    for team in data["teams"]
                ]
            }
        if task == "summaries":
            return {
                "summaries": [
                    {
                        "player_id": student["player_id"],
                        "summary": f"Nota global {student['grade']}.",
                    }
                    for student in data["students"]
                ]
            }
        if task == "extract":
            return {
                "fragments": [
                    {
                        "location": "página 1",
                        "text": "El agua hierve a 100 grados al nivel del mar.",
                    },
                    {"location": "página 2", "text": "La fotosíntesis ocurre en los cloroplastos."},
                ]
            }
        if task == "design":
            questions = []
            for index, fragment in enumerate(data["fragments"][: data["max_questions"]]):
                questions.append(
                    {
                        "topic": data["topic"],
                        "kind": "verdadero_falso",
                        "prompt": fragment["text"],
                        "answer": "verdadero",
                        "options": [],
                        "explanation": f"Lo dice el material ({fragment['location']}).",
                        "criteria": [],
                        "fragment": fragment["id"],
                        "quote": fragment["text"][:30],
                    }
                )
                if index == 0:
                    # Una sin respaldo: el validador debe descartarla.
                    questions.append({**questions[-1], "prompt": "Inventada", "quote": "no está"})
            return {"questions": questions}
        raise ValueError(f"tarea desconocida: {task}")

    async def speech(self, text: str, lang: str) -> bytes:
        # Un décimo de segundo de silencio.
        return pcm_to_wav(b"\x00\x00" * 2400)


_llm: Llm | None = None


def get_llm() -> Llm:
    global _llm
    if _llm is None:
        _llm = FakeLlm() if config.llm == "fake" else GeminiLlm()
    return _llm


def language(lang: str) -> str:
    return LANGUAGES.get(lang, LANGUAGES["es"])
