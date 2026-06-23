"""Deterministic detector for conflicting identity data across turns.

Used by camila's triage to elevate the intent to IDENTITY_CONFLICT when the
same student handle shows up with two different surnames or two different
documento numbers — the scenario from the Neiva Cortés / Neiva Torres
transcript. Pure-Python; no LLM call so handoff is reliable regardless of
classifier wobble.
"""

from __future__ import annotations

import re
from typing import Iterable

# Capture sequences like "Neiva Cortés" (two capitalized words), accent-aware.
_NAME_TOKEN = re.compile(
    r"\b([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){1,3})\b"
)
# Documento de identidad: 6–12 digits, optional dots — strip dots before compare.
_DOCUMENTO = re.compile(r"\b(\d{6,12})\b")


def _extract_names(text: str) -> set[str]:
    return {m.group(1).strip() for m in _NAME_TOKEN.finditer(text or "")}


def _extract_documentos(text: str) -> set[str]:
    return {m.group(1) for m in _DOCUMENTO.finditer((text or "").replace(".", ""))}


def detect_identity_conflict(messages: Iterable[object]) -> bool:
    """Return True if the user's messages reference two distinct full names
    or two distinct documento numbers across the conversation.

    Only USER messages are inspected — assistant prompts can mention
    multiple names without being a conflict signal.
    """
    names: set[str] = set()
    documentos: set[str] = set()
    for msg in messages or []:
        if getattr(msg, "type", "") != "human":
            continue
        text = msg.content if hasattr(msg, "content") else str(msg)
        names |= _extract_names(text)
        documentos |= _extract_documentos(text)
        if len(names) >= 2 or len(documentos) >= 2:
            return True
    return False
