"""Shared helpers for parsing JSON out of LLM responses."""


def strip_json_fences(raw: str) -> str:
    """Remove markdown code fences that Gemini adds around JSON responses.

    Handles both ```json ... ``` and ``` ... ``` variants.
    """
    s = raw.strip()
    if s.startswith("```"):
        # Drop the opening fence line (```json or ```)
        s = s.split("\n", 1)[-1]
        # Drop the closing fence
        s = s.rsplit("```", 1)[0]
    return s.strip()
