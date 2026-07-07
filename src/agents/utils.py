from langchain_core.runnables import RunnableConfig

_LANGUAGE_NAMES = {
    "es": "Spanish",
    "en": "English",
    "pt": "Portuguese",
}


def latest_user_messages(state) -> list[str]:
    """Return the texts of the user's current turn, oldest first.

    WhatsApp users often split one thought across several rapid messages
    ("Hola" / "una pregunta" / "¿tienen envíos?"). Each fragment lands as its
    own HumanMessage, so the trailing run of consecutive human messages —
    everything after the bot's last reply — is ONE logical turn. Nodes that
    only read `messages[-1]` answer the last fragment and ignore the rest.
    """
    parts: list[str] = []
    for msg in reversed(state.get("messages") or []):
        if getattr(msg, "type", "") != "human":
            break
        content = msg.content if hasattr(msg, "content") else str(msg)
        if content:
            parts.append(content)
    parts.reverse()
    return parts


def latest_user_text(state) -> str:
    """Return the user's current turn as one string (fragments joined by newlines).

    Drop-in replacement for `state["messages"][-1].content` at every site that
    feeds the user message to an extraction or conversational LLM call.
    """
    return "\n".join(latest_user_messages(state))


def language_instruction(lang: str) -> str:
    """Return an LLM instruction like 'Always respond in Spanish.' for the given language code."""
    name = _LANGUAGE_NAMES.get(lang, _LANGUAGE_NAMES["en"])
    return f"Always respond in {name}."


def format_user_context(state) -> str:
    """Return a formatted string describing the authenticated user for LLM system prompts.

    Returns an empty string when no user context is available so callers can
    safely concatenate without extra conditional logic.
    """
    ctx: dict = state.get("user_context") or {}
    if not ctx:
        return ""

    lines = []
    if ctx.get("name"):
        lines.append(f"- Nombre: {ctx['name']}")
    if ctx.get("email"):
        lines.append(f"- Email: {ctx['email']}")
    if ctx.get("phone"):
        lines.append(f"- Teléfono: {ctx['phone']}")
    if ctx.get("address"):
        lines.append(f"- Dirección: {ctx['address']}")

    if not lines:
        return ""

    return "\n\nContexto del usuario (la persona con quien estás hablando):\n" + "\n".join(lines)


def format_contact_tags(state) -> str:
    """Return a formatted string describing the contact's current tags for LLM system prompts."""
    tags: list = state.get("contact_tags") or []
    if not tags:
        return ""
    tag_names = [t.get("name", "") for t in tags if t.get("name")]
    if not tag_names:
        return ""
    return "\n\nEtiquetas actuales del contacto: " + ", ".join(tag_names)


def resolve_persona(state, default: str) -> str:
    """Return the channel-configured persona name, or `default` when unset.

    Snapshotted onto the conversation by the backend, so this value is stable
    across turns even if the channel's persona is later changed.
    """
    persona = (state.get("agent_persona_name") or "").strip()
    return persona if persona else default


def resolve_prompt(
    config: RunnableConfig,
    prompt_type: str,
    fallback: str,
    state=None,
) -> str:
    """Return tenant prompt content from config, or fallback to hardcoded default.

    When `state` is provided, any literal `{persona}` placeholder in the
    resolved string is substituted with the channel-configured persona name
    (or the default identity) BEFORE the caller runs its own `.format(...)`.
    This lets admins drop `{persona}` into any prompt template via the editor
    without each call site having to opt-in. Other placeholders (`{language_rule}`,
    etc.) are left untouched for the caller's downstream `.format(...)` to handle.
    """
    prompts = config.get("configurable", {}).get("prompts", {})
    payload = prompts.get(prompt_type, {})
    content = payload.get("content", "") if isinstance(payload, dict) else ""
    result = content if content else fallback
    if state is not None and "{persona}" in result:
        result = result.replace("{persona}", resolve_persona(state, "Helena"))
    return result


def resolve_model_config(config: RunnableConfig, prompt_type: str) -> dict:
    """Return model config overrides from tenant prompt."""
    prompts = config.get("configurable", {}).get("prompts", {})
    payload = prompts.get(prompt_type, {})
    return payload.get("model_config_data", {}) if isinstance(payload, dict) else {}
