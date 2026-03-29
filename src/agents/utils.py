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
