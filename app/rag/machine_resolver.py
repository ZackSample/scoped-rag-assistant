from rapidfuzz import process

from app.config import get_settings


def list_machines() -> list[str]:
    """Machine folder names under DOCS_ROOT. Underscore folders are shared libraries."""

    root = get_settings().docs_root
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir() and not path.name.startswith("_"))


def resolve_machine(user_query: str, selected_machine: str | None = None) -> dict:
    if selected_machine:
        # The UI always passes the selected machine, which is the strongest scope signal.
        return {"machine_id": selected_machine, "confidence": "high", "needs_clarification": False}

    machines = list_machines()
    lowered = user_query.lower()
    for machine in machines:
        # API callers may mention the full machine folder in the question.
        if machine.lower() in lowered:
            return {"machine_id": machine, "confidence": "high", "needs_clarification": False}

    # Fuzzy matching is a fallback for typed machine names in non-UI calls.
    match = process.extractOne(user_query, machines, score_cutoff=80) if machines else None
    if match:
        return {"machine_id": match[0], "confidence": "medium", "needs_clarification": False}

    cross_machine_terms = ["compare", "between", "all machines", "which machines"]
    if any(term in lowered for term in cross_machine_terms):
        # Cross-machine questions are allowed through without forcing clarification.
        return {"machine_id": None, "confidence": "medium", "needs_clarification": False}

    return {"machine_id": None, "confidence": "low", "needs_clarification": True}
