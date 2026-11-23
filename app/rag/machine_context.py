"""Tell a model which machine it is working on, in one shared place.

A machine scope used to be a retrieval filter only - nothing told the
answering model which machine the question was about. That is invisible until
a document covers more than one machine: a fluid-capacity matrix listing fifty
models in one table will happily give the model fifty candidate answers, and
without an identity it picks a row that merely looks plausible.

Identity has three consumers, which is why this lives in its own module:

  * the answering model's tool loop (tools.py) - machine_context()
  * the evidence judge (answer.py) - machine_identity_block()
  * the vision sub-call that reads pages (tools.py) - machine_identity_block()

tools.py imports answer.py, so a helper both need cannot live in either
without creating a cycle. This module imports only the machine registry.

Everything stated here is DATA the application already knows - number, model,
curated facts - injected into user messages at runtime. The system prompt
stays conventions-only; per-machine specifics never appear in prompt text.
"""


# Rendering order and labels for the curated facts fields. Unknown keys still
# render (a field added to the facts file must reach the model without a code
# change here), after the known ones, with underscores read as spaces.
_FACT_LABELS = {
    "family": "model family: {value}",
    "tonnage_tons": "clamping force: {value} US tons",
    "installed_screw_mm": "installed screw diameter: {value} mm",
    "shot_size_oz": "shot size: {value} oz",
    "datasheet_designation": "designation on its own datasheet: {value}",
}
# `notes` is free text written for the model, appended as its own sentence
# rather than squeezed into the key: value list.
_FACT_NOTES_KEY = "notes"


def machine_facts_block(record: dict) -> str:
    """Render a machine record's curated facts as one prompt-ready line.

    Empty string when the machine has no facts, so a machine without facts
    produces exactly the context it would have produced before facts existed.

    The label says "from its records", not "trusted": the model should treat
    these as identity context, not as gospel that overrides a cited document.
    """

    facts = record.get("facts") or {}
    parts = []
    for key, template in _FACT_LABELS.items():
        if key in facts and facts[key] not in (None, ""):
            parts.append(template.format(value=facts[key]))
    for key, value in facts.items():
        if key in _FACT_LABELS or key == _FACT_NOTES_KEY or value in (None, ""):
            continue
        parts.append(f"{key.replace('_', ' ')}: {value}")

    lines = []
    if parts:
        lines.append("Known facts for this machine (from its records): " + "; ".join(parts) + ".")
    notes = facts.get(_FACT_NOTES_KEY)
    if notes:
        lines.append(str(notes))
    return " ".join(lines)


def _identity_heading(record: dict) -> str:
    model = record.get("model") or ""
    heading = f"You are answering about machine {record['machine_number']}"
    heading += f" - {model}." if model else "."
    return heading


def machine_context(machine_id: str) -> str:
    """Describe the machine being asked about, for the answering model.

    Number, model and facts only. The serial identifies the physical unit,
    which questions do not turn on, and every extra token in the designation
    is another number the model can mistake for the one that selects a row.
    """

    record = _resolve(machine_id)
    if not record:
        # Unknown folder: degrade to naming the folder rather than failing the
        # question or silently saying nothing.
        return f"You are answering about machine folder {machine_id}."

    facts_block = machine_facts_block(record)
    heading = _identity_heading(record)
    if facts_block:
        heading += "\n\n" + facts_block

    # The row-matching rule went through three generations, and the wording is
    # load-bearing:
    #
    #   gen 1 (nothing): the model guessed rows and answered confidently from
    #     the wrong one.
    #   gen 2 ("match the designation as a whole"): overshot into refusals.
    #     Documents abbreviate model names, so a whole-designation match is
    #     impossible against most tables, and an obedient model declined.
    #   gen 3 (below): consistency, not spelling, and an AFFIRMATIVE branch.
    #     A rule that only prohibits leaves the model nothing it may do; this
    #     one says when it MUST commit: exactly one consistent row.
    return (
        heading
        + "\n\nYou have this machine's documents and the shared general troubleshooting "
        "library, and nothing else. A document can still cover a whole product line and "
        "list many models in one table, so a value is only this machine's if its row "
        "belongs to it. Designations are spelled differently in different documents, so "
        "judge each row by consistency, not exact spelling: a row is consistent when "
        "nothing in it contradicts this machine's model, family, or size, and "
        "inconsistent when anything does. If exactly one row is consistent with this "
        "machine, that row is this machine's - use it and name the row you used. If "
        "more than one row remains consistent, present those rows together with "
        "whatever would decide between them and ask. If none is consistent, say the "
        "document does not cover this machine rather than answering from the nearest row."
    )


def machine_identity_block(machine_id: str) -> str:
    """Compact identity + facts for the evidence judge and the vision call.

    Returns "" for anything the registry cannot resolve (the general folder, a
    misconfigured share). Those scopes are not a machine, and a heading naming
    a folder there is noise that can only mislead.
    """

    record = _resolve(machine_id)
    if not record:
        return ""

    lines = [
        "The question is about machine "
        f"{record['machine_number']}{' - ' + record['model'] if record.get('model') else ''}."
    ]
    facts_block = machine_facts_block(record)
    if facts_block:
        lines.append(facts_block)
    return "\n".join(lines)


def _resolve(machine_id: str) -> dict | None:
    # Imported at call time so tests can monkeypatch the registry. A registry
    # failure degrades the context; it never fails the question.
    from app.machines.registry import get_machine_by_id

    try:
        return get_machine_by_id(machine_id)
    except Exception:
        return None
