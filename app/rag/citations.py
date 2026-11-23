"""Structured citation helpers for clickable source references in the UI."""


# Distinct from the "General Knowledge" badge, which means the model answered
# with no document evidence at all. This label means the answer used real
# documents, just from the shared/machine-agnostic folder instead of the
# selected machine's own manuals.
GENERAL_DOCS_SCOPE_LABEL = "General Troubleshooting Docs"
MACHINE_SCOPE_LABEL = "Machine Docs"


def _source_scope(machine_id: str | None) -> str:
    from app.rag.retriever import shared_scope_ids

    return "general" if machine_id in shared_scope_ids() else "machine"


def build_citation(source: dict, citation_id: str) -> dict:
    """Convert one source row into a citation object with a caller-chosen id.

    Split out from build_citations() so callers that need stable ids across
    multiple registration calls (tools.CitationRegistry, which keys
    read_page/read_page_image off an id the model already saw) can assign ids
    themselves instead of always renumbering from 1.
    """

    page_start = source.get("page_start")
    page_end = source.get("page_end")
    # The label is what the technician sees on each clickable citation.
    page_label = f"page {page_start}" if page_start == page_end or not page_end else f"pages {page_start}-{page_end}"
    scope = source.get("source_scope") or _source_scope(source.get("machine_id"))
    return {
        "id": citation_id,
        "label": f"{source.get('source_file')}, {page_label}",
        "source_file": source.get("source_file"),
        "source_path": source.get("source_path"),
        "page_start": page_start,
        "page_end": page_end,
        "machine_id": source.get("machine_id"),
        "document_type": source.get("document_type"),
        # Lets the UI badge each citation and lets answer.py roll up a
        # per-answer badge without re-deriving scope from machine_id.
        "source_scope": scope,
        "scope_label": GENERAL_DOCS_SCOPE_LABEL if scope == "general" else MACHINE_SCOPE_LABEL,
        "preview_endpoint": "/page-preview",
        "pdf_endpoint": "/pdf",
    }


def build_citations(source_rows: list[dict]) -> list[dict]:
    """Convert source rows into stable citation objects for frontend actions."""

    return [build_citation(source, f"cite_{index}") for index, source in enumerate(source_rows, start=1)]
