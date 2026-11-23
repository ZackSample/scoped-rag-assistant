"""PDF parsing: page text plus table-aware extraction.

Plain get_text() flattens a table into run-together lines - fifty model rows
become one undifferentiated stream of numbers, which is the single upstream
cause of the wrong-row answers (a value's row membership is exactly what the
flattening destroys). PyMuPDF's find_tables() detects table geometry, so
detected tables are ALSO emitted as markdown with their rows intact.

Deliberate duplication: the flattened text stays in the page record and the
table is appended as its own record. Surgically excising the table span from
get_text() output needs brittle geometry clipping, and duplicate evidence is
cheap while missing evidence is the failure mode that matters. Retrieval's per-source caps keep the duplication from crowding
anything out.

Table records are shaped so chunking keeps rows whole: chunk_pages() chunks
each record independently, a record at most TABLE_BLOCK_WORDS long fits
inside one chunk (chunk_size defaults to 1000 words), and oversized tables
are pre-split on ROW boundaries with the header repeated per block - a row
split mid-way is exactly the damage this exists to undo.

find_tables() is best-effort geometry detection; any failure falls back to
exactly the old output for that page. PDF_TABLE_EXTRACTION=false is the
emergency kill switch (parse output is inert until a reindex runs, so the
flag only matters to indexing runs).

NOTE: new parse output reaches the index only when a file is re-embedded.
File hashes do not change when the parser does, so existing files need one
deliberate `scripts/index_documents.py --force` run after a parser change.
"""

from pathlib import Path

import pymupdf

from app.config import get_settings


# Blocks stay under this many words so a table block plus its header always
# fits one chunk (chunk_size default 1000). Not configurable on purpose: it
# is a relationship to chunk_size, not a tuning knob.
TABLE_BLOCK_WORDS = 800


def _cell_text(cell) -> str:
    """One cell as a single markdown-safe line."""

    text = " ".join(str(cell).split()) if cell is not None else ""
    return text.replace("|", "\\|")


def table_to_markdown(table) -> str:
    """Serialise one detected table as a pipe-row markdown table.

    Markdown rather than any custom format because the answering model
    already reads markdown tables natively, and because a pipe row keeps
    "which value belongs to which model" survivable through chunking,
    embedding, and the prompt.
    """

    try:
        rows = table.extract()
    except Exception:
        return ""

    lines = []
    for row in rows or []:
        cells = [_cell_text(cell) for cell in row]
        if any(cells):
            lines.append("| " + " | ".join(cells) + " |")
    if len(lines) < 2:
        # A one-line "table" is a detection artifact, not a table; the page
        # text already carries it.
        return ""
    # Header separator after the first row, which find_tables treats as the
    # header row. Even when the real header is elsewhere, a wrong separator
    # costs nothing - the rows are what matter.
    columns = lines[0].count("|") - 1
    lines.insert(1, "|" + " --- |" * columns)
    return "\n".join(lines)


def _split_table_blocks(markdown: str) -> list[str]:
    """Split an oversized table on row boundaries, repeating the header.

    A block always carries the header + separator so every value stays
    attached to its column name no matter which chunk it lands in.
    """

    lines = markdown.split("\n")
    if len(lines) < 3:
        return [markdown]
    header, separator, rows = lines[0], lines[1], lines[2:]

    blocks: list[list[str]] = []
    current: list[str] = []
    current_words = len(header.split()) + len(separator.split())
    for row in rows:
        row_words = len(row.split())
        if current and current_words + row_words > TABLE_BLOCK_WORDS:
            blocks.append(current)
            current = []
            current_words = len(header.split()) + len(separator.split())
        current.append(row)
        current_words += row_words
    if current:
        blocks.append(current)

    return ["\n".join([header, separator, *block]) for block in blocks]


def _page_tables(page) -> list[str]:
    """Markdown blocks for every table detected on a page. Never raises."""

    try:
        tables = page.find_tables()
    except Exception:
        return []

    blocks = []
    for table in getattr(tables, "tables", None) or []:
        markdown = table_to_markdown(table)
        if not markdown:
            continue
        blocks.extend(_split_table_blocks(markdown))
    return blocks


def parse_pdf(path: str | Path) -> list[dict]:
    """
    Return one record per page while preserving page numbers, plus one record
    per detected table (same page_number, so citations stay page-accurate).
    Image-only pages return empty text instead of crashing.
    """
    pdf_path = Path(path)
    pages: list[dict] = []
    extract_tables = get_settings().pdf_table_extraction

    try:
        with pymupdf.open(pdf_path) as doc:
            for page_index, page in enumerate(doc):
                # Preserve blank/scan pages so page numbers still match the PDF.
                text = page.get_text("text") or ""
                pages.append(
                    {
                        "text": text.strip(),
                        "page_number": page_index + 1,
                        "tables": [],
                        "metadata": {
                            "is_likely_scan": len(text.strip()) < 20,
                        },
                    }
                )
                if not extract_tables:
                    continue
                for block in _page_tables(page):
                    # Its own record: chunk_pages() chunks records
                    # independently, which is what keeps the table atomic
                    # instead of window-split mid-row.
                    pages.append(
                        {
                            "text": block,
                            "page_number": page_index + 1,
                            "tables": [],
                            "metadata": {"is_table_extract": True},
                        }
                    )
    except Exception as exc:
        raise RuntimeError(f"Failed to parse PDF {pdf_path}: {exc}") from exc

    return pages
