from pathlib import Path

from pptx import Presentation


def _shape_text(shape) -> list[str]:
    lines = []
    if shape.has_text_frame:
        for paragraph in shape.text_frame.paragraphs:
            line = "".join(run.text for run in paragraph.runs).strip()
            if line:
                lines.append(line)
    if shape.has_table:
        for row in shape.table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                lines.append(" | ".join(cells))
    return lines


def parse_pptx(path: str | Path) -> list[dict]:
    """Return one record per slide, matching parse_pdf's page-record shape.

    Only .pptx is supported. Legacy .ppt binary files raise here and are
    skipped by the caller's per-file try/except during indexing.
    """

    pptx_path = Path(path)
    pages: list[dict] = []

    try:
        presentation = Presentation(pptx_path)
        for slide_index, slide in enumerate(presentation.slides):
            lines = []
            for shape in slide.shapes:
                lines.extend(_shape_text(shape))
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    lines.append(f"Notes: {notes}")
            text = "\n".join(lines)
            pages.append(
                {
                    "text": text.strip(),
                    "page_number": slide_index + 1,
                    "tables": [],
                    "metadata": {"is_likely_scan": len(text.strip()) < 20},
                }
            )
    except Exception as exc:
        raise RuntimeError(f"Failed to parse PPTX {pptx_path}: {exc}") from exc

    return pages
