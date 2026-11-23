"""Citation previews: render the cited page, sheet, slide or image for the UI."""

import base64
import io
from pathlib import Path, PureWindowsPath

import pymupdf
from fastapi.responses import FileResponse

from app.config import get_settings


# Standalone images are a document of their own: a scanned drawing whose only
# content is the picture.
#
# Deliberately no SVG. SVG is a scriptable document, not an image, and these are
# served back to a browser - adding it would turn "add a drawing" into stored
# XSS. Raster formats only, and everything is re-encoded to PNG for preview
# rather than passed through, so a malformed file fails in the decoder here
# instead of in the technician's browser.
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
SUPPORTED_SUFFIXES = {".pdf", ".xlsx", ".pptx"} | IMAGE_SUFFIXES
MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}

# A preview is shown in a panel, not printed, so an enormous scan is downscaled
# rather than streamed at native size. Roughly matches a PDF page rendered at
# the zoom used below.
MAX_PREVIEW_EDGE_PIXELS = 2600


def _allowed_roots(settings) -> list[Path]:
    """Every folder a citation is allowed to be served from.

    Machine docs and the general/machine-agnostic folder can be separate roots
    (DOCS_ROOT vs GENERAL_DOCS_PATH), so both must be checked or general-folder
    citations fail the safety check even when the file is legitimate.
    """

    roots = [settings.docs_root.resolve()]
    general_root = settings.general_root.resolve()
    if general_root not in roots:
        roots.append(general_root)
    return roots


def _resolve_source_path(source_path: str) -> Path:
    """Resolve a stored, root-relative source_path to a file on this machine.

    Stored paths are DOCS_ROOT-relative (or general-root-relative) with forward
    slashes, so a row indexed on one machine resolves unchanged on another.
    Each allowed root is tried until the file is found, so the caller does not
    need to know which root a row came from.
    """

    if Path(source_path).is_absolute():
        # Already resolved (the /pdf-link registry stores validated absolute
        # paths); the caller's containment check still applies.
        return Path(source_path)
    roots = _allowed_roots(get_settings())
    parts = PureWindowsPath(source_path).parts
    for root in roots:
        candidate = root.joinpath(*parts)
        if candidate.exists():
            return candidate
    # Fall through to the first root so the containment check produces a
    # sensible "outside the configured folders" error.
    return roots[0].joinpath(*parts)


def _safe_source_path(source_path: str, allowed_suffixes: set[str] = SUPPORTED_SUFFIXES) -> Path:
    path = _resolve_source_path(source_path).resolve()
    roots = _allowed_roots(get_settings())
    # Never stream arbitrary server files just because the browser sent a path.
    if not any(root == path or root in path.parents for root in roots):
        raise RuntimeError("Requested file is outside the configured document folders.")
    if path.suffix.lower() not in allowed_suffixes:
        raise RuntimeError(f"Requested source type {path.suffix!r} is not supported.")
    if not path.exists():
        raise RuntimeError("Requested file does not exist.")
    return path


def safe_pdf_path(source_path: str) -> Path:
    """PDF-only: used by the page-image renderer, which can only render PDFs."""

    return _safe_source_path(source_path, {".pdf"})


def safe_source_path(source_path: str) -> Path:
    """Return a local path for any supported source type."""

    return _safe_source_path(source_path, SUPPORTED_SUFFIXES)


def safe_image_path(source_path: str) -> Path:
    """Return a local path for a standalone image source."""

    return _safe_source_path(source_path, IMAGE_SUFFIXES)


def render_image_page_png(source_path: str, page_number: int = 1) -> bytes:
    """Render one frame of a standalone image as PNG.

    Always re-encoded rather than passed through, for three reasons: TIFF is not
    displayable in a browser at all, PyMuPDF cannot decode WEBP so the rest of
    the preview path cannot handle it, and re-encoding means a corrupt file
    fails here rather than in the technician's browser.

    Multi-frame TIFFs are treated as multi-page documents, because that is what
    a scanned document delivered as TIFF actually is - page_number selects the
    frame, matching how every other source type in this module behaves.
    """

    from PIL import Image

    path = safe_image_path(source_path)
    with Image.open(path) as image:
        frames = getattr(image, "n_frames", 1)
        index = max(1, page_number) - 1
        if index >= frames:
            raise RuntimeError(f"Requested page {page_number} but the image has {frames} frame(s).")
        if index:
            image.seek(index)

        # PNG cannot store CMYK or exotic TIFF modes, and 1-bit scans render
        # black-on-black through some viewers, so normalise first.
        frame = image.convert("RGB")

        longest = max(frame.size)
        if longest > MAX_PREVIEW_EDGE_PIXELS:
            scale = MAX_PREVIEW_EDGE_PIXELS / longest
            frame = frame.resize((max(1, int(frame.width * scale)), max(1, int(frame.height * scale))))

        buffer = io.BytesIO()
        frame.save(buffer, format="PNG")
        return buffer.getvalue()


def render_pdf_page_png(source_path: str, page_number: int, zoom: float = 1.6) -> bytes:
    path = safe_pdf_path(source_path)
    if page_number < 1:
        raise RuntimeError("Page number must be 1 or greater.")

    with pymupdf.open(path) as doc:
        if page_number > len(doc):
            raise RuntimeError(f"PDF only has {len(doc)} pages.")
        # Render only the requested page to keep citation previews lightweight.
        page = doc[page_number - 1]
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        return pixmap.tobytes("png")


def _text_preview(pages: list[dict], page_number: int, unit_label: str) -> dict:
    if page_number < 1 or page_number > len(pages):
        raise RuntimeError(f"Requested {unit_label} {page_number} does not exist ({len(pages)} total).")
    return {"kind": "text", "unit_label": unit_label, "content": pages[page_number - 1]["text"]}


def _image_result(png: bytes) -> dict:
    # Reported as image/png regardless of the source format, because that is
    # what was encoded.
    return {"kind": "image", "media_type": "image/png", "content": base64.b64encode(png).decode("ascii")}


def render_preview(source_path: str, page_number: int, zoom: float | None = None) -> dict:
    """Return a citation preview for the artifact panel, dispatched by file type.

    `zoom` applies to PDF pages only and defaults to the panel resolution
    (1.6). The model-facing caller (read_page_image in app/rag/tools.py)
    passes a higher value: a vision model reading a dense spec table gets
    exactly the pixels it is handed, unlike a human who can pinch-zoom.

    PDFs and standalone images get a rendered page image. xlsx/pptx get a
    rendered page image through LibreOffice when it is installed
    (office_preview.py), and the extracted sheet/slide text otherwise, so a
    preview never just breaks.
    """

    path = safe_source_path(source_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _image_result(render_pdf_page_png(source_path, page_number, zoom=zoom or 1.6))

    if suffix in IMAGE_SUFFIXES:
        return _image_result(render_image_page_png(source_path, page_number))

    if suffix in (".xlsx", ".pptx"):
        png = _render_office_page_best_effort(path, page_number)
        if png is not None:
            return _image_result(png)
        unit_label = "sheet" if suffix == ".xlsx" else "slide"
        return _text_preview(_text_parser_for_suffix(suffix)(path), page_number, unit_label)

    raise RuntimeError(f"Preview is not supported for {suffix!r} files.")


def _render_office_page_best_effort(path: Path, page_number: int) -> bytes | None:
    """Render through LibreOffice if available. Return None if it is not."""

    from app.rag.office_preview import libreoffice_available, render_office_page_png

    if libreoffice_available():
        try:
            return render_office_page_png(path, page_number)
        except Exception:
            pass  # Fall back to the text preview rather than a broken panel.
    return None


def _text_parser_for_suffix(suffix: str):
    if suffix == ".xlsx":
        from app.rag.parse_xlsx import parse_xlsx

        return parse_xlsx
    if suffix == ".pptx":
        from app.rag.parse_pptx import parse_pptx

        return parse_pptx
    raise RuntimeError(f"No text parser for {suffix!r}.")


def pdf_file_response(source_path: str) -> FileResponse:
    """Serve a source file to the browser, inline."""

    path = safe_source_path(source_path)
    media_type = MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media_type, filename=path.name, content_disposition_type="inline")
