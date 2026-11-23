"""Document preview endpoints used by the technician citation viewer."""

import time
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.rag.pdf_preview import pdf_file_response, render_preview, safe_source_path


router = APIRouter()
PDF_LINK_TTL_SECONDS = 15 * 60
# Tokens live only in this process; they are short-lived browser helpers, not auth.
_pdf_links: dict[str, dict] = {}


class PagePreviewRequest(BaseModel):
    """Request body for rendering one cited page into a browser-friendly PNG."""

    source_path: str = Field(min_length=1)
    page_number: int = Field(ge=1)


class PdfLinkRequest(BaseModel):
    """Request body for creating a short browser-safe document URL."""

    source_path: str = Field(min_length=1)


@router.post("/page-preview")
def page_preview_endpoint(request: PagePreviewRequest) -> dict:
    """Return a cited page/sheet/slide preview for the artifact panel.

    Response shape:
    {"kind": "image", "media_type": ..., "content": base64 png}
    {"kind": "text", "unit_label": "sheet"|"slide", "content": "..."}
    """

    try:
        return render_preview(request.source_path, request.page_number)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/pdf-link")
def pdf_link_endpoint(request: PdfLinkRequest) -> dict:
    """Validate a source file and return a short URL that can be opened in a tab."""

    try:
        path = safe_source_path(request.source_path)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    _prune_pdf_links()
    # Store the validated local path so the browser opens a safe token URL.
    token = uuid4().hex
    _pdf_links[token] = {"source_path": str(path), "expires_at": time.time() + PDF_LINK_TTL_SECONDS}
    return {"url": f"/pdf/{token}", "expires_in_seconds": PDF_LINK_TTL_SECONDS}


@router.get("/pdf/{token}")
def pdf_token_endpoint(token: str):
    """Stream a previously validated document link to the requesting browser."""

    _prune_pdf_links()
    link = _pdf_links.get(token)
    if not link:
        raise HTTPException(status_code=404, detail="PDF link expired or was not found.")
    try:
        return pdf_file_response(link["source_path"])
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/pdf")
def pdf_endpoint(source_path: str):
    """Stream a source document to the requesting browser or tablet."""

    try:
        return pdf_file_response(source_path)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _prune_pdf_links() -> None:
    """Drop expired links from the in-process link registry."""

    now = time.time()
    # Pruned during normal requests so no background task is needed.
    for token in [token for token, link in _pdf_links.items() if link["expires_at"] < now]:
        _pdf_links.pop(token, None)
