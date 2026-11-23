"""Render xlsx/pptx citation previews as real page images via LibreOffice.

LibreOffice headless converts the source file to a PDF (preserving layout,
pictures, charts - not just extracted text), and the resulting PDF page is
rendered the same way PDF citations already are. This is a real external
program (not a Python package) - see LIBREOFFICE_PATH in app/config.py.
Callers fall back to a text preview when LibreOffice is unavailable rather
than failing outright.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pymupdf

from app.config import get_settings


CONVERT_TIMEOUT_SECONDS = 60


def libreoffice_available() -> bool:
    settings = get_settings()
    return shutil.which(settings.libreoffice_path) is not None


def _cache_key(path: Path) -> str:
    stat = path.stat()
    # mtime + size is enough to invalidate the cache when the source file
    # changes, without hashing the whole file on every preview request.
    return f"{path.stem}__{int(stat.st_mtime)}_{stat.st_size}.pdf"


def _convert_to_pdf(path: Path) -> Path:
    settings = get_settings()
    cache_dir = settings.office_preview_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_pdf = cache_dir / _cache_key(path)
    if cached_pdf.exists():
        return cached_pdf

    soffice = shutil.which(settings.libreoffice_path) or settings.libreoffice_path
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = subprocess.run(
            [
                soffice,
                "--headless",
                "--norestore",
                "--convert-to",
                "pdf",
                "--outdir",
                tmp_dir,
                str(path),
            ],
            capture_output=True,
            timeout=CONVERT_TIMEOUT_SECONDS,
        )
        # LibreOffice writes <stem>.pdf into --outdir using the source
        # filename, not our cache key, so convert into a temp dir and copy.
        produced_pdf = Path(tmp_dir) / f"{path.stem}.pdf"
        if result.returncode != 0 or not produced_pdf.exists():
            stderr = result.stderr.decode(errors="ignore")[:500]
            raise RuntimeError(f"LibreOffice conversion failed for {path.name}: {stderr}")
        shutil.copy(produced_pdf, cached_pdf)

    return cached_pdf


def render_office_page_png(path: Path, page_number: int, zoom: float = 1.6) -> bytes:
    """Render one slide/sheet as a PNG by converting the file to PDF first.

    page_number maps 1:1 onto the converted PDF's page order (slide N -> PDF
    page N for pptx; sheet N -> PDF page N for xlsx, assuming one page per
    sheet, which is LibreOffice's default for reasonably sized sheets).
    """

    pdf_path = _convert_to_pdf(path)
    with pymupdf.open(pdf_path) as doc:
        if page_number < 1:
            raise RuntimeError("Page number must be 1 or greater.")
        if page_number > len(doc):
            raise RuntimeError(f"Converted document only has {len(doc)} pages.")
        page = doc[page_number - 1]
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        return pixmap.tobytes("png")
