"""Test setup: a throwaway document tree and database, and no network calls.

Settings are read at import time (app.main builds the session middleware when
it is imported), so the environment is set here, before any app module loads.
"""

import hashlib
import math
import os
import re
import sys
import tempfile
from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WORK = Path(tempfile.mkdtemp(prefix="imm-test-"))
DOCS = WORK / "documents"
MACHINE_FOLDER = "IMM 101 - ACME (PX 200-50) SN12345"

os.environ.update(
    {
        "DOCS_ROOT": str(DOCS),
        "APP_DB_PATH": str(WORK / "app.db"),
        "SESSION_SECRET": "test-secret",
        "AUTH_REQUIRED": "true",
        "SESSION_HTTPS_ONLY": "false",
        "OPENAI_API_KEY": "test-key-not-used",
        "OPENAI_PROVIDER": "openai",
        "CHAT_MODEL": "fake-chat",
        "EMBEDDING_MODEL": "fake-embed",
    }
)


def _write_pdf(path: Path, pages: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=11)
    doc.save(path)
    doc.close()


_write_pdf(
    DOCS / MACHINE_FOLDER / "Operating Manual.pdf",
    [
        "Operating manual for the PX-200 press. Safety guards must stay closed during operation.",
        "Hydraulic system. The hydraulic oil tank capacity is 40 gallons. Use ISO VG 46 hydraulic oil. "
        "Change the oil filter every 2000 operating hours.",
        "Alarm E123 means the barrel heater zone 2 thermocouple is open. Check the thermocouple wiring.",
    ],
)
_write_pdf(
    DOCS / "_general" / "Molding Defects.pdf",
    ["Short shots are usually caused by low injection pressure, cold material, or blocked vents."],
)


def fake_embedding(text: str, dims: int = 256) -> list[float]:
    """Deterministic bag-of-words vector, so similar wording lands nearby."""

    vector = [0.0] * dims
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % dims] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


@pytest.fixture(scope="session", autouse=True)
def offline_embeddings():
    """Replace the embedding call for the whole run - nothing leaves the machine."""

    import app.rag.vector_store as vector_store

    original = vector_store.embed_texts
    vector_store.embed_texts = lambda texts, batch_size=None: [fake_embedding(t) for t in texts]
    yield
    vector_store.embed_texts = original
