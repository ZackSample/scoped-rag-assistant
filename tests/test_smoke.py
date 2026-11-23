"""End-to-end smoke test with a scripted stand-in for the model API.

Covers indexing, hybrid retrieval, login, the machine picker, a full streamed
chat turn (forced full search -> evidence judge -> answer draft -> forced page
verification -> final answer), history, feedback and citation previews.
"""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

MACHINE_FOLDER = "IMM 101 - ACME (PX 200-50) SN12345"  # created in conftest.py


# --- a scripted stand-in for the OpenAI client --------------------------------


class _Item(SimpleNamespace):
    def model_dump(self, exclude_none: bool = True) -> dict:
        return {key: value for key, value in vars(self).items() if not (exclude_none and value is None)}


class _Stream:
    def __init__(self, text: str = "", tool_call: dict | None = None):
        self._text = text
        self._tool_call = tool_call

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        yield SimpleNamespace(type="response.reasoning_summary_text.delta", delta="**Checking** the evidence.")
        yield SimpleNamespace(type="response.reasoning_summary_part.done")
        for word in self._text.split(" "):
            yield SimpleNamespace(type="response.output_text.delta", delta=word + " ")

    def get_final_response(self):
        output = []
        if self._tool_call:
            output.append(_Item(type="function_call", status="completed", **self._tool_call))
        else:
            output.append(_Item(type="message", role="assistant", content=[{"type": "output_text", "text": self._text}]))
        return SimpleNamespace(
            output=output,
            output_text="" if self._tool_call else self._text,
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        )


class FakeClient:
    """Answers like a model would, following the pipeline's own protocol."""

    def __init__(self):
        self.responses = SimpleNamespace(stream=self._stream)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.stream_calls = 0

    def _stream(self, **kwargs):
        self.stream_calls += 1
        items = kwargs["input"]
        user_texts = [m.get("content") for m in items if m.get("role") == "user" and isinstance(m.get("content"), str)]
        asked_to_verify = any("Before you answer" in text for text in user_texts)
        looked_at_page = any(m.get("name") == "read_page_image" for m in items if m.get("type") == "function_call")
        if asked_to_verify and not looked_at_page:
            return _Stream(
                tool_call={
                    "call_id": "call_1",
                    "name": "read_page_image",
                    "arguments": json.dumps({"citation_id": "cite_1", "verify": "oil capacity 40 gallons"}),
                }
            )
        return _Stream(text="The hydraulic oil tank holds **40 gallons** of ISO VG 46 oil [cite_1].")

    def _chat(self, **kwargs):
        system = kwargs["messages"][0]["content"]
        if kwargs.get("response_format"):
            content = json.dumps(
                {
                    "has_answer": True,
                    "confidence": "high",
                    "answerable_scope": "direct",
                    "best_citation_ids": ["cite_1"],
                    "reason": "Page 2 of the operating manual states a 40 gallon tank.",
                    "suggested_next_query": None,
                }
            )
        elif system.startswith("You are looking at one rendered page"):
            content = "The page states: The hydraulic oil tank capacity is 40 gallons."
        else:
            content = "40 gallons [cite_1]."
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


# --- fixtures ------------------------------------------------------------------


@pytest.fixture(scope="module")
def indexed():
    from app.rag.ingest import index_documents

    return index_documents()


@pytest.fixture(scope="module")
def fake_client():
    import app.rag.answer
    import app.rag.tools

    client = FakeClient()
    patches = [(app.rag.answer, "get_client"), (app.rag.tools, "get_client")]
    originals = [getattr(module, name) for module, name in patches]
    for module, name in patches:
        setattr(module, name, lambda: client)
    yield client
    for (module, name), original in zip(patches, originals):
        setattr(module, name, original)


@pytest.fixture(scope="module")
def client(indexed, fake_client):
    from app.auth.passwords import hash_password
    from app.auth.users import upsert_user
    from app.main import app

    upsert_user("tech1", "Tech One", hash_password("correct horse"))
    with TestClient(app) as test_client:
        yield test_client


def _sse_events(response) -> list[dict]:
    return [json.loads(line[len("data: "):]) for line in response.text.splitlines() if line.startswith("data: ")]


# --- tests ---------------------------------------------------------------------


def test_indexing_picks_up_machine_and_general_documents(indexed):
    assert indexed["indexed_files"] == 2
    assert indexed["failed_files"] == 0


def test_hybrid_retrieval_is_scoped_and_finds_codes(indexed):
    from app.rag.retriever import retrieve

    hits = retrieve("hydraulic oil tank capacity", MACHINE_FOLDER, top_k=5)
    assert hits[0]["metadata"]["source_file"] == "Operating Manual.pdf"
    assert hits[0]["metadata"]["page_start"] == 2
    assert {hit["metadata"]["machine_id"] for hit in hits} <= {MACHINE_FOLDER, "general"}

    alarm = retrieve("what does alarm E123 mean", MACHINE_FOLDER, top_k=3)
    assert alarm[0]["metadata"]["page_start"] == 3
    assert alarm[0]["retrieval_method"] in {"keyword", "hybrid"}


def test_gate_and_login(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/machine-registry").status_code == 401
    assert client.post("/auth/login", json={"username": "tech1", "password": "nope"}).json()["ok"] is False
    assert client.post("/auth/login", json={"username": "tech1", "password": "correct horse"}).json()["ok"] is True
    assert client.get("/auth/me").json()["user"]["display_name"] == "Tech One"


def test_machine_picker_lists_folder_machines(client):
    machines = client.get("/machine-registry").json()["machines"]
    assert [m["machine_number"] for m in machines] == ["101"]
    assert machines[0]["model"] == "ACME (PX 200-50)"
    assert machines[0]["serial"] == "SN12345"
    assert machines[0]["facts"]["tonnage_tons"] == 200


def test_streamed_chat_turn_end_to_end(client, fake_client):
    response = client.post("/chat/ask/stream", json={"machine_number": "101", "question": "What is the hydraulic oil capacity?"})
    assert response.status_code == 200
    events = _sse_events(response)
    kinds = {event["type"] for event in events}
    assert {"progress", "reasoning", "answer_delta", "answer_mode", "result"} <= kinds
    assert "error" not in kinds

    result = events[-1]
    assert result["type"] == "result"
    assert "40 gallons" in result["answer"]
    assert result["citations"][0]["source_file"] == "Operating Manual.pdf"
    assert result["citations"][0]["page_start"] == 2
    tool_path = result["retrieval_diagnostics"]["tool_path"]
    assert tool_path[0] == "run_full_search"
    assert "read_page_image" in tool_path
    assert result["retrieval_diagnostics"]["docs_answer_found"] is True
    assert result["source_badges"] == ["Machine Docs"]

    sessions = client.get("/chat/sessions/101").json()["sessions"]
    assert len(sessions) == 1 and sessions[0]["title"] == "What is the hydraulic oil capacity?"
    detail = client.get(f"/chat/sessions/101/{result['session_id']}").json()["session"]
    assert detail["answers"][0]["citations"][0]["source_file"] == "Operating Manual.pdf"
    assert detail["answers"][0]["activity"], "the activity trail should be stored with the answer"

    feedback = client.post("/chat/feedback", json={"answer_id": result["answer_id"], "rating": "up"}).json()
    assert feedback["stored"] is True


def test_citation_preview_and_document_link(client):
    source_path = f"{MACHINE_FOLDER}/Operating Manual.pdf"
    preview = client.post("/page-preview", json={"source_path": source_path, "page_number": 2}).json()
    assert preview["kind"] == "image" and preview["media_type"] == "image/png"

    link = client.post("/pdf-link", json={"source_path": source_path}).json()
    pdf = client.get(link["url"])
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")

    escape = client.post("/page-preview", json={"source_path": "../../etc/passwd", "page_number": 1})
    assert escape.status_code == 503


def test_web_shell_is_served(client):
    page = client.get("/app/")
    assert page.status_code == 200 and "__ASSET_VERSION__" not in page.text
    assert client.get("/login").status_code == 200


def test_viewer_pages_through_a_cited_document(client):
    import io

    from PIL import Image

    source_path = f"{MACHINE_FOLDER}/Operating Manual.pdf"
    info = client.get("/documents/info", params={"source_path": source_path}).json()
    assert info == {"title": "Operating Manual.pdf", "kind": "pdf", "page_count": 3}

    # Prev/Next walks every page of the document, not just the cited one.
    for page in (1, 2, 3):
        image = client.get("/documents/page.png", params={"source_path": source_path, "page": page, "w": 600})
        assert image.status_code == 200 and image.headers["content-type"] == "image/png"
        assert Image.open(io.BytesIO(image.content)).width == 600
    past_the_end = client.get("/documents/page.png", params={"source_path": source_path, "page": 4})
    assert past_the_end.status_code == 404

    # Thumbnails ask for a small width; requests are clamped to a sane range.
    thumb = client.get("/documents/page.png", params={"source_path": source_path, "page": 2, "w": 10})
    assert Image.open(io.BytesIO(thumb.content)).width == 120

    outside = client.get("/documents/info", params={"source_path": "../../etc/passwd"})
    assert outside.status_code == 404
