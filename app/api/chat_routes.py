"""Machine-scoped chat endpoints for the technician-facing app."""

import json
import queue
import threading
from time import perf_counter

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.db.diagnostics import record_retrieval_diagnostics
from app.db.feedback import RATINGS, feedback_for_session, record_feedback
from app.db.history import (
    add_answer,
    add_message,
    ensure_session,
    find_reuse_candidate,
    list_sessions,
    load_session,
    set_session_title_if_empty,
)
from app.machines.registry import get_machine_by_number
from app.rag.tools import answer_conversationally


router = APIRouter(prefix="/chat", tags=["chat"])


class ScopedAskRequest(BaseModel):
    """Request shape for a chat turn tied to one selected machine."""

    machine_number: str = Field(min_length=1)
    question: str = Field(min_length=1)
    session_id: int | None = None
    context_mode: str = Field(default="new_question", pattern="^(new_question|follow_up)$")
    # "expand" is the More detail button: reuse this turn's evidence and read it
    # harder, rather than running retrieval again and risking different evidence.
    intent: str = Field(default="question", pattern="^(question|expand)$")
    active_citations: list[dict] = Field(default_factory=list)
    conversation_tail: list[dict] = Field(default_factory=list)


def _related_previous_answer(request: ScopedAskRequest, machine: dict) -> dict | None:
    """A prior answer worth mentioning alongside this one, or None.

    Informational only: the question is always answered, and a close prior
    answer is offered next to it as something to open if wanted. Runs on every
    question - first, follow-up, same session - because each exclusion tried
    made the hint invisible in exactly the cases most likely to trigger it.
    """

    return find_reuse_candidate(machine["machine_id"], request.question)


def _run_and_persist(
    request: ScopedAskRequest,
    session_id: int,
    machine: dict,
    on_progress=None,
    on_answer_delta=None,
    on_reasoning=None,
    on_stream_event=None,
    user_id: int | None = None,
) -> dict:
    """Run the conversational tool-calling pipeline, then persist the answer + diagnostics.

    Shared by the plain JSON endpoint and the streaming endpoint so the two
    do not drift out of sync. The model itself decides whether to use the
    active citations, search again, or run a full search - see app/rag/tools.py.
    """

    started_at = perf_counter()
    context_mode = "follow_up" if request.context_mode == "follow_up" and request.active_citations else "new_question"
    payload = answer_conversationally(
        query=request.question,
        selected_machine=machine["machine_id"],
        conversation_tail=request.conversation_tail,
        active_citations=request.active_citations,
        on_progress=on_progress,
        on_answer_delta=on_answer_delta,
        on_reasoning=on_reasoning,
        on_stream_event=on_stream_event,
        intent=request.intent,
    )

    duration_ms = int((perf_counter() - started_at) * 1000)
    add_message(session_id, "assistant", payload["answer"])
    answer_id = add_answer(
        session_id=session_id,
        machine_id=machine["machine_id"],
        question=request.question,
        answer=payload["answer"],
        citations=payload.get("citations", []),
        user_id=user_id,
        full_answer=payload.get("full_answer"),
    )
    run_id = record_retrieval_diagnostics(
        answer_id=answer_id,
        session_id=session_id,
        machine_id=machine["machine_id"],
        question=request.question,
        diagnostics=payload.get("retrieval_diagnostics"),
        source_badges=payload.get("source_badges", []),
        context_mode=context_mode,
        duration_ms=duration_ms,
    )
    return {
        **payload,
        "session_id": session_id,
        "answer_id": answer_id,
        "retrieval_run_id": run_id,
        "machine": machine,
        "retrieval_scope": _retrieval_scope(machine, mode=context_mode),
    }


def _start_turn(request: ScopedAskRequest, http_request: Request) -> tuple[dict, int, int | None]:
    machine = get_machine_by_number(request.machine_number)
    if not machine:
        raise HTTPException(status_code=404, detail="Machine number was not found.")

    user_id = _identified_user_id(http_request)
    # Every turn belongs to a machine session so history and reuse stay scoped.
    session_id = ensure_session(machine["machine_id"], request.session_id, user_id=user_id)
    add_message(session_id, "user", request.question)
    set_session_title_if_empty(session_id, _session_title(request.question))
    return machine, session_id, user_id


@router.post("/ask")
def scoped_ask(request: ScopedAskRequest, http_request: Request) -> dict:
    """Answer a question using only the selected machine's indexed documents."""

    machine, session_id, user_id = _start_turn(request, http_request)
    related = _related_previous_answer(request, machine)
    result = _run_and_persist(request, session_id, machine, user_id=user_id)
    return {**result, "related_previous_answer": related}


def _sse_event(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


@router.post("/ask/stream")
def scoped_ask_stream(request: ScopedAskRequest, http_request: Request) -> StreamingResponse:
    """Same behavior as /ask, but streams live progress events over SSE.

    Event shapes sent as `data: {...}` lines:
        {"type": "progress", "stage": ..., "message": ..., ...}
        {"type": "answer_delta", "text": ...}   - display-only partial answer
        {"type": "answer_mode", "mode": "full_search"} - reroute deltas to panel
        {"type": "answer_discard"}              - streamed text was superseded
        {"type": "reasoning", "text": ...}      - the model's own commentary
        {"type": "result", ...}                 - same shape /ask returns
        {"type": "error", "message": ...}
    """

    machine, session_id, user_id = _start_turn(request, http_request)
    related = _related_previous_answer(request, machine)

    def generate():
        # The pipeline makes blocking model and database calls, so it runs on a
        # worker thread; progress events are relayed through a queue to this
        # generator, which streams them to the client as they arrive.
        event_queue: queue.Queue = queue.Queue()

        def worker() -> None:
            try:
                result = _run_and_persist(
                    request,
                    session_id,
                    machine,
                    on_progress=lambda event: event_queue.put({"type": "progress", **event}),
                    # Display-only fragments. The final "result" event still
                    # carries the canonical answer with citations renumbered,
                    # and the client replaces the streamed text with it.
                    on_answer_delta=lambda text: event_queue.put({"type": "answer_delta", "text": text}),
                    on_reasoning=lambda text: event_queue.put({"type": "reasoning", "text": text}),
                    # answer_mode / answer_discard, already shaped for the client.
                    on_stream_event=event_queue.put,
                    user_id=user_id,
                )
                event_queue.put({"type": "result", **result, "related_previous_answer": related})
            except Exception as exc:
                event_queue.put({"type": "error", "message": str(exc)})
            finally:
                event_queue.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        while True:
            item = event_queue.get()
            if item is None:
                break
            yield _sse_event(item)
        thread.join()

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/sessions/{machine_number}")
def sessions_for_machine(machine_number: str) -> dict:
    """List prior sessions for a machine without loading them by default."""

    machine = get_machine_by_number(machine_number)
    if not machine:
        raise HTTPException(status_code=404, detail="Machine number was not found.")
    return {"machine": machine, "sessions": list_sessions(machine["machine_id"])}


@router.get("/sessions/{machine_number}/{session_id}")
def session_detail(machine_number: str, session_id: int, request: Request) -> dict:
    """Load one selected session so history is an explicit user action."""

    machine = get_machine_by_number(machine_number)
    if not machine:
        raise HTTPException(status_code=404, detail="Machine number was not found.")

    session = load_session(session_id, machine["machine_id"])
    if not session:
        raise HTTPException(status_code=404, detail="Session was not found for this machine.")

    # Reattach this user's own ratings so reopening a session shows the thumbs
    # already selected rather than inviting them to rate the same answer twice.
    ratings = feedback_for_session(session_id, user_id=request.session.get("user_id"))
    for answer in session.get("answers", []):
        answer["feedback"] = ratings.get(answer["id"])

    return {"machine": machine, "session": session}


class FeedbackRequest(BaseModel):
    """A technician's rating of one answer.

    `reasons` and `comment` are optional and arrive on a second call: the
    thumb is recorded on click, and the detail only if they bother to add it.
    """

    answer_id: int
    rating: str = Field(pattern="^(up|down)$")
    reasons: list[str] = Field(default_factory=list)
    comment: str | None = Field(default=None, max_length=1000)


def _identified_user_id(request: Request) -> int | None:
    """The signed-in user; anonymous asks are refused while auth is on.

    The middleware should make the refusal unreachable. It exists so that if a
    future gate change lets an unauthenticated ask through, the failure is a
    loud 401 rather than a silently unattributed session.
    """

    user_id = request.session.get("user_id")
    if user_id is None and get_settings().auth_required:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user_id


def _session_title(question: str) -> str:
    """First question, flattened and capped, as the session's display name."""

    title = " ".join(question.split())
    return title[:77] + "..." if len(title) > 80 else title


@router.post("/feedback")
def submit_feedback(payload: FeedbackRequest, request: Request) -> dict:
    """Record a thumbs up/down on an answer, or update an existing one."""

    if payload.rating not in RATINGS:
        raise HTTPException(status_code=400, detail="Unknown rating.")

    result = record_feedback(
        answer_id=payload.answer_id,
        rating=payload.rating,
        user_id=request.session.get("user_id"),
        reasons=payload.reasons,
        comment=payload.comment,
    )
    if not result["stored"] and result.get("reason") == "unknown_answer":
        raise HTTPException(status_code=404, detail="Answer was not found.")
    return result


def _retrieval_scope(machine: dict, mode: str = "machine_search") -> dict:
    """Describe the enforced retrieval scope for UI and audit displays."""

    return {
        "mode": mode,
        "machine_id": machine["machine_id"],
        "machine_number": machine["machine_number"],
        # Every search merges in the general troubleshooting folder alongside
        # the selected machine (see retriever.retrieval_scope).
        "includes_general": True,
    }
