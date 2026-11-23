"""Chat history and answer persistence for machine-scoped sessions."""

import json
import re

from rapidfuzz import fuzz

from app.db.database import connect


def normalize_question(question: str) -> str:
    """Reduce question text to a stable key for simple history matching."""

    return re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()


def create_session(machine_id: str, title: str | None = None, user_id: int | None = None) -> int:
    """Create a chat session tied to one machine and return its database id."""

    with connect() as connection:
        return connection.insert_returning_id(
            "INSERT INTO chat_sessions (machine_id, title, user_id) VALUES (?, ?, ?)",
            (machine_id, title, user_id),
        )


def ensure_session(machine_id: str, session_id: int | None = None, user_id: int | None = None) -> int:
    """Use an existing session when valid, otherwise start a new machine session."""

    if session_id is None:
        return create_session(machine_id, user_id=user_id)

    # Do not let a browser reuse a session id from a different machine.
    with connect() as connection:
        row = connection.execute(
            "SELECT id FROM chat_sessions WHERE id = ? AND machine_id = ?",
            (session_id, machine_id),
        ).fetchone()
    if row:
        return session_id
    return create_session(machine_id, user_id=user_id)


def set_session_title_if_empty(session_id: int, title: str) -> None:
    """Name a session after its first question, exactly once.

    The guard lives in the WHERE clause rather than in a read-then-write so two
    concurrent asks on a fresh session cannot both see "no title yet" and the
    later write win with the second question.
    """

    if not title:
        return
    with connect() as connection:
        connection.execute(
            "UPDATE chat_sessions SET title = ? WHERE id = ? AND (title IS NULL OR title = '')",
            (title, session_id),
        )


def add_message(session_id: int, role: str, content: str) -> int:
    """Store one chat message so the session can be reviewed later."""

    with connect() as connection:
        message_id = connection.insert_returning_id(
            "INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        connection.execute(
            "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (session_id,)
        )
        return message_id


def list_sessions(machine_id: str, limit: int = 25) -> list[dict]:
    """Return recent sessions for one machine so history is loaded intentionally."""

    with connect() as connection:
        rows = connection.execute(
            """
            SELECT
                s.id,
                s.machine_id,
                s.title,
                s.created_at,
                s.updated_at,
                COUNT(m.id) AS message_count,
                MAX(m.created_at) AS last_message_at
            FROM chat_sessions s
            LEFT JOIN chat_messages m ON m.session_id = s.id
            WHERE s.machine_id = ?
            GROUP BY s.id
            ORDER BY COALESCE(MAX(m.created_at), s.created_at) DESC, s.id DESC
            LIMIT ?
            """,
            (machine_id, limit),
        ).fetchall()

    return [
        {
            "id": row["id"],
            "machine_id": row["machine_id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "message_count": row["message_count"],
            "last_message_at": row["last_message_at"] or row["created_at"],
        }
        for row in rows
    ]


def _citation_from_row(row) -> dict:
    """Rebuild the same citation shape the frontend receives from live answers."""

    return {
        "id": row["citation_key"],
        "label": row["label"],
        "source_file": row["source_file"],
        "source_path": row["source_path"],
        "page_start": row["page_start"],
        "page_end": row["page_end"],
        "machine_id": row["machine_id"],
        "preview_endpoint": "/page-preview",
        "pdf_endpoint": "/pdf",
    }


def load_session(session_id: int, machine_id: str) -> dict | None:
    """Load messages and answer citations for an explicitly selected session."""

    with connect() as connection:
        session = connection.execute(
            """
            SELECT id, machine_id, title, created_at, updated_at
            FROM chat_sessions
            WHERE id = ? AND machine_id = ?
            """,
            (session_id, machine_id),
        ).fetchone()
        if not session:
            return None

        messages = connection.execute(
            "SELECT id, role, content, created_at FROM chat_messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        answers = connection.execute(
            """
            SELECT id, question, answer, full_answer, created_at
            FROM answers
            WHERE session_id = ?
            ORDER BY id
            """,
            (session_id,),
        ).fetchall()
        citations = connection.execute(
            """
            SELECT answer_id, citation_key, label, source_file, source_path, page_start, page_end, machine_id
            FROM citations
            WHERE answer_id IN (SELECT id FROM answers WHERE session_id = ?)
            ORDER BY id
            """,
            (session_id,),
        ).fetchall()
        run_rows = connection.execute(
            """
            SELECT answer_id, source_badges_json, reasoning_json, activity_json
            FROM retrieval_runs
            WHERE answer_id IN (SELECT id FROM answers WHERE session_id = ?)
            ORDER BY id
            """,
            (session_id,),
        ).fetchall()

    citations_by_answer: dict[int, list[dict]] = {}
    for row in citations:
        citations_by_answer.setdefault(row["answer_id"], []).append(_citation_from_row(row))
    runs_by_answer = {row["answer_id"]: row for row in run_rows}

    def run_list(answer_id: int, column: str) -> list:
        row = runs_by_answer.get(answer_id)
        return _decode_list(row[column]) if row else []

    return {
        "id": session["id"],
        "machine_id": session["machine_id"],
        "title": session["title"],
        "created_at": session["created_at"],
        "updated_at": session["updated_at"],
        "messages": [
            {"id": row["id"], "role": row["role"], "content": row["content"], "created_at": row["created_at"]}
            for row in messages
        ],
        "answers": [
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "full_answer": row["full_answer"],
                "created_at": row["created_at"],
                "citations": citations_by_answer.get(row["id"], []),
                "source_badges": run_list(row["id"], "source_badges_json"),
                # Reopening an old answer should show how it was reached, not
                # just what it concluded - the reasoning is often the part a
                # technician wants to check.
                "reasoning": run_list(row["id"], "reasoning_json"),
                "activity": run_list(row["id"], "activity_json"),
            }
            for row in answers
        ],
    }


def _decode_list(raw: str | None) -> list:
    """Load a JSON list column; a malformed row must not break the whole session."""

    try:
        value = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def add_answer(
    session_id: int,
    machine_id: str,
    question: str,
    answer: str,
    citations: list[dict],
    user_id: int | None = None,
    full_answer: str | None = None,
) -> int:
    """Store an answer and its structured citations for reuse and review.

    full_answer is the complete pre-summary text of a full-search turn (the
    More-detail reveal); None on light-path turns.
    """

    with connect() as connection:
        answer_id = connection.insert_returning_id(
            """
            INSERT INTO answers (session_id, machine_id, question, normalized_question, answer, full_answer, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, machine_id, question, normalize_question(question), answer, full_answer, user_id),
        )
        connection.executemany(
            """
            INSERT INTO citations (
                answer_id, citation_key, label, source_file, source_path, page_start, page_end, machine_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    answer_id,
                    citation["id"],
                    citation["label"],
                    citation.get("source_file"),
                    citation.get("source_path"),
                    citation.get("page_start"),
                    citation.get("page_end"),
                    citation.get("machine_id"),
                )
                for citation in citations
            ],
        )
        return answer_id


# token_set_ratio scores 100 whenever one question's tokens are a subset of the
# other's, so a low threshold matched "oil change interval" against "what is
# the oil change interval for the hydraulic pump on the clamp side" as a
# certainty. The match is a hint shown beside a real answer, so it only has to
# be worth a glance - it should not be generous.
REUSE_MIN_SCORE = 88


def find_reuse_candidate(machine_id: str, question: str, exclude_session_id: int | None = None) -> dict | None:
    """Find a prior answer for the same machine that may answer this question.

    This is a hint, not a gate. It is shown alongside a real answer instead of
    in place of one: interrupting every question with "use the previous
    answer?" on a fuzzy match that is often wrong is noise.
    """

    normalized = normalize_question(question)
    with connect() as connection:
        # Any downvote excludes an answer from reuse, and upvotes do not
        # rehabilitate it. A downvote is the exact failure signal reuse exists
        # to avoid resurfacing; over-excluding a good answer that caught a stray
        # downvote costs almost nothing, under-excluding hands a technician a
        # known-bad number.
        rows = connection.execute(
            """
            SELECT id, question, normalized_question, answer, created_at
            FROM answers
            WHERE machine_id = ?
              AND (? IS NULL OR session_id <> ?)
              AND NOT EXISTS (
                  SELECT 1 FROM answer_feedback f
                  WHERE f.answer_id = answers.id AND f.rating = 'down'
              )
            ORDER BY created_at DESC, id DESC
            LIMIT 200
            """,
            (machine_id, exclude_session_id, exclude_session_id),
        ).fetchall()

    best_row = None
    best_score = 0
    for row in rows:
        # Token-set matching catches reordered wording like "hydraulic alarm reset".
        score = fuzz.token_set_ratio(normalized, row["normalized_question"])
        if score > best_score:
            best_score = score
            best_row = row

    if not best_row or best_score < REUSE_MIN_SCORE:
        return None

    return {
        "answer_id": best_row["id"],
        "question": best_row["question"],
        "answer_preview": best_row["answer"][:700],
        "match_score": round(float(best_score), 2),
        "confidence": "strong" if best_score >= 95 else "medium",
        "created_at": best_row["created_at"],
        "message": "A similar answer already exists for this machine.",
    }
