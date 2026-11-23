"""Persistence for retrieval diagnostics.

Two audiences: technicians reopening an old answer want to see how it was
reached (the reasoning and activity trail), and whoever tunes the pipeline
needs an audit trail of how hard the assistant had to search before it
answered - each retrieval pass, what the judge concluded, which tools ran.
"""

import json

from app.db.database import connect


def evidence_status(diagnostics: dict) -> str:
    """Describe what the independent evidence judge actually established.

    Intentionally not derived from citations or source badges. A lightweight
    ``search_documents`` turn can cite a real manual page without invoking the
    separate judge, while ``run_full_search`` produces an explicit verdict.
    Keeping those cases distinct stops an unjudged follow-up from being
    counted as a documentation gap.
    """

    judge = diagnostics.get("judge") or {}
    if not judge:
        return "not_checked"
    if diagnostics.get("docs_answer_found"):
        return "confirmed"
    if judge.get("answerable_scope") == "partial" or judge.get("has_answer"):
        return "partial"
    return "not_found"


def record_retrieval_diagnostics(
    *,
    answer_id: int,
    session_id: int,
    machine_id: str,
    question: str,
    diagnostics: dict | None,
    source_badges: list[str] | None = None,
    context_mode: str | None = None,
    duration_ms: int | None = None,
) -> int | None:
    """Store one answer's retrieval trail and return the retrieval run id.

    A mocked response may not have diagnostics; then nothing is recorded
    rather than inventing pass data.
    """

    if not diagnostics:
        return None

    judge = diagnostics.get("judge") or {}
    token_usage = diagnostics.get("token_usage") or {}
    with connect() as connection:
        run_id = connection.insert_returning_id(
            """
            INSERT INTO retrieval_runs (
                answer_id, session_id, machine_id, question, context_mode,
                docs_answer_found, evidence_status, termination_reason,
                final_judge_json, tool_path_json, source_badges_json,
                reasoning_json, activity_json, input_tokens, output_tokens, duration_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                answer_id,
                session_id,
                machine_id,
                question,
                context_mode,
                1 if diagnostics.get("docs_answer_found") else 0,
                evidence_status(diagnostics),
                diagnostics.get("termination_reason"),
                json.dumps(judge),
                json.dumps(diagnostics.get("tool_path") or []),
                json.dumps(source_badges or []),
                json.dumps(diagnostics.get("reasoning") or []),
                json.dumps(diagnostics.get("progress_events") or []),
                token_usage.get("input_tokens"),
                token_usage.get("output_tokens"),
                duration_ms,
            ),
        )

        # Each pass records how retrieval widened and what the judge decided.
        connection.executemany(
            """
            INSERT INTO retrieval_passes (
                run_id, pass_number, max_chunks_per_doc, max_total_chunks,
                candidate_count, new_chunk_count, seen_chunk_count, evidence_chunk_count,
                judge_has_answer, judge_confidence, judge_answerable_scope, judge_reason, judge_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [_pass_row(run_id, retrieval_pass) for retrieval_pass in diagnostics.get("passes") or []],
        )
        return run_id


def _pass_row(run_id: int, retrieval_pass: dict) -> tuple:
    """Flatten one in-memory retrieval pass into the database row shape."""

    judge = retrieval_pass.get("judge") or {}
    return (
        run_id,
        retrieval_pass.get("pass_number"),
        retrieval_pass.get("max_chunks_per_doc"),
        retrieval_pass.get("max_total_chunks"),
        retrieval_pass.get("candidate_count"),
        retrieval_pass.get("new_chunk_count"),
        retrieval_pass.get("seen_chunk_count"),
        retrieval_pass.get("evidence_chunk_count"),
        1 if judge.get("has_answer") else 0,
        judge.get("confidence"),
        judge.get("answerable_scope"),
        judge.get("reason"),
        json.dumps(judge),
    )
