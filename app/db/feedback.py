"""Thumbs up/down on answers.

The rating on its own says little - "37 down" tells nobody what to change.
What makes it actionable is that every rating joins straight onto the
diagnostics already recorded for that answer in `retrieval_runs` and
`retrieval_passes`: which judge verdict it got, how many retrieval passes it
needed, which sources it cited. So this table stores almost nothing beyond
the vote itself.

Ratings are deliberately cheap to give. A downvote is recorded the moment it
is clicked; the "what was wrong?" detail is optional and comes after. Asking
for the reason first would mean complete reasons and sparse ratings, which is
the wrong trade - the rating is the signal that scales.
"""

from app.db.database import connect


# Fixed vocabulary for "what was wrong". Free text alone cannot be counted, and
# these are the failure modes the pipeline can actually be tuned against:
#
#   wrong_info    - retrieval found something and the model misread it
#   not_in_answer - the manual covers it but the answer did not surface it
#                   (a retrieval problem, not a generation one)
#   too_vague     - correct but not actionable at the machine
#   wrong_machine - scope leak, which should be impossible and is worth an alert
FEEDBACK_REASONS = {
    "wrong_info": "Information is wrong",
    "not_in_answer": "It is in the manual but the answer missed it",
    "too_vague": "Too vague to act on",
    "wrong_machine": "Not about this machine",
    "other": "Something else",
}

RATINGS = ("up", "down")


def _clean_reasons(reasons: list[str] | None) -> str | None:
    """Keep only recognised reason codes, in a stable order, as a CSV string."""

    if not reasons:
        return None
    kept = [code for code in FEEDBACK_REASONS if code in set(reasons)]
    return ",".join(kept) or None


def record_feedback(
    answer_id: int,
    rating: str,
    user_id: int | None = None,
    reasons: list[str] | None = None,
    comment: str | None = None,
) -> dict:
    """Record or update one person's rating of one answer.

    session_id and machine_id are read from the answer rather than trusted from
    the caller - taking them from the request would let a client attribute
    feedback to a machine the answer does not belong to.
    """

    if rating not in RATINGS:
        raise ValueError(f"rating must be one of {RATINGS}")

    with connect() as connection:
        answer = connection.execute(
            "SELECT id, session_id, machine_id FROM answers WHERE id = ?", (answer_id,)
        ).fetchone()
        if not answer:
            return {"stored": False, "reason": "unknown_answer"}

        cleaned_reasons = _clean_reasons(reasons)
        trimmed_comment = (comment or "").strip()[:1000] or None

        existing = connection.execute(
            "SELECT id FROM answer_feedback WHERE answer_id = ? AND user_id IS ?",
            (answer_id, user_id),
        ).fetchone()

        if existing:
            connection.execute(
                """
                UPDATE answer_feedback
                SET rating = ?, reasons = ?, comment = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (rating, cleaned_reasons, trimmed_comment, existing["id"]),
            )
            feedback_id = existing["id"]
        else:
            feedback_id = connection.insert_returning_id(
                """
                INSERT INTO answer_feedback (answer_id, session_id, machine_id, user_id, rating, reasons, comment)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    answer_id,
                    answer["session_id"],
                    answer["machine_id"],
                    user_id,
                    rating,
                    cleaned_reasons,
                    trimmed_comment,
                ),
            )

    return {
        "stored": True,
        "feedback_id": feedback_id,
        "answer_id": answer_id,
        "rating": rating,
        "reasons": cleaned_reasons.split(",") if cleaned_reasons else [],
        "comment": trimmed_comment,
    }


def feedback_for_session(session_id: int, user_id: int | None = None) -> dict[int, dict]:
    """This user's ratings for a session's answers, keyed by answer_id.

    Used when reopening a past session so the thumbs come back already
    selected. Scoped to one user on purpose: shop-floor devices are shared, and
    showing someone else's rating as though it were yours would mislead.
    """

    with connect() as connection:
        rows = connection.execute(
            """
            SELECT answer_id, rating, reasons, comment
            FROM answer_feedback
            WHERE session_id = ? AND user_id IS ?
            ORDER BY answer_id
            """,
            (session_id, user_id),
        ).fetchall()

    return {
        row["answer_id"]: {
            "rating": row["rating"],
            "reasons": row["reasons"].split(",") if row["reasons"] else [],
            "comment": row["comment"],
        }
        for row in rows
    }
