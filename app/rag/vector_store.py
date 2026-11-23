"""Chunk storage and search over the SQLite index.

Every other module in app/rag imports storage/search from here, so swapping in
a dedicated vector database stays a one-file change.

Vector search is a brute-force cosine scan over the scoped machine's chunks.
That is exact, simple, and fast enough for a corpus of a few hundred manuals;
an ANN index only pays for itself well past that.
"""

import hashlib

import numpy as np

from app.db.database import connect
from app.rag.embeddings import embed_texts

_TABLE = "document_chunks"
_COLUMNS = (
    "chunk_id, machine_id, source_path, source_file, relative_path, "
    "document_type, page_start, page_end, content"
)


def _pk(chunk_id: str) -> str:
    # Fixed-width key regardless of how long a chunk_id (machine/path/page) gets.
    return hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()


def _to_blob(vector: list[float]) -> bytes:
    array = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(array)
    # Stored unit-length so cosine similarity is a plain dot product at query time.
    return (array / norm if norm else array).tobytes()


def ensure_collection() -> None:
    """The schema is created by init_db(); kept so ingest reads the same."""


def delete_by_source_path(source_path: str) -> None:
    with connect() as conn:
        conn.execute(f"DELETE FROM {_TABLE} WHERE source_path = ?", (source_path,))


def upsert_chunks(chunks: list[dict]) -> None:
    if not chunks:
        return

    embeddings = embed_texts([chunk["text"] for chunk in chunks])
    rows = []
    for chunk, embedding in zip(chunks, embeddings):
        metadata = chunk["metadata"]
        rows.append(
            (
                _pk(metadata["chunk_id"]),
                metadata["chunk_id"],
                metadata["machine_id"],
                metadata["source_path"],
                metadata.get("source_file"),
                metadata.get("relative_path"),
                metadata.get("document_type"),
                metadata.get("page_start"),
                metadata.get("page_end"),
                chunk["text"],
                _to_blob(embedding),
            )
        )

    with connect() as conn:
        conn.executemany(
            f"""
            INSERT OR REPLACE INTO {_TABLE}
                (id, chunk_id, machine_id, source_path, source_file, relative_path,
                 document_type, page_start, page_end, content, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def _machine_filter_sql(machine_id: str | list[str] | None) -> tuple[str, list[str]]:
    """Build a parameterized machine_id filter clause.

    Accepts a list so one call can search the selected machine plus the
    general/machine-agnostic folder without loosening scope to "all machines".
    """

    if not machine_id:
        return "", []
    if isinstance(machine_id, list):
        placeholders = ", ".join("?" for _ in machine_id)
        return f"machine_id IN ({placeholders})", list(machine_id)
    return "machine_id = ?", [machine_id]


def _row_to_dict(row, score: float | None = None) -> dict:
    return {
        "id": row["chunk_id"],
        "text": row["content"],
        "score": 0.0 if score is None else score,
        "metadata": {
            "chunk_id": row["chunk_id"],
            "machine_id": row["machine_id"],
            "source_path": row["source_path"],
            "source_file": row["source_file"],
            "relative_path": row["relative_path"],
            "document_type": row["document_type"],
            "page_start": row["page_start"],
            "page_end": row["page_end"],
        },
    }


def search_chunks(query: str, machine_id: str | list[str] | None, top_k: int = 10) -> list[dict]:
    where_sql, params = _machine_filter_sql(machine_id)
    where_clause = f"WHERE {where_sql}" if where_sql else ""
    with connect() as conn:
        rows = conn.execute(
            f"SELECT {_COLUMNS}, embedding FROM {_TABLE} {where_clause}", params
        ).fetchall()
    if not rows:
        return []

    query_vector = np.frombuffer(_to_blob(embed_texts([query])[0]), dtype=np.float32)
    matrix = np.stack([np.frombuffer(row["embedding"], dtype=np.float32) for row in rows])
    similarities = matrix @ query_vector
    order = np.argsort(-similarities)[:top_k]
    # Same 0..1 scale as 1/(1 + cosine distance), so diagnostics read the same
    # whichever backend produced them.
    return [_row_to_dict(rows[i], score=1 / (2 - float(similarities[i]))) for i in order]


_LIKE_ESCAPE = "\\"


def _like_pattern(term: str) -> str:
    """Build a contains-pattern for LIKE, escaping the wildcard characters.

    Query terms are tokenised with a pattern that allows underscores, and `_`
    is a single-character wildcard in LIKE. An unescaped part number like
    "AL_12" would silently match "AL-12", "AL 12" and "ALx12" as well.
    """

    escaped = (
        term.replace(_LIKE_ESCAPE, _LIKE_ESCAPE * 2)
        .replace("%", f"{_LIKE_ESCAPE}%")
        .replace("_", f"{_LIKE_ESCAPE}_")
    )
    return f"%{escaped}%"


def keyword_candidates(
    needles: list[str],
    machine_id: str | list[str] | None = None,
    limit: int = 400,
) -> list[dict]:
    """Return chunks containing at least one needle, coarsely ranked in SQL.

    The ranking here is deliberately coarse (how many distinct needles appear
    at all, not how often, and no word boundaries). Its only job is to choose
    which few hundred rows are worth returning; app/rag/keyword_search.py then
    does the precise scoring on that much smaller set, where it can afford
    regex word boundaries and rarity weighting.
    """

    if not needles:
        return []

    where_sql, where_params = _machine_filter_sql(machine_id)
    clauses, score_terms, params = [], [], []
    # Filenames and folder paths are searched too: document names carry real
    # signal ("hydraulic schematic", a part number).
    haystack = "(content || ' ' || IFNULL(source_file, '') || ' ' || IFNULL(relative_path, ''))"
    for needle in needles:
        condition = f"{haystack} LIKE ? ESCAPE '{_LIKE_ESCAPE}'"
        clauses.append(condition)
        # One CASE per needle, summed: "how many distinct needles hit this row".
        score_terms.append(f"CASE WHEN {condition} THEN 1 ELSE 0 END")
        params.append(_like_pattern(needle))

    where_parts = [f"({' OR '.join(clauses)})"]
    if where_sql:
        where_parts.append(where_sql)

    sql = f"""
        SELECT {_COLUMNS}, ({' + '.join(score_terms)}) AS needle_hits
        FROM {_TABLE}
        WHERE {' AND '.join(where_parts)}
        ORDER BY needle_hits DESC
        LIMIT ?
    """
    # The scoring expression and the WHERE clause bind the same patterns, so
    # the parameter list carries them twice, scoring first.
    with connect() as conn:
        rows = conn.execute(sql, [*params, *params, *where_params, limit]).fetchall()
    return [_row_to_dict(row) for row in rows]


def load_manifest() -> dict:
    """Every indexed file's signature, keyed by source_path."""

    with connect() as conn:
        rows = conn.execute(
            "SELECT source_path, file_hash, mtime, size, chunk_count FROM index_manifest"
        ).fetchall()
    return {
        row["source_path"]: {
            "file_hash": row["file_hash"],
            "mtime": float(row["mtime"]),
            "size": int(row["size"]),
            "chunk_count": int(row["chunk_count"]),
        }
        for row in rows
    }


def save_manifest_entry(source_path: str, entry: dict) -> None:
    """Record one file's signature. Called per file as indexing progresses."""

    with connect() as conn:
        conn.execute(
            """
            INSERT INTO index_manifest (source_path, file_hash, mtime, size, chunk_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source_path) DO UPDATE SET
                file_hash = excluded.file_hash,
                mtime = excluded.mtime,
                size = excluded.size,
                chunk_count = excluded.chunk_count,
                indexed_at = CURRENT_TIMESTAMP
            """,
            (source_path, entry["file_hash"], entry["mtime"], entry["size"], entry["chunk_count"]),
        )


def delete_manifest_entries(source_paths: list[str]) -> None:
    if not source_paths:
        return
    with connect() as conn:
        conn.executemany(
            "DELETE FROM index_manifest WHERE source_path = ?",
            [(path,) for path in source_paths],
        )


def list_source_paths(machine_id: str | list[str] | None = None) -> list[str]:
    """Every distinct source_path currently in the index.

    Used by the ingest reconciliation pass to find chunks whose document is no
    longer on disk.
    """

    where_sql, params = _machine_filter_sql(machine_id)
    where_clause = f"WHERE {where_sql}" if where_sql else ""
    with connect() as conn:
        rows = conn.execute(f"SELECT DISTINCT source_path FROM {_TABLE} {where_clause}", params).fetchall()
    return [row["source_path"] for row in rows]


def get_chunks(
    machine_id: str | list[str] | None = None,
    source_path: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> list[dict]:
    where_sql, params = _machine_filter_sql(machine_id)
    clauses = [where_sql] if where_sql else []
    if source_path:
        clauses.append("source_path = ?")
        params.append(source_path)
    if page_start is not None:
        clauses.append("page_start >= ?")
        params.append(page_start)
    if page_end is not None:
        # Chunks are single-page, so page_start is enough for range reads.
        clauses.append("page_start <= ?")
        params.append(page_end)

    where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT {_COLUMNS} FROM {_TABLE} {where_clause} ORDER BY source_path, page_start, chunk_id"
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.extend([limit, offset or 0])

    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(row) for row in rows]

