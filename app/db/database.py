"""Application database: one SQLite file for the chunk index, users and history.

The connection wrapper keeps the call sites small: rows support both
row["column"] and row.column access, and inserts hand back the new id.
"""

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.config import get_settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS app_users (
    user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE,
    display_name  TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    is_active     INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login_at TEXT
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id            TEXT PRIMARY KEY,
    chunk_id      TEXT NOT NULL,
    machine_id    TEXT NOT NULL,
    source_path   TEXT NOT NULL,
    source_file   TEXT,
    relative_path TEXT,
    document_type TEXT,
    page_start    INTEGER,
    page_end      INTEGER,
    content       TEXT NOT NULL,
    embedding     BLOB NOT NULL,
    created_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_chunks_machine ON document_chunks(machine_id);
CREATE INDEX IF NOT EXISTS ix_chunks_source ON document_chunks(source_path);

CREATE TABLE IF NOT EXISTS index_manifest (
    source_path TEXT PRIMARY KEY,
    file_hash   TEXT NOT NULL,
    mtime       REAL NOT NULL,
    size        INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    indexed_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id TEXT NOT NULL,
    title      TEXT,
    user_id    INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_sessions_machine ON chat_sessions(machine_id);

CREATE TABLE IF NOT EXISTS chat_messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES chat_sessions(id),
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_messages_session ON chat_messages(session_id);

CREATE TABLE IF NOT EXISTS answers (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id            INTEGER NOT NULL REFERENCES chat_sessions(id),
    machine_id            TEXT NOT NULL,
    question              TEXT NOT NULL,
    normalized_question   TEXT NOT NULL,
    answer                TEXT NOT NULL,
    full_answer           TEXT,
    user_id               INTEGER,
    created_at            TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_answers_machine ON answers(machine_id);
CREATE INDEX IF NOT EXISTS ix_answers_session ON answers(session_id);

CREATE TABLE IF NOT EXISTS citations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    answer_id    INTEGER NOT NULL REFERENCES answers(id),
    citation_key TEXT NOT NULL,
    label        TEXT,
    source_file  TEXT,
    source_path  TEXT,
    page_start   INTEGER,
    page_end     INTEGER,
    machine_id   TEXT
);
CREATE INDEX IF NOT EXISTS ix_citations_answer ON citations(answer_id);

CREATE TABLE IF NOT EXISTS retrieval_runs (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    answer_id          INTEGER REFERENCES answers(id),
    session_id         INTEGER,
    machine_id         TEXT,
    question           TEXT,
    context_mode       TEXT,
    docs_answer_found  INTEGER,
    evidence_status    TEXT,
    termination_reason TEXT,
    final_judge_json   TEXT,
    tool_path_json     TEXT,
    source_badges_json TEXT,
    reasoning_json     TEXT,
    activity_json      TEXT,
    input_tokens       INTEGER,
    output_tokens      INTEGER,
    duration_ms        INTEGER,
    created_at         TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_runs_answer ON retrieval_runs(answer_id);

CREATE TABLE IF NOT EXISTS retrieval_passes (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                 INTEGER NOT NULL REFERENCES retrieval_runs(id),
    pass_number            INTEGER,
    max_chunks_per_doc     INTEGER,
    max_total_chunks       INTEGER,
    candidate_count        INTEGER,
    new_chunk_count        INTEGER,
    seen_chunk_count       INTEGER,
    evidence_chunk_count   INTEGER,
    judge_has_answer       INTEGER,
    judge_confidence       TEXT,
    judge_answerable_scope TEXT,
    judge_reason           TEXT,
    judge_json             TEXT
);

CREATE TABLE IF NOT EXISTS answer_feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    answer_id  INTEGER NOT NULL REFERENCES answers(id),
    session_id INTEGER,
    machine_id TEXT,
    user_id    INTEGER,
    rating     TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    reasons    TEXT,
    comment    TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_feedback_answer ON answer_feedback(answer_id);
"""


class Row(dict):
    """A result row supporting both row["col"] and row.col access."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _row_factory(cursor: sqlite3.Cursor, raw: tuple) -> Row:
    return Row(zip((column[0] for column in cursor.description), raw))


class Connection:
    """Thin wrapper giving the call sites the shape they expect."""

    def __init__(self, raw: sqlite3.Connection):
        self._raw = raw

    def execute(self, sql: str, params: tuple | list = ()) -> sqlite3.Cursor:
        return self._raw.execute(sql, list(params))

    def executemany(self, sql: str, seq_of_params) -> None:
        rows = [list(params) for params in seq_of_params]
        if rows:
            self._raw.executemany(sql, rows)

    def insert_returning_id(self, sql: str, params: tuple | list = ()) -> int:
        return int(self._raw.execute(sql, list(params)).lastrowid)


_init_lock = threading.Lock()
_initialized_path: Path | None = None


def _db_path() -> Path:
    return get_settings().app_db_path


@contextmanager
def connect() -> Iterator[Connection]:
    """Open a connection, committing on a clean exit and rolling back on error.

    A connection per unit of work is cheap for SQLite and sidesteps sharing one
    connection across the request threads the streaming endpoint uses.
    """

    init_db()
    raw = sqlite3.connect(_db_path(), timeout=30)
    raw.row_factory = _row_factory
    raw.execute("PRAGMA foreign_keys = ON")
    try:
        yield Connection(raw)
        raw.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()


def init_db() -> None:
    """Create the database file and schema if needed. Safe to call repeatedly."""

    global _initialized_path
    path = _db_path()
    if _initialized_path == path and path.exists():
        return
    with _init_lock:
        if _initialized_path == path and path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = sqlite3.connect(path, timeout=30)
        try:
            # WAL lets the indexer write while the app keeps answering questions.
            raw.execute("PRAGMA journal_mode = WAL")
            raw.executescript(SCHEMA)
            raw.commit()
        finally:
            raw.close()
        _initialized_path = path
