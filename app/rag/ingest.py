import hashlib
import queue
import threading
from pathlib import Path

from app.config import get_settings
from app.rag.chunk import chunk_pages
from app.rag.parse_pdf import parse_pdf
from app.rag.parse_pptx import parse_pptx
from app.rag.parse_xlsx import parse_xlsx
from app.rag.vector_store import (
    delete_by_source_path,
    delete_manifest_entries,
    ensure_collection,
    list_source_paths,
    load_manifest,
    save_manifest_entry,
    upsert_chunks,
)


# A handful of malformed/pathological PDFs can make pymupdf spin for many
# minutes on a file that's a few hundred KB - without a guard, one bad file
# hangs the entire corpus run instead of just failing that one file.
PARSE_TIMEOUT_SECONDS = 120


def _parse_with_timeout(parser, path):
    """Run parser(path) on a daemon thread and give up after
    PARSE_TIMEOUT_SECONDS instead of waiting for it indefinitely.

    Deliberately does NOT use concurrent.futures.ThreadPoolExecutor as a
    context manager - its __exit__ calls shutdown(wait=True), which blocks
    until the worker thread finishes regardless of any future.result()
    timeout, silently defeating the whole point of this guard. A daemon
    thread we simply stop waiting on has no such problem: if it's still
    spinning after the timeout, it's abandoned (and killed for good when the
    process exits) rather than blocking the rest of the corpus run.
    """

    result: queue.Queue = queue.Queue(maxsize=1)

    def worker():
        try:
            result.put(("ok", parser(path)))
        except Exception as exc:  # noqa: BLE001 - re-raised on the caller's side
            result.put(("error", exc))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(PARSE_TIMEOUT_SECONDS)
    if thread.is_alive():
        raise RuntimeError(
            f"Parsing exceeded {PARSE_TIMEOUT_SECONDS}s, likely a malformed file - skipped"
        )

    status, payload = result.get()
    if status == "error":
        raise payload
    return payload


PARSERS_BY_SUFFIX = {
    ".pdf": parse_pdf,
    ".xlsx": parse_xlsx,
    ".pptx": parse_pptx,
}
# Discovered but intentionally unsupported: legacy .ppt/.doc/.xls are binary
# formats the parsers cannot open. Files with this suffix are skipped with a
# clear reason instead of failing indexing with a raw exception.
UNSUPPORTED_SUFFIXES = {".ppt", ".doc", ".xls"}


DOCUMENT_TYPES = {
    # These labels are simple metadata hints for filtering and source badges later.
    "manual": ["manual", "tm", "service"],
    "maintenance": ["maintenance", "maint"],
    "electrical": ["electrical", "electric"],
    "hydraulic": ["hydraulic"],
    "parts": ["parts", "spare"],
    "schematic": ["schematic", "print", "drawing"],
    "safety": ["safety", "warning"],
}


def infer_document_type(path: Path) -> str:
    text = " ".join(path.parts).lower()
    for doc_type, needles in DOCUMENT_TYPES.items():
        if any(needle in text for needle in needles):
            return doc_type
    return "unknown"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        # Read in blocks so large manuals do not need to load into memory.
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _discover_files(folder: Path) -> list[Path]:
    """Return every file under folder whose suffix we know how to parse."""

    return sorted(
        path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in PARSERS_BY_SUFFIX
    )


def discover_documents(root: str | Path | None = None, only_machine: str | None = None) -> list[dict]:
    docs_root = Path(root) if root else get_settings().docs_root
    if not docs_root.exists():
        return []

    # Underscore-prefixed folders are not machines. The general troubleshooting
    # library lives at DOCS_ROOT/_general by default - without this guard it
    # would be indexed twice: once here as a machine literally named
    # "_general", and again by discover_general_documents().
    machine_dirs = sorted(
        path for path in docs_root.iterdir() if path.is_dir() and not path.name.startswith("_")
    )
    if only_machine:
        # Scoped reindex: only walk the one machine folder instead of every
        # machine's files, so adding one doc doesn't touch the whole corpus.
        machine_dirs = [path for path in machine_dirs if path.name == only_machine]

    documents = []
    for machine_dir in machine_dirs:
        # Each top-level folder is treated as one machine scope.
        for file_path in _discover_files(machine_dir):
            relative_path = file_path.relative_to(machine_dir)
            documents.append(
                {
                    "machine_id": machine_dir.name,
                    "machine_name": machine_dir.name,
                    # `path` is the real on-disk location, used to read the file
                    # during this run. `source_path` is the *stored* identifier
                    # and is deliberately DOCS_ROOT-relative with forward
                    # slashes, so a row indexed on one machine resolves
                    # identically on another (a Windows laptop vs. a Linux
                    # container). An absolute path would break citation
                    # previews the moment the index moved.
                    "path": str(file_path),
                    "source_file": file_path.name,
                    "source_path": (Path(machine_dir.name) / relative_path).as_posix(),
                    "relative_path": str(relative_path),
                    "document_type": infer_document_type(relative_path),
                }
            )
    return documents


def discover_general_documents() -> list[dict]:
    """Discover machine-agnostic documents tagged with the general machine id.

    Unlike discover_documents, this folder is flat in meaning (everything under
    it belongs to one shared scope), and it is not required to exist.
    """

    settings = get_settings()
    general_root = settings.general_root
    if not general_root.exists():
        return []

    documents = []
    for file_path in _discover_files(general_root):
        relative_path = file_path.relative_to(general_root)
        documents.append(
            {
                "machine_id": settings.general_machine_id,
                "machine_name": "General Troubleshooting",
                "path": str(file_path),
                # Relative to the general root rather than DOCS_ROOT; the
                # preview resolver tries each allowed root in turn, so it does
                # not need to know which one a given row came from.
                "source_file": file_path.name,
                "source_path": relative_path.as_posix(),
                "relative_path": str(relative_path),
                "document_type": infer_document_type(relative_path),
            }
        )
    return documents


def _is_under(child: Path, parent: Path) -> bool:
    """True when `child` sits inside `parent`, for de-duplicating folder walks."""

    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def _unsupported_files(folder: Path | None) -> list[str]:
    """List files with a known-but-unsupported extension, for visibility only."""

    if not folder or not folder.exists():
        return []
    return sorted(
        str(path)
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in UNSUPPORTED_SUFFIXES
    )


def index_documents(
    force: bool = False,
    limit_files: int | None = None,
    scope: str = "all",
    progress_callback=None,
) -> dict:
    """Index documents into the vector store.

    scope controls which files are even discovered/hashed, not just which
    get re-embedded:
        "all"      - every machine folder plus the general folder (default)
        "general"  - only the general/machine-agnostic folder
        <folder>   - only that one machine's folder (the machine_id folder
                     name under DOCS_ROOT)

    Scoping avoids re-hashing the entire corpus when one machine gets a new
    document.

    progress_callback, if given, is called as progress_callback(current, total, doc)
    before each file is processed, so callers can show live status instead of
    waiting for the final stats dict.
    """

    ensure_collection()
    manifest = load_manifest()
    settings = get_settings()

    if scope == "general":
        docs = discover_general_documents()
        unsupported = _unsupported_files(settings.general_root)
    elif scope == "all":
        docs = discover_documents() + discover_general_documents()
        # _unsupported_files(docs_root) already walks every folder beneath it,
        # including a general folder that lives inside it.
        unsupported = _unsupported_files(settings.docs_root)
        if not _is_under(settings.general_root, settings.docs_root):
            unsupported += _unsupported_files(settings.general_root)
    else:
        # Scope is a specific machine folder name.
        docs = discover_documents(only_machine=scope)
        unsupported = _unsupported_files(settings.docs_root / scope)

    if limit_files:
        docs = docs[:limit_files]

    stats = {
        "scope": scope,
        "indexed_files": 0,
        "skipped_files": 0,
        "failed_files": 0,
        "failures": [],
        # Files found but not indexed because the format isn't supported
        # (e.g. legacy .ppt). Surfaced so a silent gap doesn't go unnoticed.
        "unsupported_files": unsupported,
    }
    total = len(docs)
    indexed_paths: set[str] = set()
    for index, doc in enumerate(docs, start=1):
        if progress_callback:
            progress_callback(index, total, doc)

        path = Path(doc["path"])
        key = doc["source_path"]
        parser = PARSERS_BY_SUFFIX.get(path.suffix.lower())
        try:
            if parser is None:
                raise RuntimeError(f"No parser registered for suffix {path.suffix!r}")

            file_stat = path.stat()
            manifest_entry = manifest.get(key, {})
            # Cheap pre-check first: if mtime and size match the last indexed
            # run, skip without reading the file at all. Hashing every
            # already-indexed file just to confirm "no change" would make a
            # full reindex scale with corpus size instead of with what changed.
            unchanged_by_stat = (
                not force
                and manifest_entry.get("mtime") == file_stat.st_mtime
                and manifest_entry.get("size") == file_stat.st_size
            )
            if unchanged_by_stat:
                stats["skipped_files"] += 1
                indexed_paths.add(key)
                continue

            current_hash = file_hash(path)
            if not force and manifest_entry.get("file_hash") == current_hash:
                # Content is actually unchanged (e.g. file was just touched/copied).
                # Update the cheap signature so next run can skip via stat alone.
                save_manifest_entry(
                    key,
                    {**manifest_entry, "mtime": file_stat.st_mtime, "size": file_stat.st_size},
                )
                stats["skipped_files"] += 1
                indexed_paths.add(key)
                continue

            # Parse BEFORE deleting. Deleting first means a parse failure on a
            # changed file destroys its existing chunks with nothing to replace
            # them - the document vanishes from search because one re-index
            # attempt hit a bad byte. Parsing first turns that into "the old
            # chunks survive and the failure is reported".
            pages = _parse_with_timeout(parser, path)
            chunks = chunk_pages(pages, doc)
            # Delete old chunks so changed files do not leave stale pages.
            delete_by_source_path(key)

            upsert_chunks(chunks)
            save_manifest_entry(
                key,
                {
                    "file_hash": current_hash,
                    "mtime": file_stat.st_mtime,
                    "size": file_stat.st_size,
                    "chunk_count": len(chunks),
                },
            )
            stats["indexed_files"] += 1
            indexed_paths.add(key)
        except Exception as exc:
            stats["failed_files"] += 1
            stats["failures"].append({"source_path": key, "error": str(exc)})

    stats["removed_documents"] = _remove_deleted_documents(
        scope, indexed_paths, stats, limited=bool(limit_files)
    )
    return stats


# A reconciliation run that would delete more than this share of the index is
# treated as a symptom rather than an instruction. Deleting a few superseded
# manuals is routine; deleting most of the corpus means the source went away,
# not that the documents did.
MAX_ORPHAN_SHARE = 0.20


def _remove_deleted_documents(
    scope: str, indexed_paths: set[str], stats: dict, limited: bool
) -> list[str]:
    """Delete chunks whose source document is no longer on disk.

    Without this, removing a document from the share left its chunks in the
    index forever: still searchable, still citable, and pointing at a file the
    citation viewer can no longer open. A superseded manual would go on
    answering questions alongside the one that replaced it.

    The guards matter more than the deletion. This runs off "which files did we
    just see", so anything that legitimately narrows that set looks identical
    to mass deletion:

      * a scoped run only walks one machine's folder
      * --limit-files deliberately truncates the list
      * a failed parse means "could not read this time", not "gone"
      * an unmounted DOCS_ROOT yields zero documents - if it is a network
        share, that is a real failure mode, and without a guard one run
        against a missing mount would wipe the whole index
    """

    def skip(reason: str) -> list[str]:
        stats["reconciliation_skipped"] = reason
        return []

    if scope != "all":
        return skip(f"scope was {scope!r}, which only walks part of the corpus.")
    if limited:
        return skip("--limit-files truncated the file list.")
    if stats["failed_files"]:
        return skip(
            f"{stats['failed_files']} file(s) failed to index; a parse failure cannot "
            "be told apart from a deleted document."
        )
    if not indexed_paths:
        return skip(
            "no documents were discovered at all - DOCS_ROOT is empty or not mounted."
        )

    known = set(list_source_paths())
    orphans = sorted(known - indexed_paths)
    if known and len(orphans) / len(known) > MAX_ORPHAN_SHARE:
        return skip(
            f"{len(orphans)} of {len(known)} indexed documents look deleted "
            f"(over {MAX_ORPHAN_SHARE:.0%}). Refusing to act on that automatically - "
            "if the removal is genuine, re-index that machine's scope directly."
        )

    for source_path in orphans:
        delete_by_source_path(source_path)
    delete_manifest_entries(orphans)
    return orphans
