"""Build or refresh the document index.

    python scripts/index_documents.py                    # everything under DOCS_ROOT
    python scripts/index_documents.py --scope general    # only the general library
    python scripts/index_documents.py --scope "IMM 101 - ACME (PX 200-50) SN12345"
    python scripts/index_documents.py --force            # re-embed unchanged files too
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.rag.ingest import index_documents  # noqa: E402


def _print_progress(current: int, total: int, doc: dict) -> None:
    print(f"[{current}/{total}] {doc['machine_id']} :: {doc['source_file']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index machine documents.")
    parser.add_argument("--docs-root", help="Override DOCS_ROOT for this run.")
    parser.add_argument("--force", action="store_true", help="Re-index unchanged files.")
    parser.add_argument("--limit-files", type=int, help="Only index the first N discovered files.")
    parser.add_argument(
        "--scope",
        default="all",
        help="'all' (default), 'general' (the shared library), or one machine folder name.",
    )
    parser.add_argument("--quiet", action="store_true", help="Do not print per-file progress.")
    args = parser.parse_args()

    if args.docs_root:
        os.environ["DOCS_ROOT"] = args.docs_root
        get_settings.cache_clear()

    stats = index_documents(
        force=args.force,
        limit_files=args.limit_files,
        scope=args.scope,
        progress_callback=None if args.quiet else _print_progress,
    )
    print(stats)


if __name__ == "__main__":
    main()
