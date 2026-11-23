from app.config import get_settings


def _word_chunks(words: list[str], size: int, overlap: int) -> list[str]:
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        # Overlap keeps important context from being split hard between chunks.
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_pages(pages: list[dict], base_metadata: dict) -> list[dict]:
    settings = get_settings()
    chunks: list[dict] = []
    chunk_index = 1

    for page in pages:
        page_text = (page.get("text") or "").strip()
        if not page_text:
            continue

        # Put machine/file/page in the chunk text so the model sees source context.
        header = f"{base_metadata['machine_id']} | {base_metadata['source_file']} | page {page['page_number']}"
        for text in _word_chunks(
            page_text.split(),
            size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        ):
            metadata = {
                **base_metadata,
                "page_start": page["page_number"],
                "page_end": page["page_number"],
                # Stable ids let reindexing replace chunks from the same PDF/page.
                "chunk_id": (
                    f"{base_metadata['machine_id']}/"
                    f"{base_metadata['relative_path']}/"
                    f"page_{page['page_number']}/chunk_{chunk_index:04d}"
                ),
            }
            chunks.append({"text": f"{header}\n\n{text}", "metadata": metadata})
            chunk_index += 1

    return chunks
