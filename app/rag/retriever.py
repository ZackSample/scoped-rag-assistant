from app.config import get_settings
from app.rag.keyword_search import keyword_search
from app.rag.vector_store import search_chunks


# Reciprocal rank fusion constant. 60 is the value from the original RRF paper
# and the de-facto default; it damps the difference between the top few ranks
# so a single method's #1 hit cannot dominate outright.
RRF_K = 60


def _diversify_by_source(
    chunks: list[dict],
    top_k: int,
    max_chunks_per_source: int,
) -> list[dict]:
    source_counts: dict[str, int] = {}
    selected: list[dict] = []
    deferred: list[dict] = []

    for chunk in chunks:
        # Cap each document early so one long manual cannot crowd out every source.
        source = chunk["metadata"].get("source_path") or chunk["metadata"].get("source_file") or ""
        count = source_counts.get(source, 0)
        if count < max_chunks_per_source:
            selected.append(chunk)
            source_counts[source] = count + 1
        else:
            deferred.append(chunk)

        if len(selected) >= top_k:
            return selected

    for chunk in deferred:
        # If we still need context, backfill from overrepresented sources.
        selected.append(chunk)
        if len(selected) >= top_k:
            break

    return selected


def shared_scope_ids() -> set[str]:
    """Scopes whose documents are shared references rather than one machine's."""

    return {get_settings().general_machine_id}


def retrieval_scope(machine_id: str | None) -> str | list[str] | None:
    """Widen a single machine scope to also include the shared general library.

    The general folder's chunks carry general_machine_id as their machine_id,
    so widening the scope is all it takes to make them searchable - there is no
    separate filtering path for them.

    machine_id=None already means "no filter" (used for cross-machine
    questions), so it is left alone rather than narrowed.
    """

    general = get_settings().general_machine_id
    if not machine_id or machine_id == general:
        return machine_id
    return [machine_id, general]


def _fuse(vector_candidates: list[dict], keyword_candidates: list[dict]) -> list[dict]:
    """Merge vector and keyword results with reciprocal rank fusion.

    Fusing on *rank* rather than raw score is the point. The two methods
    produce scores on incompatible scales: a cosine-similarity score is
    1/(1+distance), which is mathematically pinned between 0.33 and 1.0, while
    keyword scores are a hand-tuned tally divided by 100 and routinely land
    below 0.1. Comparing them directly meant the vector score essentially
    always won, so a chunk found by BOTH methods was overwritten by its
    vector-only copy, lost its keyword provenance, and then sorted *below*
    chunks that had merely matched one weak substring. Agreement between the
    two methods - the single strongest relevance signal available here - was
    actively penalised.

    RRF has no scale problem: each method contributes 1/(K + rank), so a chunk
    both methods rank highly beats a chunk either one ranks highly alone, and
    neither method needs its scores normalised or calibrated against the other.
    """

    fused: dict[str, dict] = {}

    for method, candidates in (("vector", vector_candidates), ("keyword", keyword_candidates)):
        for rank, chunk in enumerate(candidates, start=1):
            chunk_id = chunk["id"]
            existing = fused.get(chunk_id)
            contribution = 1.0 / (RRF_K + rank)
            if existing is None:
                # Keep the first copy seen but preserve each method's own score
                # under a method-specific key, so diagnostics can still show
                # why a chunk surfaced.
                fused[chunk_id] = {
                    **chunk,
                    "rrf_score": contribution,
                    "retrieval_method": method,
                    f"{method}_score": chunk.get("score", 0.0),
                    f"{method}_rank": rank,
                }
                continue

            existing["rrf_score"] += contribution
            existing[f"{method}_score"] = chunk.get("score", 0.0)
            existing[f"{method}_rank"] = rank
            # Found by both methods - that agreement is the signal worth naming.
            existing["retrieval_method"] = "hybrid"

    ranked = sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)
    for chunk in ranked:
        # `score` stays the ranking score every downstream caller reads
        # (answer.py thresholds, _preview_rows diagnostics), so it must be the
        # fused value rather than whichever method happened to be seen first.
        chunk["score"] = chunk["rrf_score"]
    return ranked


def retrieve(
    query: str,
    machine_id: str | None,
    top_k: int = 10,
    candidate_k: int | None = None,
    max_chunks_per_source: int | None = None,
    include_keyword: bool = True,
) -> list[dict]:
    """Hybrid search over one machine's documents plus the general folder.

    include_keyword=False skips the keyword half. The progressive retrieval
    loop uses this on later passes: those passes widen only the vector depth
    (candidate_k goes 500 -> 2000 -> 6000) while the keyword side is capped at
    a fixed keyword_candidate_k and its query is often unchanged, so re-running
    it would return the same rows only for the caller to discard them as
    already seen.
    """

    settings = get_settings()
    scope = retrieval_scope(machine_id)

    # Vector search catches semantic matches even when wording is different.
    vector_candidates = search_chunks(
        query=query,
        machine_id=scope,
        top_k=max(candidate_k or settings.retrieval_candidate_k, top_k),
    )

    # Keyword search catches exact alarms, part numbers, and oil grades.
    keyword_candidates = (
        keyword_search(query=query, machine_id=scope, limit=settings.keyword_candidate_k)
        if include_keyword
        else []
    )

    shared = shared_scope_ids()
    for chunk in vector_candidates + keyword_candidates:
        # Tag general-folder hits so callers/citations can label them
        # separately from the selected machine's own docs.
        chunk["source_scope"] = "general" if chunk["metadata"].get("machine_id") in shared else "machine"

    return _diversify_by_source(
        _fuse(vector_candidates, keyword_candidates),
        top_k=top_k,
        max_chunks_per_source=max_chunks_per_source or settings.retrieval_max_chunks_per_source,
    )
