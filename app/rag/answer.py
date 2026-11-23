import json
import re
from pathlib import Path

from app.rag.citations import GENERAL_DOCS_SCOPE_LABEL, MACHINE_SCOPE_LABEL, build_citations
from app.rag.openai_client import chat_completion, chat_model_name, get_client
from app.rag.retriever import retrieve


JUDGE_PROMPT = """You check whether retrieved machine-document excerpts contain enough evidence to answer a technician's question.

Return JSON only with this shape:
{
  "has_answer": true,
  "confidence": "high",
  "answerable_scope": "direct",
  "best_citation_ids": ["cite_1"],
  "reason": "short reason",
  "suggested_next_query": null
}

Rules:
- Use only the retrieved excerpts.
- Do not use general knowledge.
- has_answer=true only when the excerpts actually support the answer.
- When a machine identity is provided, the excerpts support an answer only if
  the value can be tied to that machine - for a table covering several machines
  or configurations, the row consistent with the identity given. Name the row
  you matched in your reason; if no row can be tied to the machine, that is
  partial evidence at best, and say so in the reason.
- Use answerable_scope="partial" when excerpts are related but incomplete.
- Use answerable_scope="not_found" when the excerpts do not answer the question."""

# These passes widen retrieval only when the judge says the current evidence is
# weak. That keeps common questions cheap while still digging deeper before the
# assistant tells a technician the docs do not contain an answer.
#
# candidate_k is how deep the ranked search looks; max_total_chunks is how much
# of that survives diversification and reaches the judge. They are deliberately
# NOT the same number. They used to be, and it was the single biggest source of
# wrong answers: a machine datasheet holds one terse "Clamping force kN 2000"
# line inside a wall of other spec labels, so its embedding is diffuse and it
# ranked in the hundreds - while the operating manual has whole pages of prose
# *about* clamping force. Looking only 100 deep made the datasheet unreachable
# no matter how many passes ran.
#
# Looking deeper is close to free: the vector search is a brute-force scan that
# already scores every chunk for the machine, so a larger candidate_k only
# changes how many rows come back. Cost to the model is unchanged, because
# max_total_chunks still caps what reaches the judge and the answer prompt.
#
# There is deliberately no score_threshold. Retrieval scores are
# reciprocal-rank-fusion values (see retriever._fuse), which measure agreement
# between the keyword and vector rankings - not absolute relevance - so no
# fixed cutoff is meaningful against them.
RETRIEVAL_PASSES = [
    {"pass_number": 1, "max_chunks_per_doc": 4, "max_total_chunks": 50, "candidate_k": 500},
    {"pass_number": 2, "max_chunks_per_doc": 8, "max_total_chunks": 100, "candidate_k": 2000},
    {"pass_number": 3, "max_chunks_per_doc": 12, "max_total_chunks": 100, "candidate_k": 6000},
]


def _source_rows(chunks: list[dict]) -> list[dict]:
    seen = set()
    sources = []
    for chunk in chunks:
        metadata = chunk["metadata"]
        key = (
            metadata.get("machine_id"),
            metadata.get("source_file"),
            metadata.get("page_start"),
            metadata.get("page_end"),
        )
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "machine_id": metadata.get("machine_id"),
                "source_file": metadata.get("source_file"),
                "source_path": metadata.get("source_path"),
                "page_start": metadata.get("page_start"),
                "page_end": metadata.get("page_end"),
                "document_type": metadata.get("document_type"),
                # Set by retrieve() when the chunk came from the merged general
                # search; falls back to machine_id comparison for chunks read
                # directly (e.g. read_surrounding_pages), which skip retrieve().
                "source_scope": chunk.get("source_scope"),
            }
        )
    return sources


def _source_badges(cited_sources: list[dict]) -> list[str]:
    """Roll cited sources up into answer-level badges, in a stable order.

    Distinct from the "General Knowledge" badge added separately in
    tools.py when the model writes uncited general-knowledge commentary -
    this covers the case where real documents were used, from one or both
    scopes. A single answer can carry both badges at once.

    Returns [] when nothing was actually cited - e.g. the model found nothing
    usable in the retrieved chunks and wrote "the documents do not
    contain...". Defaulting to ["Machine Docs"] there would mislabel a
    no-evidence answer as document-backed.
    """

    scopes = {source.get("source_scope") for source in cited_sources}
    badges = []
    if "machine" in scopes:
        badges.append(MACHINE_SCOPE_LABEL)
    if "general" in scopes:
        badges.append(GENERAL_DOCS_SCOPE_LABEL)
    return badges


def _preview_rows(chunks: list[dict]) -> list[dict]:
    previews = []
    for chunk in chunks:
        metadata = chunk["metadata"]
        text = " ".join((chunk.get("text") or "").split())
        previews.append(
            {
                # RRF scores are small (~1/60 at best), so 4dp would collapse
                # most of the ranking detail these previews exist to expose.
                "score": round(float(chunk.get("score", 0)), 6),
                "vector_rank": chunk.get("vector_rank"),
                "keyword_rank": chunk.get("keyword_rank"),
                "machine_id": metadata.get("machine_id"),
                "source_file": metadata.get("source_file"),
                "source_path": metadata.get("source_path"),
                "page_start": metadata.get("page_start"),
                "page_end": metadata.get("page_end"),
                "document_type": metadata.get("document_type"),
                "retrieval_method": chunk.get("retrieval_method"),
                "preview": text[:1200],
            }
        )
    return previews


def _citation_context(chunks: list[dict]) -> tuple[list[dict], list[dict], str]:
    """Build source rows, citation objects, and model context with citation ids."""

    sources = _source_rows(chunks)
    citations = build_citations(sources)
    citation_by_key = {
        (
            citation.get("machine_id"),
            citation.get("source_file"),
            citation.get("page_start"),
            citation.get("page_end"),
        ): citation
        for citation in citations
    }
    blocks = []
    for chunk in chunks:
        metadata = chunk["metadata"]
        citation = citation_by_key.get(
            (
                metadata.get("machine_id"),
                metadata.get("source_file"),
                metadata.get("page_start"),
                metadata.get("page_end"),
            )
        )
        citation_id = citation["id"] if citation else "cite_unknown"
        scope_label = "general troubleshooting reference" if citation and citation.get("source_scope") == "general" else "machine-specific"
        blocks.append(
            "\n".join(
                [
                    f"Citation id: [{citation_id}]",
                    f"Source file: {metadata.get('source_file')}",
                    f"Page: {metadata.get('page_start')}",
                    f"Source scope: {scope_label}",
                    chunk["text"],
                ]
            )
        )
    return sources, citations, "\n\n---\n\n".join(blocks)


def _chunk_key(chunk: dict) -> str:
    """Return a stable id used to avoid resending the same chunk in wide searches."""

    metadata = chunk.get("metadata", {})
    return str(
        chunk.get("id")
        or metadata.get("chunk_id")
        or "|".join(
            [
                str(metadata.get("source_path") or ""),
                str(metadata.get("page_start") or ""),
                str(metadata.get("page_end") or ""),
                str(metadata.get("chunk_index") or ""),
            ]
        )
    )


def _filter_new_chunks(chunks: list[dict], seen_chunk_ids: set[str]) -> list[dict]:
    """Keep only chunks not already sent during this question's retrieval loop."""

    new_chunks = []
    for chunk in chunks:
        chunk_id = _chunk_key(chunk)
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        new_chunks.append(chunk)
    return new_chunks


def _judge_defaults(reason: str = "The evidence judge did not return a usable decision.") -> dict:
    """Return the conservative default when the judge cannot make a clean call."""

    return {
        "has_answer": False,
        "confidence": "low",
        "answerable_scope": "not_found",
        "best_citation_ids": [],
        "reason": reason,
        "suggested_next_query": None,
    }


def judge_retrieved_evidence(
    query: str, chunks: list[dict], established: str = "", machine_block: str = ""
) -> dict:
    """Ask a private model call whether retrieved chunks are enough to answer.

    The judge is intentionally separate from the final answer. Its only job is
    to decide whether the backend should widen retrieval before the user sees a
    response.

    `chunks` is the evidence that is NEW in this pass, not everything gathered
    so far - see _judge_new_evidence_only in run_progressive_rag for why.
    `established` carries a one-line summary of what earlier passes already
    concluded, so continuity survives without resending their text.

    `machine_block` is the identity of the machine being asked about (see
    app.rag.machine_context.machine_identity_block). Without it the judge's
    verdict on a multi-machine table is a coin toss: it can see fifty rows and
    has no way to know which one the question is about. Empty for non-machine
    scopes.

    Note: the citation ids in the returned best_citation_ids are numbered
    per-call by build_citations() and do NOT correspond to the ids the answering
    model sees (which come from tools.CitationRegistry). They are diagnostic
    breadcrumbs only - do not resolve them against a registry.
    """

    if not chunks:
        return _judge_defaults("No new chunks were retrieved.")

    _, _, context = _citation_context(chunks)
    prefix = f"{machine_block}\n\n" if machine_block else ""
    prefix += f"Already established by earlier searches: {established}\n\n" if established else ""
    client = get_client()
    response = chat_completion(client,
        model=chat_model_name(),
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {
                "role": "user",
                "content": f"{prefix}Question: {query}\n\nRetrieved excerpts:\n{context}",
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    try:
        decision = json.loads(raw)
    except json.JSONDecodeError:
        return _judge_defaults("The evidence judge returned invalid JSON.")

    return {
        "has_answer": bool(decision.get("has_answer")),
        "confidence": decision.get("confidence") if decision.get("confidence") in {"high", "medium", "low"} else "low",
        "answerable_scope": decision.get("answerable_scope")
        if decision.get("answerable_scope") in {"direct", "partial", "not_found"}
        else "not_found",
        "best_citation_ids": decision.get("best_citation_ids") or [],
        "reason": decision.get("reason") or "",
        "suggested_next_query": decision.get("suggested_next_query"),
    }


def _judge_found_answer(decision: dict) -> bool:
    """Return true only when the judge says the docs directly support an answer."""

    return (
        bool(decision.get("has_answer"))
        and decision.get("confidence") in {"high", "medium"}
        and decision.get("answerable_scope") == "direct"
    )


_SCOPE_STRENGTH = {"direct": 2, "partial": 1, "not_found": 0}
_CONFIDENCE_STRENGTH = {"high": 2, "medium": 1, "low": 0}


def _verdict_strength(decision: dict) -> tuple[int, int, int]:
    """Order verdicts so the best one across passes can be kept.

    Ranked on scope first, then confidence, then has_answer as a tiebreak: a
    "partial/high" reading of a spec table is more useful than a
    "direct/low" guess, because scope is about whether the evidence is even
    the right material while confidence is about how sure the judge is of it.
    """

    return (
        _SCOPE_STRENGTH.get(decision.get("answerable_scope"), 0),
        _CONFIDENCE_STRENGTH.get(decision.get("confidence"), 0),
        1 if decision.get("has_answer") else 0,
    )


def _best_verdict(current: dict | None, candidate: dict) -> dict:
    """Keep whichever verdict is stronger, preferring the earlier one on a tie.

    Needed because each pass now judges only its own new evidence. Without
    this, a later pass that turned up nothing useful would overwrite an
    earlier pass's good verdict and the answer would be labelled unconfirmed
    even though the evidence for it is still sitting in the chunk list.

    current=None means "no pass has been judged yet", which is distinct from a
    real not_found verdict even though both are the weakest possible strength.
    Without that distinction the placeholder starting verdict ties with a
    genuine not_found and wins the tiebreak, so the run reports the
    placeholder's "retrieval did not run" reason for a run that did.
    """

    if current is None:
        return candidate
    return candidate if _verdict_strength(candidate) > _verdict_strength(current) else current


def _emit(on_progress, event: dict) -> dict:
    """Fire the live progress callback (if any) and return the event for logging."""

    if on_progress:
        on_progress(event)
    return event


def run_progressive_rag(
    query: str,
    machine_id: str,
    top_k: int | None = None,
    on_progress=None,
) -> dict:
    """Retrieve, judge, and widen evidence before deciding the docs failed.

    Each pass judges ONLY the evidence that pass newly found, and the best
    verdict across passes is carried forward. Previously every pass re-judged
    everything accumulated so far, which was wrong on three counts:

    1. Cost and latency grew quadratically. Pass 1 judged ~50 chunks, pass 2
       re-read those 50 plus its own new ones, pass 3 re-read all of them
       again - roughly 240k tokens of judging for a single question, most of
       it spent re-reading text the judge had already ruled on.
    2. It made the judge less decisive, not more. Burying a handful of
       genuinely new excerpts inside a hundred already-rejected ones is a
       worse signal-to-noise ratio than showing them on their own.
    3. The verdict was overwritable. A late pass that surfaced nothing useful
       replaced an earlier pass's good verdict, so an answer with solid
       evidence still in the chunk list got labelled unconfirmed.

    Judging new evidence in isolation does give up the case where a fact is
    only answerable by combining excerpts across two passes. That is rare here
    (chunks are page-scoped, and a spec value lives in one of them), it is
    partly recovered by passing forward what earlier passes established, and
    it costs nothing in the final answer: the answering model still receives
    every accumulated chunk regardless of how the judge scored each batch.

    on_progress, if given, is called with each stage event the moment it
    happens (not just collected for the final diagnostics payload), so a
    streaming caller can show the technician what is actually happening
    instead of a canned/fake status message.
    """

    seen_chunk_ids: set[str] = set()
    evidence_chunks: list[dict] = []
    retrieval_passes = []
    progress_events = []
    search_query = query
    # None until a pass is actually judged - see _best_verdict.
    best_decision: dict | None = None
    established = ""
    last_keyword_query: str | None = None
    # Computed once, passed to every pass's judge call. Empty for non-machine
    # scopes - see machine_identity_block for why silence beats a folder-name
    # heading there.
    from app.rag.machine_context import machine_identity_block

    machine_block = machine_identity_block(machine_id) if machine_id else ""

    for pass_settings in RETRIEVAL_PASSES:
        # Each pass widens the search only if the previous evidence was weak.
        progress_events.append(
            _emit(
                on_progress,
                {
                    "stage": "retrieval",
                    # Say what is actually being searched for. On a widened
                    # pass the query is often not the technician's wording any
                    # more - the judge rewrote it - and seeing that is the
                    # difference between "it is still going" and understanding
                    # what it decided to do.
                    "message": (
                        f"Searching this machine's documents for \"{search_query}\"..."
                        if pass_settings["pass_number"] == 1
                        else f"Widening the search (pass {pass_settings['pass_number']}) for \"{search_query}\"..."
                    ),
                    "pass_number": pass_settings["pass_number"],
                },
            )
        )
        max_total = top_k or pass_settings["max_total_chunks"]
        # Only run the keyword half when its query has actually changed. Later
        # passes widen the vector depth, not the keyword cap, so repeating an
        # unchanged keyword query costs a full text scan and returns rows that
        # are then dropped as already-seen.
        include_keyword = search_query != last_keyword_query
        if include_keyword:
            last_keyword_query = search_query
        # Retrieve more candidates than the final answer may use so the judge has options.
        candidates = retrieve(
            search_query,
            machine_id,
            top_k=max_total,
            # Never let a caller's top_k override shrink the search depth - that
            # is exactly the coupling this pass table exists to avoid.
            candidate_k=max(pass_settings["candidate_k"], max_total),
            max_chunks_per_source=pass_settings["max_chunks_per_doc"],
            include_keyword=include_keyword,
        )
        new_chunks = _filter_new_chunks(candidates, seen_chunk_ids)
        evidence_chunks.extend(new_chunks)

        pass_record = {
            **pass_settings,
            "candidate_count": len(candidates),
            "new_chunk_count": len(new_chunks),
            "seen_chunk_count": len(seen_chunk_ids),
            "evidence_chunk_count": len(evidence_chunks),
        }

        if new_chunks:
            documents = {
                (chunk.get("metadata") or {}).get("source_file") for chunk in new_chunks
            }
            progress_events.append(
                _emit(
                    on_progress,
                    {
                        "stage": "retrieval_result",
                        "message": (
                            f"Found {len(new_chunks)} new excerpt(s) across "
                            f"{len(documents)} document(s)."
                        ),
                        "pass_number": pass_settings["pass_number"],
                    },
                )
            )

        if not new_chunks:
            # A wider search that returned nothing this pass had not already
            # seen means widening further will not help either: the passes only
            # ever increase depth over the same corpus and machine scope. Stop
            # instead of paying for another judge call on nothing.
            pass_record["judge"] = None
            pass_record["skipped_judge"] = "no new chunks"
            retrieval_passes.append(pass_record)
            break

        progress_events.append(
            _emit(
                on_progress,
                {
                    "stage": "judge",
                    "message": "Checking whether those excerpts actually answer the question...",
                    "pass_number": pass_settings["pass_number"],
                },
            )
        )
        decision = judge_retrieved_evidence(
            query, new_chunks, established=established, machine_block=machine_block
        )
        pass_record["judge"] = decision

        # The judge's own words. This is the most informative thing the
        # pipeline produces, so it is streamed to the technician rather than
        # kept only in the diagnostics.
        verdict = {
            "direct": "Found a direct answer",
            "partial": "Found related but incomplete material",
            "not_found": "No answer in these excerpts",
        }.get(decision.get("answerable_scope"), "Checked the excerpts")
        progress_events.append(
            _emit(
                on_progress,
                {
                    "stage": "judge_result",
                    "message": f"{verdict} - {decision.get('reason') or 'no reason given'}",
                    "pass_number": pass_settings["pass_number"],
                },
            )
        )
        retrieval_passes.append(pass_record)

        best_decision = _best_verdict(best_decision, decision)

        if _judge_found_answer(decision):
            break

        if decision.get("reason"):
            # Continuity without resending text: the next pass's judge is told
            # what the last one concluded, in one line.
            established = decision["reason"]
        if decision.get("suggested_next_query"):
            # Let the judge steer the next search phrase when it knows what is missing.
            search_query = decision["suggested_next_query"]

    progress_events.append(_emit(on_progress, {"stage": "final_answer", "message": "Writing the final answer..."}))
    final_decision = best_decision or _judge_defaults("No pass produced a usable verdict.")
    return {
        "chunks": evidence_chunks,
        "judge": final_decision,
        "passes": retrieval_passes,
        "progress_events": progress_events,
        "docs_answer_found": _judge_found_answer(final_decision),
    }


CITATION_ID_RE = re.compile(r"cite_\d+")


SUMMARY_PROMPT = """You condense a completed troubleshooting answer for display to a
technician reading on an iPad at a running machine. The full answer stays available
behind a "More detail" button; your summary is the first thing they see.

- Lead with the value: the number, setting, part, or first action. Context second.
- Two to five short sentences, or one tight list, or one small table if the answer
  hinges on comparing rows. No preamble ("Based on the manual...") and no closing
  offer of more detail - the app provides that button.
- Reuse citation markers like [cite_1] exactly as they appear in the full answer,
  attached to the same facts they support. NEVER write a citation marker that does
  not appear in the full answer.
- Preserve units, configuration qualifiers ("for the 48oz barrel"), and safety
  warnings verbatim - never round a value, drop a qualifier, or soften a warning.
- If the full answer says two sources disagree, the summary must still say so.
- Add nothing that is not in the full answer.
- Output only the summary text - no headers, no "Summary:" label."""


def summarize_full_answer(client, full_answer: str, question: str) -> str | None:
    """Condense the verified full answer for the chat bubble.

    Returns None on any failure so the caller can fall back to showing the
    full answer - a missing summary must never cost the technician the turn.
    A citation marker the model invents (not present in the full answer) is
    stripped here rather than trusted.
    """

    try:
        response = chat_completion(
            client,
            model=chat_model_name(),
            messages=[
                {"role": "system", "content": SUMMARY_PROMPT},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nFull answer:\n{full_answer}",
                },
            ],
            temperature=0.2,
            # The input is a complete long answer; the default request timeout
            # is tuned for short judge calls and cuts this one off.
            timeout=60,
        )
        summary = (response.choices[0].message.content or "").strip()
    except Exception:
        return None
    if not summary:
        return None
    allowed = set(CITATION_ID_RE.findall(full_answer))
    summary = CITATION_ID_RE.sub(
        lambda match: match.group(0) if match.group(0) in allowed else "", summary
    )
    # Dropping an invented id can leave empty brackets behind.
    summary = re.sub(r"\[\s*(?:,\s*)*\]", "", summary)
    return summary.strip() or None


def _citations_used_in_answer(answer: str, citations: list[dict]) -> list[dict]:
    """Return citations in the order their ids appear in the model answer.

    Matches individual cite_N tokens rather than requiring a single-id
    bracket like [cite_1] - the model sometimes groups several ids in one
    bracket, e.g. [cite_6, cite_7, cite_10, cite_11], which a
    bracket-anchored regex would miss entirely.
    """

    citation_by_id = {citation["id"]: citation for citation in citations}
    used = []
    seen = set()
    for match in CITATION_ID_RE.finditer(answer):
        citation_id = match.group(0)
        if citation_id in seen or citation_id not in citation_by_id:
            continue
        # Keep display citations in the same order the model referenced them.
        seen.add(citation_id)
        used.append(citation_by_id[citation_id])
    return used


def _renumber_answer_citations(answer: str, citations: list[dict]) -> tuple[str, list[dict]]:
    """Compress used citation ids into cite_1, cite_2, etc. for display."""

    renumbered = []
    id_map = {}
    for index, citation in enumerate(citations, start=1):
        new_id = f"cite_{index}"
        id_map[citation["id"]] = new_id
        renumbered.append({**citation, "id": new_id})

    # Replace every cite_N token found, not just ones wrapped in a
    # single-id bracket - keeps multi-id brackets like [cite_6, cite_7]
    # correctly renumbered instead of left untouched.
    updated_answer = CITATION_ID_RE.sub(lambda match: id_map.get(match.group(0), match.group(0)), answer)
    return updated_answer, renumbered


def _normalize_reference_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _cited_source_rows(answer: str, sources: list[dict]) -> list[dict]:
    # Find sources when the model mentions a file/page instead of [cite_n].
    answer_norm = _normalize_reference_text(answer)
    answer_lower = answer.lower()
    cited = []
    seen = set()

    for source in sources:
        source_file = source.get("source_file") or ""
        source_stem = Path(source_file).stem
        file_norm = _normalize_reference_text(source_file)
        stem_norm = _normalize_reference_text(source_stem)
        page_start = source.get("page_start")
        page_end = source.get("page_end")

        # This fallback keeps citations clickable even if the model writes prose.
        file_mentioned = bool(file_norm and file_norm in answer_norm) or bool(
            stem_norm and stem_norm in answer_norm
        )
        page_patterns = [rf"\bpage\s+{page_start}\b", rf"\bpages\s+{page_start}\b"]
        if page_end and page_end != page_start:
            page_patterns.append(rf"\bpages\s+{page_start}\s*[-–]\s*{page_end}\b")
        page_mentioned = any(re.search(pattern, answer_lower) for pattern in page_patterns)

        if not file_mentioned and not page_mentioned:
            continue

        key = (
            source.get("source_path"),
            source.get("page_start"),
            source.get("page_end"),
        )
        if key in seen:
            continue
        seen.add(key)
        cited.append(source)

    return cited
