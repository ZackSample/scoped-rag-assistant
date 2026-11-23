"""Conversational tool-calling loop: the model decides what to search/read.

The model gets a fixed tool set and decides for itself whether it already has enough
context (conversation history, citations already shown), needs a quick
targeted lookup, needs to read more of a page it already cited, needs to
look at a page/slide image, needs a full exhaustive search, or wants to
check whether this machine has answered something similar before.

The judge (see answer.py) only runs inside run_full_search - not on every
tool call - since targeted lookups are things the model can evaluate itself
from context, and running a second LLM call to judge every tool result would
add cost/latency without a real safety benefit.
"""

import json
import re
from dataclasses import dataclass, field
from time import perf_counter

from app.config import get_settings
from app.rag.answer import (
    _cited_source_rows,
    _citations_used_in_answer,
    _preview_rows,
    _renumber_answer_citations,
    _source_badges,
    _source_rows,
    run_progressive_rag,
    summarize_full_answer,
)
from app.rag.citations import build_citation, build_citations
from app.rag.machine_context import machine_context, machine_identity_block
from app.rag.machine_resolver import resolve_machine
from app.rag.openai_client import chat_completion, chat_model_name, get_client
from app.rag.pdf_preview import render_preview
from app.rag.retriever import retrieval_scope, retrieve, shared_scope_ids
from app.rag.vector_store import get_chunks


# A ceiling on runaway loops, not a budget the model is meant to spend. It binds
# on follow-ups, where answering properly is a chain: search, read the excerpt,
# open the page image, check the pages either side, and if that candidate does
# not hold the answer, start again on the next one. Each cycle is several rounds,
# so a handful of candidates exhausts a low limit and the turn ends mid-search.
#
# Most turns use a single call (the forced cold-start search), but the longest
# chains get close to 15. Verification is also deliberately expensive: the model
# is pushed to open the page when reading a value out of a table, because those
# tables survive text extraction as one run-together line.
#
# Raising it costs nothing on the common path, since the loop exits as soon as
# the model stops calling tools. Context does not grow without bound either -
# _trim_tool_context() caps it at MAX_TOOL_CONTEXT_CHARS every round - and
# exhaustion is handled: the model is asked for its best answer and to say
# whether it was close.
MAX_TOOL_ROUNDS = 30
GENERAL_KNOWLEDGE_LABEL = "General Knowledge"

GENERAL_KNOWLEDGE_MARKER_RE = re.compile(
    r"\[general_knowledge\](.*?)\[/general_knowledge\]", re.DOTALL | re.IGNORECASE
)

# The model appends this when it deliberately kept the first answer short and
# has more to give. Stripped here and surfaced as a flag so the app can offer a
# button, rather than the model writing "want more detail?" into every answer
# in slightly different words.
MORE_DETAIL_MARKER_RE = re.compile(r"\[/?more[_ ]?detail\]", re.IGNORECASE)

VISION_PROMPT = """You are looking at one rendered page or slide from an injection molding machine document.

Describe only what is visible in the image - do not guess beyond what you can see.
Focus on diagrams, defect photos, schematics, warnings, or labeled parts relevant to the technician's question.
When a machine identity is provided and the page holds a table or list covering
several machines or configurations, find the entry consistent with that machine,
transcribe its row exactly as printed, and also name the other designations
present on the page - a mismatch is only catchable if the alternatives are visible.
If the image does not show anything relevant to the question, say so plainly."""

TOOL_SYSTEM_PROMPT = """You are a conversational assistant for injection molding machine equipment documents and troubleshooting.

You have tools to search and read documents. Use them as needed - you do not
need to search on every turn if the conversation already has enough context
(citations already shown to you, or your own memory of this conversation).

Tool usage guidance:
- search_documents: quick targeted lookup. Use your own judgment on whether
  the results are good enough - do not call a second tool just to double-check
  a clear result.
- read_page: reach for this readily. A retrieved excerpt is a fragment, and
  tables, procedures, and spec lists routinely run past where it was cut. If
  the excerpt you are about to answer from looks clipped, or the technician
  needs an exact value out of it, read the page before you answer rather than
  working from the fragment.
- read_page_image: use whenever the answer may live in something the text
  extract could not carry - a diagram, a photo, a schematic, an exploded parts
  drawing, or a lookup/spec table that came through as jumbled text or not at
  all. Many pages in this corpus are images with no extractable text, so a
  thin or garbled excerpt on a page that should have a table is a reason to
  look at the page, not a reason to give up. Only call this on a citation you
  already have; never guess at a file/page.
- run_full_search: use for a brand new question, or when search_documents
  found nothing and you have no other leads. This is a slower, exhaustive
  multi-pass search with automatic evidence checking - do not call it
  repeatedly for the same question.
- search_answer_history: use when a question sounds like it may be a repeat
  for this machine.

You get a limited number of tool calls this turn. If you run out before
finding a clean answer, write your best answer with what you have, and if
you believe you were close, tell the technician and ask whether they want
you to keep digging.

Answering rules:
- Every factual claim sourced from a document must cite one of the citation
  ids you were given, like [cite_1]. Use only citation ids that were
  actually shown to you. If a claim is backed by multiple citations, use a
  separate bracket for each, like [cite_1][cite_2], not [cite_1, cite_2].
- Do not write raw source names or parenthetical citations like
  "(Source: file.pdf, page 3)" - use [cite_n] only.
- When answering about one machine, do not use documents from other machines.
- A specification listed once per configuration - several values on one line,
  a column per option, a row per model - is not this machine's value until you
  know which configuration it has. Look for that stated for this machine,
  usually in the same document as the table. If you find it, use the matching
  one and say which. If you cannot, give the options together with whatever
  selects between them and ask, rather than choosing. A technician given the
  choice can resolve it; one given the wrong value confidently cannot tell.
  Never state a property of the machine as your reason for choosing unless a
  document says it. That turns a guess into a fact.
- If two sources give different answers to the same question, say so plainly.
  State both values, cite each one separately, and tell the technician they
  disagree. Never quietly pick one and present it as the answer, and never
  average or blend them.
  If there is a reason to trust one more - it is specific to this machine
  rather than the whole product line, it is the machine's own manual rather
  than a shared reference, it is newer - say which you would use and why, but
  still show the other. A technician who knows two documents disagree can go
  and check; one handed a single confident number cannot.
  The same applies to one source that contradicts itself, and to a value you
  cannot tie to this machine's row in a multi-machine table.
- Never open your answer with a value the rest of the answer disclaims,
  corrects, or attributes to a different configuration. If verification
  changed your mind, the verified value is the answer - lead with it. An
  answer that headlines one number and then names a different one as correct
  is worse than either number alone: the reader takes the headline.
- You may combine machine-specific documents, general troubleshooting
  documents, and your own general knowledge in one answer - technicians want
  the full picture. When you do, briefly make clear which parts are
  confirmed by this machine's own documents, which are from general/shared
  references, and which are general knowledge not from any document.
- Wrap any sentence that is general knowledge with no document behind it in
  [general_knowledge] and [/general_knowledge] tags, e.g.
  "[general_knowledge]Generally, excessive clamp pressure can cause
  flash.[/general_knowledge]" These tags are stripped before the technician
  sees them - they exist so the app can label that sentence correctly.
- If nothing you found actually answers the question, say so clearly. You
  may still mention related material you found, and you may add general
  knowledge (tagged as above), but do not present either as a confirmed
  answer to the exact question.
- For safety, maintenance, electrical, hydraulic, or setup procedures,
  include relevant warnings from the context. Do not suggest bypassing
  guards, interlocks, alarms, or safety systems.

Answer shape:
- Your reader is an operator or technician standing at a machine. Lead with
  the answer itself - the number, the setting, the first step - never with
  background or setup.
- Answer COMPLETELY. Include what a careful tech would want behind the
  headline: the applicable table rows, the remaining steps, other plausible
  causes, related warnings, and the qualifier that says which configuration
  a value belongs to. When your answer came from documents, a display layer
  may condense it for the first view, so write the full treatment here
  rather than trimming yourself - but completeness is not padding: no
  restating the question, no theory the question didn't ask for, no filler.
- Structure long answers so they scan: short lead, then bullets, steps, or a
  table.

Formatting - the app renders plain markdown and nothing else:
- Allowed: **bold**, bullet lists, numbered steps, `inline code`, and
  markdown tables with pipes. Use a table whenever the source is a table or
  you are comparing values across sizes, temperatures, or models.
- NEVER use LaTeX or TeX math markup. No backslash-bracket or
  backslash-parenthesis math delimiters, no $ math, no TeX commands such as
  frac, times, approx, or gtrsim. None of it renders - the technician sees
  raw backslashes and braces.
- Write formulas as plain text on their own line, using / for division and x
  for multiplication, then name the symbols underneath. For example:
      Re = 3160 x GPM / (v x D)
      v = kinematic viscosity in centistokes, D = diameter in inches
- Write units and symbols as plain words or characters: degrees F, cSt, GPM.
- Skip headings in a short answer; they make three sentences look like a
  report."""



TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Run a single targeted search over this machine's documents plus the "
                "general troubleshooting folder. Use for quick, specific lookups (a part "
                "number, a spec value, a term) when you don't already have relevant "
                "citations in context. Does not widen or retry automatically - if results "
                "look thin, either refine the query and search again, or call "
                "run_full_search for an exhaustive search."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search text."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": (
                "Read the full text of the page(s) a citation points to, optionally "
                "including nearby pages. Use when a citation's excerpt looks cut off or "
                "you need more surrounding context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "citation_id": {
                        "type": "string",
                        "description": "A citation id already shown to you, like cite_2.",
                    },
                    "radius": {
                        "type": "integer",
                        "description": "Pages before/after to include. 0 means just the cited page.",
                        "minimum": 0,
                        "maximum": 5,
                    },
                },
                "required": ["citation_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page_image",
            "description": (
                "Look at the actual rendered image of the page/slide a citation points "
                "to, using vision. Use when the question is about something visual and "
                "the text excerpt does not describe it. When you are checking a value "
                "you are about to state, pass the claim via `verify` so the reader "
                "knows exactly what to confirm or refute."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "citation_id": {
                        "type": "string",
                        "description": "A citation id already shown to you, like cite_2.",
                    },
                    "verify": {
                        "type": "string",
                        "description": (
                            "Optional. The specific claim you are checking against this "
                            "page - the value and the row or label you believe it comes "
                            "from, e.g. 'oil capacity 40 gal from the 200-ton row'."
                        ),
                    },
                },
                "required": ["citation_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_full_search",
            "description": (
                "Run an exhaustive, multi-pass document search with automatic evidence "
                "checking. Slower and more thorough than search_documents. Use this for a "
                "brand new question, or when search_documents found nothing useful and "
                "you have no other leads."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search text."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_answer_history",
            "description": "Check whether a similar question has already been answered for this machine before.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The technician's question."}},
                "required": ["query"],
            },
        },
    },
]


def _source_scope(machine_id: str) -> str:
    return "general" if machine_id in shared_scope_ids() else "machine"


class CitationRegistry:
    """Tracks citations discovered during one conversational turn.

    Citation ids stay stable for the whole turn so read_page/read_page_image
    can be keyed off an id the model already saw instead of a raw path -
    that makes machine-scope enforcement automatic, since the model can only
    ever request a page it was already shown a citation for.
    """

    def __init__(self, seed_citations: list[dict] | None = None):
        self._by_key: dict[tuple, dict] = {}
        self._chunks_by_key: dict[tuple, dict] = {}
        self._order: list[str] = []
        self._next_index = 1
        for citation in seed_citations or []:
            self._register_source(citation, chunk=None)

    def _key(self, source: dict) -> tuple:
        return (
            source.get("machine_id"),
            source.get("source_file"),
            source.get("page_start"),
            source.get("page_end"),
        )

    def _register_source(self, source: dict, chunk: dict | None) -> dict:
        key = self._key(source)
        existing = self._by_key.get(key)
        if existing:
            if chunk is not None:
                self._chunks_by_key.setdefault(key, chunk)
            return existing

        new_id = f"cite_{self._next_index}"
        self._next_index += 1
        citation = build_citation(source, new_id)
        self._by_key[key] = citation
        self._order.append(new_id)
        if chunk is not None:
            self._chunks_by_key[key] = chunk
        return citation

    def register_chunks(self, chunks: list[dict]) -> list[dict]:
        """Register chunks as citations (reusing ids for already-seen pages).

        Returns the citation dicts in the same order as the input chunks.
        """

        registered = []
        for chunk in chunks:
            metadata = chunk["metadata"]
            source = {
                "machine_id": metadata.get("machine_id"),
                "source_file": metadata.get("source_file"),
                "source_path": metadata.get("source_path"),
                "page_start": metadata.get("page_start"),
                "page_end": metadata.get("page_end"),
                "document_type": metadata.get("document_type"),
                "source_scope": chunk.get("source_scope") or _source_scope(metadata.get("machine_id")),
            }
            registered.append(self._register_source(source, chunk))
        return registered

    def get(self, citation_id: str) -> dict | None:
        for citation in self._by_key.values():
            if citation["id"] == citation_id:
                return citation
        return None

    def all_ordered(self) -> list[dict]:
        by_id = {citation["id"]: citation for citation in self._by_key.values()}
        return [by_id[cid] for cid in self._order if cid in by_id]

    def chunks(self) -> list[dict]:
        """Raw chunks backing registered citations (empty for seeded citations,
        which arrive from the frontend as citation objects with no chunk text)."""

        return list(self._chunks_by_key.values())

    def preview_rows(self) -> list[dict]:
        return _preview_rows(self.chunks())


def format_chunks_as_context(chunks: list[dict], registry: CitationRegistry) -> str:
    """Register chunks and format them as citation-id-tagged text blocks."""

    citations = registry.register_chunks(chunks)
    blocks = []
    for chunk, citation in zip(chunks, citations):
        metadata = chunk["metadata"]
        scope_label = "general troubleshooting reference" if citation["source_scope"] == "general" else "machine-specific"
        lines = [
            f"Citation id: [{citation['id']}]",
            f"Source file: {metadata.get('source_file')}",
            f"Page: {metadata.get('page_start')}",
            f"Source scope: {scope_label}",
        ]
        lines.append(chunk["text"])
        blocks.append("\n".join(lines))
    return "\n\n---\n\n".join(blocks) if blocks else "(No results.)"


@dataclass
class ToolContext:
    machine_id: str
    registry: CitationRegistry
    on_progress: object
    query: str
    diagnostics: dict = field(default_factory=dict)
    # Set once when the turn starts so every progress event shares one clock.
    started_at: float = field(default_factory=perf_counter)


def _emit(ctx: ToolContext, stage: str, message: str) -> None:
    # elapsed_ms turns the stored progress trail into stage timings after the
    # fact, without deciding up front which stage boundaries matter.
    ctx.diagnostics.setdefault("progress_events", []).append(
        {"stage": stage, "message": message, "elapsed_ms": int((perf_counter() - ctx.started_at) * 1000)}
    )
    if ctx.on_progress:
        ctx.on_progress({"stage": stage, "message": message})


def _reasoning_emitter(ctx: ToolContext, on_reasoning):
    """Record the model's commentary in diagnostics and forward it live."""

    def emit(text: str) -> None:
        if not text:
            return
        ctx.diagnostics.setdefault("reasoning", []).append(text)
        if on_reasoning:
            on_reasoning(text)

    return emit


def _tool_search_documents(arguments: dict, ctx: ToolContext) -> str:
    query = arguments.get("query") or ctx.query
    _emit(ctx, "tool_call", f"Searching documents for \"{query}\"...")
    chunks = retrieve(query, ctx.machine_id, top_k=12, candidate_k=40, max_chunks_per_source=3)
    if not chunks:
        return "No results found for this search."
    return format_chunks_as_context(chunks, ctx.registry)


def _tool_read_page(arguments: dict, ctx: ToolContext) -> str:
    citation_id = arguments.get("citation_id")
    radius = int(arguments.get("radius") or 0)
    citation = ctx.registry.get(citation_id)
    if not citation:
        return f"Unknown citation id {citation_id!r}. Use a citation id you were already shown."

    source_path = citation.get("source_path")
    page_start = citation.get("page_start")
    if not source_path or not page_start:
        return f"Citation {citation_id} has no page to read."

    _emit(ctx, "tool_call", f"Reading page {page_start} of {citation.get('source_file')}...")
    chunks = get_chunks(
        machine_id=ctx.machine_id,
        source_path=source_path,
        page_start=max(1, int(page_start) - radius),
        page_end=int(page_start) + radius,
    )
    if not chunks:
        return f"No text found around page {page_start} of {citation.get('source_file')}."
    for chunk in chunks:
        chunk["source_scope"] = citation.get("source_scope")
    return format_chunks_as_context(chunks, ctx.registry)


def _tool_read_page_image(arguments: dict, ctx: ToolContext) -> str:
    citation_id = arguments.get("citation_id")
    citation = ctx.registry.get(citation_id)
    if not citation:
        return f"Unknown citation id {citation_id!r}. Use a citation id you were already shown."

    source_path = citation.get("source_path")
    page_start = citation.get("page_start")
    if not source_path or not page_start:
        return f"Citation {citation_id} has no page to view."

    _emit(ctx, "tool_call", f"Looking at page {page_start} of {citation.get('source_file')}...")
    try:
        preview = render_preview(source_path, int(page_start), zoom=MODEL_PAGE_ZOOM)
    except Exception as exc:
        return f"Could not render an image for this citation: {exc}"

    if preview["kind"] != "image":
        # No image renderer available for this file type on this machine -
        # fall back to whatever text preview exists rather than failing outright.
        return f"No image renderer available; extracted text instead:\n\n{preview.get('content', '')}"

    verify = (arguments.get("verify") or "").strip()
    return _describe_image(
        preview["content"],
        preview.get("media_type", "image/png"),
        ctx.query,
        machine_block=machine_identity_block(ctx.machine_id),
        verify=verify,
    )


# Zoom for pages rendered FOR THE MODEL, not for the human preview panel.
# 1.6 (~115 DPI) is sized for a panel a technician can already zoom with their
# fingers; the vision model gets exactly the pixels it is given and nothing
# more, and the pages that matter most here are fifty-row spec matrices in
# small print - misreading one digit at 115 DPI is a plausible failure. 2.5 is
# ~220 DPI, which keeps a letter page around 2100x2800px: comfortably readable
# without ballooning the request payload. Standalone images take a different
# path (render_image_page_png), already capped at a similar density.
MODEL_PAGE_ZOOM = 2.5


def _describe_image(
    image_b64: str, media_type: str, query: str, machine_block: str = "", verify: str = ""
) -> str:
    # The machine identity is the difference between "describe this table" and
    # "find this machine's row in this table". Row selection is a VISUAL task -
    # the describing model is the only one that can see which designations sit
    # in which row - so withholding the identity from it forced the answering
    # model to re-derive the row from a prose description, which is where the
    # wrong-row answers came from. Empty for non-machine scopes; the prompt
    # rule is conditional on it being present.
    #
    # `verify` is the claim the ANSWERING model is checking. Without it the
    # describing model only knows the technician's question, so it describes
    # pages blind to what is actually at stake - "what does this page say about
    # oil?" instead of "confirm 40 gal from the 200-ton row".
    parts = []
    if machine_block:
        parts.append(machine_block)
    if verify:
        parts.append(
            "The assistant is checking this claim against the page: "
            f"{verify}\n"
            "Confirm or refute it from what is printed - quote the relevant row or "
            "label exactly as it appears, and say plainly if the page does not "
            "support the claim."
        )
    parts.append(
        f"Technician's question: {query}\n\nDescribe what is shown, focusing on anything relevant to the question."
    )

    client = get_client()
    response = chat_completion(client,
        model=chat_model_name(),
        messages=[
            {"role": "system", "content": VISION_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "\n\n".join(parts)},
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_b64}"}},
                ],
            },
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content or "(The vision model returned no description.)"


def _cap_per_source(chunks: list[dict], max_per_source: int, limit: int) -> list[dict]:
    """Trim evidence for the prompt without letting one document take it over.

    Walks in the order retrieval produced (already relevance-sorted and
    diversified) and keeps at most max_per_source chunks per file, so a single
    large manual cannot crowd out the datasheet or spec sheet that holds the
    actual answer.
    """

    kept: list[dict] = []
    per_source: dict[str, int] = {}
    for chunk in chunks:
        source = (chunk.get("metadata") or {}).get("source_file") or ""
        if per_source.get(source, 0) >= max_per_source:
            continue
        per_source[source] = per_source.get(source, 0) + 1
        kept.append(chunk)
        if len(kept) >= limit:
            break
    return kept


def _tool_run_full_search(arguments: dict, ctx: ToolContext) -> str:
    query = arguments.get("query") or ctx.query

    def forward_progress(event: dict) -> None:
        _emit(ctx, event.get("stage", "retrieval"), event.get("message", "Searching..."))

    rag_result = run_progressive_rag(query, ctx.machine_id, on_progress=forward_progress)
    ctx.diagnostics["run_full_search"] = {
        "docs_answer_found": rag_result["docs_answer_found"],
        "judge": rag_result["judge"],
        "passes": rag_result["passes"],
    }

    # Always return the best evidence found, even when the judge said
    # not_found across all passes - the model decides what to do with
    # unconfirmed evidence instead of the backend discarding it.
    #
    # Do NOT re-sort by raw score here. Retrieval deliberately diversifies
    # across documents, capping how many chunks any one file contributes, and
    # sorting by score throws that away: a 600-page manual with pages of prose
    # about a term always outscores the one-line datasheet row that actually
    # states the value, so the manual reclaims every slot. Keeping retrieval's
    # order preserves the balance, and the per-source cap below keeps the
    # prompt from blowing up on a widened search.
    chunks = _cap_per_source(rag_result["chunks"], max_per_source=4, limit=50)
    if not chunks:
        return f"No documents found. Evidence judge reason: {rag_result['judge'].get('reason') or 'none given'}."

    context = format_chunks_as_context(chunks, ctx.registry)
    judge = rag_result.get("judge") or {}
    reason = (judge.get("reason") or "").strip()

    # The judge reports what it found. It does not decide what the answer may
    # say - ranking decides what the model sees, and the model decides what to
    # write. The previous header told the model to "clearly tell the technician
    # this is not a confirmed answer", which is the judge gating the answer.
    #
    # The judge's REASON is passed through too. When only the verdict was
    # passed, the model was told *that* the evidence was confirmed but never
    # *what* had been confirmed - and could answer from a different row of the
    # same table than the one the judge had already identified.
    if rag_result["docs_answer_found"]:
        state = "An independent evidence check found these excerpts do answer the question."
    else:
        state = (
            "An independent evidence check did not confirm a direct answer in these "
            "excerpts. They are the closest matches from a widened search, and may still "
            "contain the answer."
        )

    lines = [state]
    if reason:
        # Explicitly a second opinion rather than a conclusion: it read the same
        # excerpts and can pick the wrong row itself.
        lines.append(
            f"What that check concluded, for your consideration - verify it against the "
            f"excerpts rather than repeating it: {reason}"
        )
    lines.append(
        "If you are reading a value out of a table, confirm the row belongs to this "
        "machine. Tables often survive text extraction as one run-together line, so if an "
        "excerpt looks jumbled, truncated, or you cannot tell which row is which, call "
        "read_page_image on that citation and read the page instead of guessing."
    )
    return "\n\n".join(lines) + "\n\n" + context


def _tool_search_answer_history(arguments: dict, ctx: ToolContext) -> str:
    from app.db.history import find_reuse_candidate

    query = arguments.get("query") or ctx.query
    _emit(ctx, "tool_call", "Checking this machine's answer history...")
    candidate = find_reuse_candidate(ctx.machine_id, query)
    if not candidate:
        return "No similar previous answer found for this machine."
    return (
        f"A similar question was answered before (match confidence: {candidate['confidence']}, "
        f"score {candidate['match_score']}):\n"
        f"Previous question: {candidate['question']}\n"
        f"Previous answer: {candidate['answer_preview']}\n\n"
        "This is informational only - it has no citation id and cannot be cited with [cite_n]. "
        "You may mention that this was asked before, but ground any factual claim in a "
        "document via search_documents or run_full_search."
    )


TOOL_EXECUTORS = {
    "search_documents": _tool_search_documents,
    "read_page": _tool_read_page,
    "read_page_image": _tool_read_page_image,
    "run_full_search": _tool_run_full_search,
    "search_answer_history": _tool_search_answer_history,
}


def execute_tool(name: str, arguments: dict, ctx: ToolContext) -> str:
    executor = TOOL_EXECUTORS.get(name)
    if not executor:
        return f"Unknown tool {name!r}."
    ctx.diagnostics.setdefault("tool_path", []).append(name)
    try:
        return executor(arguments, ctx)
    except Exception as exc:
        # Returned to the model rather than raised: it can retry or work around
        # a failed tool, and the technician still gets an answer.
        return f"Tool {name} failed: {exc}"


def _strip_general_knowledge_markers(answer: str) -> tuple[str, bool]:
    used = bool(GENERAL_KNOWLEDGE_MARKER_RE.search(answer))
    stripped = GENERAL_KNOWLEDGE_MARKER_RE.sub(lambda match: match.group(1), answer)
    return stripped, used


def _strip_more_detail_marker(answer: str) -> tuple[str, bool]:
    offered = bool(MORE_DETAIL_MARKER_RE.search(answer))
    stripped = MORE_DETAIL_MARKER_RE.sub("", answer)
    return stripped.rstrip(), offered


# Anything the model writes that must not reach the screen verbatim: the
# general-knowledge tags are internal labelling, and cite_N ids are renumbered
# for display only once the full answer is known (see _renumber_answer_citations).
_STREAM_HIDDEN_RE = re.compile(
    r"\[/?general_knowledge\]|\[cite_\d+\]|\[/?more[_ ]?detail\]",
    re.IGNORECASE,
)


def streamable_prefix(raw: str) -> str:
    """Return the part of a partial answer that is safe to show right now.

    Streaming raw model text would leak two things the technician should never
    see: the [general_knowledge] tags, and citation ids that are still using
    their pre-renumbering values. Both are stripped here.

    The trailing-bracket guard is the subtle part. A delta can arrive
    mid-token, so raw may end with "...pressure [cite_" - which matches no
    pattern yet and would be emitted as literal text, then never retracted
    once the closing bracket arrives. So everything from an unclosed "[" is
    held back until the bracket completes.
    """

    visible = _STREAM_HIDDEN_RE.sub("", raw)
    open_bracket = visible.rfind("[")
    if open_bracket != -1 and "]" not in visible[open_bracket:]:
        return visible[:open_bracket]
    return visible


def _tag_citation_roles(citations: list[dict]) -> list[dict]:
    """Tag the first-cited scope as primary, every other scope as supporting."""

    if not citations:
        return []
    primary_scope = citations[0].get("source_scope")
    return [
        {**citation, "role": "primary" if citation.get("source_scope") == primary_scope else "supporting"}
        for citation in citations
    ]


def _final_badges(tagged_citations: list[dict], used_general_knowledge: bool) -> list[str]:
    badges = _source_badges(tagged_citations)
    if used_general_knowledge:
        badges.append(GENERAL_KNOWLEDGE_LABEL)
    return badges


# Roughly how much tool output the conversation is allowed to carry forward.
# Characters, not tokens, to avoid shipping a tokeniser on the hot path; at a
# pessimistic ~3 chars/token this is about 40k tokens of evidence.
MAX_TOOL_CONTEXT_CHARS = 120_000
# The newest tool results are always kept whole - they are what the model asked
# for most recently and is most likely reasoning about.
MIN_INTACT_TOOL_RESULTS = 4
_TRUNCATION_NOTE = "\n\n[... earlier excerpt trimmed to keep this conversation within context ...]"


def _trim_tool_context(messages: list[dict]) -> list[dict]:
    """Shrink older tool results once the conversation gets large.

    Every round resends the entire message list. A forced full search puts ~50
    chunks into the first tool result, so an unbounded 15-round conversation
    could resend tens of thousands of tokens of evidence per round - the single
    largest driver of both cost and time-to-answer on a hard question.

    Only tool results are trimmed, and only the older ones. The system prompt,
    the question, the tool-call history and the model's reasoning items stay
    intact: losing those changes what the model is doing (and for a reasoning
    model, breaks the chain), whereas an older excerpt has usually already been
    read and either used or discarded.
    """

    def is_tool_output(message: dict) -> bool:
        # Responses calls these function_call_output items; the chat-completions
        # shape was a "tool" role with a tool_call_id.
        return message.get("type") == "function_call_output"

    total = sum(len(message.get("output") or "") for message in messages if is_tool_output(message))
    if total <= MAX_TOOL_CONTEXT_CHARS:
        return messages

    tool_indexes = [index for index, message in enumerate(messages) if is_tool_output(messages[index])]
    protected = set(tool_indexes[-MIN_INTACT_TOOL_RESULTS:])

    budget = MAX_TOOL_CONTEXT_CHARS - sum(len(messages[index].get("output") or "") for index in protected)
    # Distribute what is left over the older results rather than dropping them
    # outright, so the model still knows what those searches turned up.
    older = [index for index in tool_indexes if index not in protected]
    per_message = max(600, budget // len(older)) if older and budget > 0 else 600

    trimmed = list(messages)
    for index in older:
        output = trimmed[index].get("output") or ""
        if len(output) > per_message:
            trimmed[index] = {**trimmed[index], "output": output[:per_message] + _TRUNCATION_NOTE}
    return trimmed


@dataclass
class TurnResult:
    """One model turn: either tool calls to run, or final answer text.

    output_items are the raw items the API returned, ready to be appended to
    the next request's input. They must be carried forward verbatim - for a
    reasoning model the reasoning items are part of the chain, and dropping
    them makes each turn start over.
    """

    content: str
    tool_calls: list[dict]
    output_items: list[dict]
    # input/output token counts for THIS turn, recorded with the diagnostics.
    # Empty when the API returned no usage block.
    usage: dict = field(default_factory=dict)


def _responses_tool_specs(allowed: set[str] | None = None) -> list[dict]:
    """TOOL_SPECS in the shape the Responses API expects.

    Chat completions nests the schema under a "function" key; Responses puts
    name/description/parameters at the top level alongside "type".

    `allowed` narrows the tool set for one turn. Withholding a tool is stronger
    than telling the model not to use it - see EXPAND_INSTRUCTION.
    """

    return [
        {
            "type": "function",
            "name": spec["function"]["name"],
            "description": spec["function"]["description"],
            "parameters": spec["function"]["parameters"],
        }
        for spec in TOOL_SPECS
        if allowed is None or spec["function"]["name"] in allowed
    ]


# Tools left available when expanding an answer the model already gave. Search
# is deliberately absent: a second retrieval run re-ranks from scratch and can
# surface a different, conflicting set of evidence than the answer being
# expanded was built on - or miss it entirely and walk back a correct answer.
# The evidence is already in hand; what is worth doing is looking at it harder.
EXPAND_TOOLS = {"read_page", "read_page_image"}

# How many of the previous answer's citations get their pages read automatically
# when expanding. Enough to cover the sources a short answer actually leaned on,
# without pulling the whole ledger (which includes citations it merely saw).
EXPAND_FORCED_READS = 4

EXPAND_INSTRUCTION = """The technician tapped "more detail" on your last answer. They are not
asking a new question - they want the answer you just gave, in full.

You already have the evidence. Do NOT search again. The full text of the pages
behind your last answer's citations has already been read for you and is above.
Work from that, and read further where it helps:
- read_page again with a larger radius where a table or procedure clearly runs
  past the pages you were given.
- read_page_image on any citation where the page holds a diagram, photo,
  schematic, or a table that the text extract mangled. Lookup tables, spec
  tables, and exploded parts drawings routinely survive only as pictures - if
  your last answer gave a formula and the page also has a table of worked
  values, the table is what the technician actually wants.

Then write the full answer: the detail you left out, the numbers straight off
the page, and the exact values or steps rather than a summary of them. Correct
anything the closer read shows you got wrong, and say plainly that you are
correcting it."""


def clean_reasoning(text: str) -> str:
    """Tidy a reasoning summary block for display.

    The model formats these as markdown with a bold title line, e.g.
    "**Clarifying page references**\\n\\nI see that the search...". The activity
    log escapes what it renders, so the asterisks would show literally. The
    title is folded into the body as a lead-in rather than dropped - it is
    usually the most scannable part of the block.
    """

    body = text.strip()
    heading = re.match(r"^\*\*(.+?)\*\*\s*(.*)$", body, re.DOTALL)
    if heading:
        title, rest = heading.group(1).strip(), heading.group(2).strip()
        body = f"{title} - {rest}" if rest else title
    body = re.sub(r"\*\*(.+?)\*\*", r"\1", body)
    # Reasoning arrives as paragraphs; one line reads better in a log.
    return re.sub(r"\s+", " ", body).strip()


def _as_input_item(item) -> dict:
    """Convert a returned output item into something acceptable as input.

    `status` is populated by the server and rejected on the way back in
    ("Unknown parameter: 'input[1].status'"), which is the one non-obvious
    part of feeding a turn's own output into the next request.
    """

    data = item.model_dump(exclude_none=True)
    data.pop("status", None)
    return data


def _run_turn(
    client,
    input_items: list[dict],
    tools: list[dict] | None,
    on_answer_delta=None,
    on_reasoning=None,
) -> TurnResult:
    """Run one model turn, streaming reasoning and answer text as they arrive.

    Uses the Responses API rather than chat completions, which is what makes
    the model's reasoning visible. Some endpoints reject reasoning_effort on
    /v1/chat/completions when function tools are present, and a chat completion
    never returns reasoning text at all. Here a reasoning summary streams as
    response.reasoning_summary_text.delta.

    Streaming is also what removes most of the *perceived* wait. The pipeline
    takes as long as it takes, but the technician watches it work instead of
    watching a spinner - and on a shop floor that difference decides whether
    the tool gets used at all.

    Answer text and reasoning arrive on separate event types here, so unlike
    the chat-completions version there is no need to guess whether text is an
    answer or a preamble, and nothing ever has to be retracted.
    """

    settings = get_settings()
    kwargs = {"model": chat_model_name(), "input": input_items}
    if settings.reasoning_effort:
        # Summaries are what get shown; effort is how hard it thinks first.
        # Left out entirely for non-reasoning models, which reject the field.
        kwargs["reasoning"] = {"effort": settings.reasoning_effort, "summary": settings.reasoning_summary}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    # Note: no temperature. Reasoning models reject it.

    reasoning_parts: list[str] = []
    answer_emitted = 0
    answer_parts: list[str] = []

    with client.responses.stream(**kwargs) as stream:
        for event in stream:
            kind = event.type

            if kind == "response.reasoning_summary_text.delta":
                reasoning_parts.append(event.delta)

            elif kind == "response.reasoning_summary_part.done":
                # One coherent block of reasoning finished; flush it as a unit
                # rather than dribbling half-sentences into the activity log.
                block = clean_reasoning("".join(reasoning_parts))
                reasoning_parts = []
                if block and on_reasoning:
                    on_reasoning(block)

            elif kind == "response.output_text.delta":
                answer_parts.append(event.delta)
                if on_answer_delta:
                    visible = streamable_prefix("".join(answer_parts))
                    if len(visible) > answer_emitted:
                        on_answer_delta(visible[answer_emitted:])
                        answer_emitted = len(visible)

        final = stream.get_final_response()

    # Anything left unflushed (no *_part.done, e.g. an interrupted stream).
    trailing = clean_reasoning("".join(reasoning_parts))
    if trailing and on_reasoning:
        on_reasoning(trailing)

    tool_calls = [
        {"id": item.call_id, "name": item.name, "arguments": item.arguments}
        for item in final.output
        if item.type == "function_call"
    ]
    content = final.output_text or ""
    output_items = [_as_input_item(item) for item in final.output]

    usage = {}
    final_usage = getattr(final, "usage", None)
    if final_usage is not None:
        usage = {
            "input_tokens": getattr(final_usage, "input_tokens", 0) or 0,
            "output_tokens": getattr(final_usage, "output_tokens", 0) or 0,
        }

    return TurnResult(content=content, tool_calls=tool_calls, output_items=output_items, usage=usage)


def _build_messages(
    query: str,
    conversation_tail: list[dict] | None,
    registry: CitationRegistry,
    intent: str = "question",
    machine_id: str | None = None,
) -> list[dict]:
    conversation = "\n".join(
        f"{message.get('role', 'user')}: {message.get('content', '')}"
        for message in (conversation_tail or [])[-6:]
    )
    seeded = registry.all_ordered()
    user_parts = []
    if machine_id:
        user_parts.append(machine_context(machine_id))
    if conversation:
        user_parts.append(f"Recent conversation:\n{conversation}")
    if seeded:
        seeded_blocks = "\n".join(f"[{c['id']}] {c['label']} ({c['scope_label']})" for c in seeded)
        user_parts.append(f"Citations already available from this session:\n{seeded_blocks}")
    if intent == "expand":
        user_parts.append(EXPAND_INSTRUCTION)
    else:
        user_parts.append(f"Question: {query}")

    return [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def answer_conversationally(
    query: str,
    selected_machine: str | None = None,
    conversation_tail: list[dict] | None = None,
    active_citations: list[dict] | None = None,
    on_progress=None,
    on_answer_delta=None,
    on_reasoning=None,
    on_stream_event=None,
    intent: str = "question",
) -> dict:
    """Answer a question (fresh or follow-up) via model-driven tool calls.

    The model decides which tools to call and when; run_full_search is the
    only tool that invokes the evidence judge.

    on_answer_delta, if given, receives fragments of the answer as the model
    writes them, already stripped of internal markers (see streamable_prefix).
    The fragments are for display only - the caller must still use the returned
    `answer`, which is the canonical text with citations renumbered.

    on_reasoning receives the model's own commentary from tool-calling turns -
    what it says it is about to look up and why. See _run_turn for why that is
    not the same as chain-of-thought.

    on_stream_event receives raw event dicts for the streaming client:
    {"type": "answer_mode", "mode": "full_search"} once when run_full_search
    executes (the client reroutes streamed text into the activity panel), and
    {"type": "answer_discard"} whenever text already streamed turns out not to
    be the final answer (pre-tool commentary, or a draft superseded by the
    forced verification rewrite).
    """

    machine = resolve_machine(query, selected_machine)
    if machine["needs_clarification"]:
        return {
            "answer": "Please specify which machine you want to ask about.",
            "machine_id": None,
            "needs_clarification": True,
            "sources": [],
            "cited_sources": [],
            "citations": [],
            "retrieved_chunks": [],
            "source_badges": [],
        }

    machine_id = machine["machine_id"]
    # Carry forward the citations a follow-up is allowed to reuse. This must
    # match what retrieval itself considers in scope - this machine PLUS the
    # shared general library. Comparing against machine_id alone would silently
    # drop every general-library citation and make the next turn start over.
    scope = retrieval_scope(machine_id)
    in_scope = set(scope if isinstance(scope, list) else [scope])
    seed_citations = [c for c in (active_citations or []) if c.get("machine_id") in in_scope]
    registry = CitationRegistry(seed_citations=seed_citations)
    ctx = ToolContext(machine_id=machine_id, registry=registry, on_progress=on_progress, query=query)

    messages = _build_messages(query, conversation_tail, registry, intent=intent, machine_id=machine_id)
    client = get_client()
    final_text = None
    # Expanding reuses the evidence the answer was already built on, so the
    # search tools are withheld for the turn rather than merely discouraged.
    turn_tools = _responses_tool_specs(EXPAND_TOOLS if intent == "expand" else None)

    full_search_announced = False

    def _announce_full_search() -> None:
        # Tells the streaming client this turn is a full-search turn (summary
        # display, draft text belongs in the activity panel). Emitted at most
        # once, the moment the tool actually runs.
        nonlocal full_search_announced
        if not full_search_announced and on_stream_event:
            on_stream_event({"type": "answer_mode", "mode": "full_search"})
        full_search_announced = True

    turn_delta_count = 0

    def _counted_delta(text: str) -> None:
        nonlocal turn_delta_count
        turn_delta_count += 1
        if on_answer_delta:
            on_answer_delta(text)

    def _discard_streamed_draft() -> None:
        # Text streamed during a turn that went on to call tools (or got the
        # verification demand) was never going to be the final answer; without
        # this the client shows it appended to the real one.
        nonlocal turn_delta_count
        if turn_delta_count and on_stream_event:
            on_stream_event({"type": "answer_discard"})
        turn_delta_count = 0

    # A genuinely fresh question (no prior answer in this session, no active
    # citations) is forced through run_full_search before the model gets a
    # turn, so the judge-checked full search always runs on a cold start.
    # Left to the model's own judgment, it will
    # sometimes decide a single search_documents call is good enough and
    # skip the judge entirely, even on the first question. The model can
    # still call further tools afterward (e.g. read_page_image) if this
    # isn't enough.
    #
    # conversation_tail is NOT a reliable "is this the first question" check
    # by itself: the frontend pushes the current user question onto the
    # message list before sending it, so conversation_tail always contains
    # at least that one entry even on a brand-new session. The real signal
    # is whether any assistant turn has happened yet.
    #
    # Expanding: pull the actual page text for the citations the answer being
    # expanded was built on, before the model gets a turn.
    #
    # This is not an optimisation, it is what makes the expansion honest. A new
    # turn only carries the conversation and a LIST of citation ids and labels -
    # the chunk text itself lived in the previous turn's tool output and is
    # gone. Without this the model would be elaborating from memory of its own
    # summary, which is exactly where invented detail comes from. Forcing the
    # reads puts the real pages back in front of it, wider than the excerpt that
    # was originally retrieved.
    if intent == "expand" and seed_citations:
        for index, citation in enumerate(registry.all_ordered()[:EXPAND_FORCED_READS]):
            call_id = f"forced_read_page_{index}"
            arguments = {"citation_id": citation["id"], "radius": 1}
            messages.extend(
                [
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": "read_page",
                        "arguments": json.dumps(arguments),
                    },
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": execute_tool("read_page", arguments, ctx),
                    },
                ]
            )

    has_prior_turn = any(message.get("role") == "assistant" for message in (conversation_tail or []))
    if intent != "expand" and not has_prior_turn and not seed_citations:
        forced_call_id = "forced_run_full_search"
        _announce_full_search()
        forced_result = execute_tool("run_full_search", {"query": query}, ctx)
        messages.extend(
            [
                {
                    "type": "function_call",
                    "call_id": forced_call_id,
                    "name": "run_full_search",
                    "arguments": json.dumps({"query": query}),
                },
                {
                    "type": "function_call_output",
                    "call_id": forced_call_id,
                    "output": forced_result,
                },
            ]
        )

    # Verification is demanded once, and only after run_full_search. That tool
    # returns evidence from a widened multi-pass search with an independent
    # judge attached, and the model treats it as settled: left alone, its tool
    # path on table lookups was exactly ["run_full_search"], never a look at the
    # page - so it could read the wrong row of a fifty-model matrix and state it
    # confidently, because the instruction to open the page was conditional on
    # feeling unsure and it never was.
    #
    # Deliberately scoped to this one tool. search_documents, read_page and
    # search_answer_history are the model's own judgement calls and stay that
    # way; forcing a vision call after every one of those would buy latency on
    # answers that never had a row to pick.
    verification_demanded = False

    def _tally_usage(turn_usage: dict) -> None:
        if not turn_usage:
            return
        tally = ctx.diagnostics.setdefault(
            "token_usage", {"input_tokens": 0, "output_tokens": 0, "answer_turns": 0}
        )
        tally["input_tokens"] += turn_usage.get("input_tokens", 0)
        tally["output_tokens"] += turn_usage.get("output_tokens", 0)
        tally["answer_turns"] += 1

    for _ in range(MAX_TOOL_ROUNDS):
        turn_delta_count = 0
        turn = _run_turn(
            client,
            messages,
            turn_tools,
            on_answer_delta=_counted_delta,
            on_reasoning=_reasoning_emitter(ctx, on_reasoning),
        )
        _tally_usage(turn.usage)
        if not turn.tool_calls:
            used = ctx.diagnostics.get("tool_path", [])
            if (
                not verification_demanded
                and "run_full_search" in used
                and "read_page_image" not in used
                # Nothing to open if the search found nothing.
                and registry.all_ordered()
            ):
                verification_demanded = True
                _discard_streamed_draft()
                _emit(ctx, "tool_call", "Checking the answer against the page itself...")
                # Carry the drafted answer and its reasoning items into the
                # verification turn. Without them the demand refers to "what
                # you are about to say" while the draft is no longer in
                # context, and the model re-derives the answer from scratch -
                # sometimes leading with the value it goes on to disclaim. For a
                # reasoning model the reasoning items are part of the chain.
                messages.extend(turn.output_items)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Before you answer: open the page behind the citation you are "
                            "relying on with read_page_image and confirm your value against it, "
                            "passing the claim you are checking via the verify argument. "
                            "Text extraction flattens tables into run-together lines, so a row "
                            "can look like it belongs to this machine when it "
                            "belongs to another one entirely. What you read off the page is "
                            "the answer: lead your reply with the verified value. If it "
                            "differs from what you were about to say, the page wins - state "
                            "the page's value and mention the discarded one only to say why "
                            "it does not apply. If the page contradicts you, or you "
                            "cannot tell which row applies to this machine, say that instead.\n\n"
                            "Then write your COMPLETE answer from scratch. The technician sees "
                            "only what you write after this check - your earlier draft is never "
                            "shown to them - so never refer back to it, never compress your "
                            "answer into a one-line confirmation, and do not open with the word "
                            "'Verified' or commentary about checking. Just answer the question, "
                            "in full. If the question was not about a specific value at all, "
                            "confirm what you cite and answer the question in full as normal."
                        ),
                    }
                )
                continue
            final_text = turn.content
            break

        # Carry the turn's own output forward verbatim, reasoning items
        # included - for a reasoning model those are part of the chain, and
        # dropping them makes the next turn start its thinking over.
        _discard_streamed_draft()
        messages.extend(turn.output_items)
        for call in turn.tool_calls:
            try:
                arguments = json.loads(call["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {}
            if call["name"] == "run_full_search":
                _announce_full_search()
            result_text = execute_tool(call["name"], arguments, ctx)
            messages.append(
                {
                    "type": "function_call_output",
                    "call_id": call["id"],
                    "output": result_text,
                }
            )

        messages = _trim_tool_context(messages)

    if final_text is None:
        _emit(ctx, "final_answer", "Reached the tool-call limit, writing the best answer so far...")
        ctx.diagnostics["termination_reason"] = "tool_round_limit"
        messages.append(
            {
                "role": "user",
                "content": (
                    "You have reached the tool-call limit for this turn. Answer now using "
                    "whatever you have gathered so far. If you believe you were close to a "
                    "confirmed answer, say so and ask the technician whether you should keep digging."
                ),
            }
        )
        limit_turn = _run_turn(
            client,
            messages,
            None,
            on_answer_delta=on_answer_delta,
            on_reasoning=_reasoning_emitter(ctx, on_reasoning),
        )
        _tally_usage(limit_turn.usage)
        final_text = limit_turn.content
    else:
        ctx.diagnostics["termination_reason"] = "completed"
        _emit(ctx, "final_answer", "Writing the final answer...")

    answer_text, used_general_knowledge = _strip_general_knowledge_markers(final_text)
    answer_text, offers_detail = _strip_more_detail_marker(answer_text)
    all_citations = registry.all_ordered()
    used_citations = _citations_used_in_answer(answer_text, all_citations)
    if used_citations:
        # Only show citations the answer actually used, renumbered for display.
        answer_text, used_citations = _renumber_answer_citations(answer_text, used_citations)

    sources = _source_rows(registry.chunks())
    # Fall back to fuzzy file/page matching if the model wrote prose instead
    # of [cite_n] markers, same as the pre-tool-calling behavior. Raw source
    # rows have no id/label (only build_citation output does) - db/history.py
    # add_answer() indexes citations by citation["id"] when persisting, so
    # this must be citation-shaped, not a bare source dict, or saving a
    # fuzzy-matched-only answer throws KeyError('id').
    cited_sources = used_citations or build_citations(_cited_source_rows(answer_text, sources))
    tagged_cited_sources = _tag_citation_roles(cited_sources)

    # Full-search turns split into a stored full answer and a displayed
    # summary. The summary is written AFTER renumbering so its [cite_n]
    # markers are the display ids; markers it invents are stripped inside
    # summarize_full_answer. When the summary fails, or barely shortens the
    # answer, the full text simply stays the display text - the split is an
    # optimization for the reader, never a risk to the turn.
    is_full_search = "run_full_search" in ctx.diagnostics.get("tool_path", [])
    full_answer_text = None
    has_full_detail = False
    if is_full_search and answer_text:
        _emit(ctx, "final_answer", "Condensing for display...")
        summary = summarize_full_answer(client, answer_text, query)
        if summary and len(answer_text) - len(summary) > 200:
            full_answer_text = answer_text
            answer_text = summary
            has_full_detail = True
        # The tap-to-expand pipeline path is replaced by the instant reveal.
        offers_detail = False

    run_full_search_diagnostics = ctx.diagnostics.get("run_full_search")
    return {
        "answer": answer_text,
        "full_answer": full_answer_text,
        "has_full_detail": has_full_detail,
        "machine_id": machine_id,
        "needs_clarification": False,
        # Light-path turns only: drives the legacy tap-to-expand offer. Not
        # persisted - a reused/history answer should not carry a button that
        # promises a follow-up the model never planned.
        "offers_detail": offers_detail,
        "sources": sources,
        "cited_sources": tagged_cited_sources,
        # DB ledger falls back to every citation seen this turn (not the
        # fuzzy-matched subset) only when nothing was cited or fuzzy-matched
        # at all - matches the pre-tool-calling fallback priority.
        "citations": tagged_cited_sources or all_citations,
        "retrieved_chunks": registry.preview_rows(),
        "source_badges": _final_badges(tagged_cited_sources, used_general_knowledge),
        "retrieval_diagnostics": {
            "tool_path": ctx.diagnostics.get("tool_path", []),
            "progress_events": ctx.diagnostics.get("progress_events", []),
            "reasoning": ctx.diagnostics.get("reasoning", []),
            # How the turn ended and what the answering turns cost. Judge and
            # vision sub-calls run on chat completions with their own usage
            # blocks and are not counted here.
            "termination_reason": ctx.diagnostics.get("termination_reason"),
            "token_usage": ctx.diagnostics.get("token_usage") or {},
            "run_full_search": run_full_search_diagnostics,
            # Flattened to the top level for app/db/diagnostics.py. Absent
            # (None/[]) when this turn never called run_full_search - that is
            # real information (a lighter tool path), not a missing value.
            "docs_answer_found": (run_full_search_diagnostics or {}).get("docs_answer_found"),
            "judge": (run_full_search_diagnostics or {}).get("judge"),
            "passes": (run_full_search_diagnostics or {}).get("passes", []),
        },
    }
