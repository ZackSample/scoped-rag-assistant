"""Exact-match search over document chunks.

Two stages on purpose:

1. SQL narrows the corpus to rows that contain at least one query term (see
   vector_store.keyword_candidates), so only a few hundred rows ever leave the
   database.
2. This module scores that much smaller candidate set precisely, with word
   boundaries and rarity weighting.
"""

import math
import re
from collections import Counter

from app.config import get_settings
from app.rag.vector_store import keyword_candidates


CODE_PATTERNS = [
    # Machine docs often use alarms, oil grades, and part-like codes verbatim.
    re.compile(r"\bE\d{2,4}\b", re.IGNORECASE),
    re.compile(r"\bAL[- ]?\d{1,4}\b", re.IGNORECASE),
    re.compile(r"\bISO\s*VG\s*\d+\b", re.IGNORECASE),
    re.compile(r"\b[A-Z]{1,5}[-_]?\d{2,6}\b", re.IGNORECASE),
]

STOPWORDS = {
    "about", "after", "also", "and", "any", "are", "can", "does", "find",
    "for", "from", "have", "here", "how", "info", "look", "more", "other",
    "show", "that", "the", "this", "what", "when", "where", "with",
}

# How many candidate rows SQL returns for precise scoring. Generous relative to
# keyword_candidate_k (the number that survives) so the precise pass has real
# choices, but bounded so a single-common-word query cannot pull the corpus.
_CANDIDATE_MULTIPLIER = 8
_CANDIDATE_CEILING = 600


def _query_terms(query: str) -> tuple[list[str], list[str]]:
    lowered = query.lower()
    # Drop filler words so common phrases do not dominate exact matching.
    terms = [
        token
        for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_\-]{2,}", lowered)
        if token not in STOPWORDS
    ]
    codes = []
    for pattern in CODE_PATTERNS:
        codes.extend(match.group(0).lower() for match in pattern.finditer(query))
    return sorted(set(terms)), sorted(set(codes))


def _word_count(haystack: str, needle: str) -> int:
    """Count whole-word occurrences of needle in haystack.

    Substring counting was wrong in a way that quietly skewed ranking: "oil"
    matched "boiler" and "spoil", and because scores were a raw tally, a page
    that merely used a common substring often outscored the page that used the
    actual term. Hyphens and underscores are treated as word characters so
    part numbers like "AL-12" survive.
    """

    return len(re.findall(rf"(?<![a-z0-9_-]){re.escape(needle)}(?![a-z0-9_-])", haystack))


def _rarity_weights(candidates: list[dict], needles: list[str]) -> dict[str, float]:
    """Weight each needle by how rare it is across the candidate set.

    Without this every term counted the same, so "temperature" in a question
    about "melt temperature setpoint" carried as much weight as "setpoint"
    even though half the corpus mentions temperature. Document frequency
    within the candidate set is a decent stand-in for corpus-wide IDF and
    needs no precomputed statistics.
    """

    total = len(candidates) or 1
    weights = {}
    for needle in needles:
        hits = sum(1 for candidate in candidates if needle in candidate["_haystack"])
        # +1 keeps a needle in every candidate at a non-zero weight rather than
        # discarding it entirely.
        weights[needle] = math.log(1 + total / (1 + hits))
    return weights


def keyword_search(
    query: str, machine_id: str | list[str] | None = None, limit: int | None = None
) -> list[dict]:
    settings = get_settings()
    terms, codes = _query_terms(query)
    if not terms and not codes:
        return []

    max_results = limit or settings.keyword_candidate_k
    candidates = keyword_candidates(
        needles=sorted(set(terms) | set(codes)),
        machine_id=machine_id,
        limit=min(max_results * _CANDIDATE_MULTIPLIER, _CANDIDATE_CEILING),
    )
    if not candidates:
        return []

    for candidate in candidates:
        metadata = candidate["metadata"]
        # Filenames and folder names carry real signal, so they are searched
        # alongside the text.
        candidate["_haystack"] = " ".join(
            [
                (candidate["text"] or "").lower(),
                str(metadata.get("source_file") or "").lower(),
                str(metadata.get("relative_path") or "").lower(),
                str(metadata.get("document_type") or "").lower(),
            ]
        )

    weights = _rarity_weights(candidates, terms)
    phrase = query.lower().strip()
    scored = []

    for candidate in candidates:
        haystack = candidate.pop("_haystack")
        score = 0.0
        matched_terms = Counter()

        for term in terms:
            count = _word_count(haystack, term)
            if count:
                matched_terms[term] = count
                # Repeated hits help, but are capped so a noisy page that
                # happens to repeat one word cannot outrank a precise match.
                score += min(count, 6) * weights.get(term, 1.0)

        for code in codes:
            count = _word_count(haystack, code)
            if count:
                # Exact code matches are weighted heavily: an alarm number or
                # part number in a question is almost always the whole point.
                score += 12 + min(count, 5)

        if phrase and len(phrase) > 8 and phrase in haystack:
            score += 10
        if len(matched_terms) >= 2:
            # Covering several distinct query terms beats hammering one.
            score += len(matched_terms) * 2

        if score:
            scored.append({**candidate, "score": score / 100, "retrieval_method": "keyword"})

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:max_results]
