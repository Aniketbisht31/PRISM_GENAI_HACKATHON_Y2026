"""Helpers for canonical source markers and conservative evidence checks."""

from __future__ import annotations

import re


def normalize_section(value: object) -> str:
    """Repair replacement characters and common UTF-8 mojibake in section IDs."""
    section = str(value or "").strip()
    section = section.replace("\ufffd", "\u00a7")
    for broken in ("\u00c3\u0192\u00e2\u20ac\u0161\u00c3\u201a\u00c2\u00a7", "\u00c3\u201a\u00c2\u00a7", "\u00c2\u00a7"):
        section = section.replace(broken, "\u00a7")
    return section


def tokenize_claim(text: str) -> list[str]:
    normalized = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    return re.findall(r"[a-z0-9]+", normalized)


def claim_matches_answer(answer: str, claim: str) -> bool:
    """Answer facts must repeat their citation's evidence quote exactly."""
    normalize = lambda value: " ".join(str(value).strip().strip("\"\'.,;:!? ").split()).casefold()
    return bool(normalize(claim)) and normalize(answer) == normalize(claim)


def required_named_entities(query: str) -> list[str]:
    """Extract corpus-style named venue references that must match the evidence."""
    return re.findall(r"\bVenue\s+[A-Z]\b", str(query), flags=re.IGNORECASE)


def evidence_matches_query_scope(query: str, evidence: str) -> bool:
    """Prevent a supported quote about a different explicitly named venue."""
    entities = required_named_entities(query)
    normalized_evidence = " ".join(str(evidence).split()).casefold()
    return all(" ".join(entity.split()).casefold() in normalized_evidence for entity in entities)


def evidence_supports_claim(claim: str, evidence: str) -> bool:
    """Accept a claim only when it is an exact excerpt of the cited source."""
    normalize = lambda value: " ".join(str(value).split()).casefold()
    normalized_claim = normalize(claim).strip("\"'????")
    normalized_evidence = normalize(evidence)
    return bool(normalized_claim) and normalized_claim in normalized_evidence
