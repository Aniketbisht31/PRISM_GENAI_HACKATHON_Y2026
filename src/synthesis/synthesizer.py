"""
Session-Aware Synthesizer — generates grounded answers with citations,
supports refinement with late constraints, and handles presentation-only requests.
All state is ephemeral, scoped to one active session, never persisted.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from src.llm.adapter import get_adapter, BaseLLMAdapter, LLMResponse
from src.telemetry.logger import get_logger, TelemetryLogger
from src.provenance import (
    claim_matches_answer,
    evidence_matches_query_scope,
    evidence_supports_claim,
    normalize_section,
    required_named_entities,
)


@dataclass
class AnswerVersion:
    """One version of an answer within a session."""
    version: int
    answer: str
    citations: list[dict]
    uncertainty: Optional[str]
    sub_queries: list[str]
    retrieved_chunks: list[dict]
    is_refinement: bool = False


@dataclass
class SessionState:
    """Ephemeral session state — never persisted, never shared across sessions."""
    session_id: str
    answer_versions: list[AnswerVersion] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    sub_query_history: list[str] = field(default_factory=list)
    all_retrieved_chunks: list[dict] = field(default_factory=list)

    @property
    def current_version(self) -> int:
        return len(self.answer_versions)

    @property
    def latest_answer(self) -> Optional[AnswerVersion]:
        return self.answer_versions[-1] if self.answer_versions else None


@dataclass
class SynthesisResult:
    """Output schema for synthesis."""
    retrieval_events: list[dict]
    sub_queries: list[str]
    answer: str
    citations: list[dict]
    uncertainty: Optional[str]
    answer_version: int
    session_id: str


class SessionSynthesizer:
    """
    Maintains in-memory session state and produces grounded answers.
    """

    def __init__(self):
        self.adapter: BaseLLMAdapter = get_adapter()
        self.logger: TelemetryLogger = get_logger()
        self._sessions: dict[str, SessionState] = {}
        self._synth_prompt = self.adapter.load_prompt_template("prompts/synthesize.txt")
        self._reformat_prompt = self.adapter.load_prompt_template("prompts/reformat.txt")

    def _get_or_create_session(self, session_id: Optional[str] = None) -> SessionState:
        if session_id is None:
            session_id = str(uuid.uuid4())
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]

    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    def create_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = SessionState(session_id=session_id)
        return session_id

    def clear_session(self, session_id: str) -> None:
        """Remove a session from memory."""
        self._sessions.pop(session_id, None)

    def synthesize(
        self,
        session_id: str,
        sub_queries: list[str],
        retrieved_chunks: list[dict],
        retrieval_events: list[dict],
    ) -> SynthesisResult:
        """
        Generate a grounded answer from retrieved chunks.
        Every claim must be cited, and uncertainty must be flagged.
        """
        session = self._get_or_create_session(session_id)

        # Format chunks for the prompt
        chunks_text = self._format_chunks(retrieved_chunks)
        sub_queries_text = "\n".join(f"- {q}" for q in sub_queries)

        # Prior context for continuity
        prior_context = ""
        if session.latest_answer:
            prior_context = session.latest_answer.answer

        # Build the prompt from template
        prompt = self._synth_prompt.format(
            chunks=chunks_text,
            sub_queries=sub_queries_text,
            prior_context=prior_context or "(No prior answer — this is the first query in this session.)",
        )

        if retrieved_chunks:
            try:
                response = self.adapter.complete(
                    prompt=prompt,
                    system_prompt="You are a grounded answer synthesis engine. Answer ONLY from the provided chunks.",
                    json_mode=True,
                    temperature=0.1,
                    max_tokens=2048,
                )
                self.logger.log_llm_call(
                    session_id=session_id,
                    purpose="synthesis",
                    model=response.model,
                    usage=response.usage,
                    latency_ms=response.latency_ms,
                )
                parsed = self._parse_response(response.content)
            except Exception as error:
                self.logger.log_llm_failure(session_id, "synthesis", error)
                parsed = {
                    "answer": "",
                    "citations": [],
                    "uncertainty": "The model could not complete synthesis; the response uses only exact retrieved source excerpts.",
                }
        else:
            missing_topics = ", ".join(sub_queries) or "the requested information"
            parsed = {
                "answer": f"Based on the available documents, there is no information about {missing_topics}.",
                "citations": [],
                "uncertainty": f"No supporting corpus chunks were retrieved for: {missing_topics}.",
            }
        answer = parsed.get("answer", "Unable to generate answer from available documents.")
        citations = parsed.get("citations", [])
        uncertainty = parsed.get("uncertainty")

        # Validate citations — ensure no hallucinated doc IDs
        chunks_by_key = {
            (c["doc_id"], self._normalize_section(c.get("section", ""))): c
            for c in retrieved_chunks
        }
        valid_doc_ids = {key: chunk.get("section", "") for key, chunk in chunks_by_key.items()}
        validated_citations = []
        for cit in citations:
            cit_key = (cit.get("doc_id", ""), self._normalize_section(cit.get("section", "")))
            claim = str(cit.get("claim", "")).strip()
            chunk = chunks_by_key.get(cit_key, {})
            source = f"{chunk.get('title', '')} {chunk.get('text', '')}"
            scoped_queries = [q for q in sub_queries if required_named_entities(q)]
            scope_matches = (
                not scoped_queries
                or any(evidence_matches_query_scope(query, source) for query in scoped_queries)
            )
            if (cit_key in valid_doc_ids and claim and evidence_supports_claim(claim, source)
                    and scope_matches):
                validated_citations.append({**cit, "section": valid_doc_ids[cit_key]})
            # Drop hallucinated citations silently

        answer, removed_unsupported = self._keep_supported_sentences(
            answer, validated_citations, chunks_by_key
        )
        if answer == "I could not verify an answer from the retrieved corpus sections.":
            answer, extracted_citations, extraction_uncertainty = self._extractive_fallback(
                sub_queries, retrieved_chunks
            )
            if extracted_citations:
                validated_citations = extracted_citations
                uncertainty = uncertainty or extraction_uncertainty
        if removed_unsupported:
            uncertainty = uncertainty or "Some generated statements could not be tied to retrieved corpus sections and were omitted."

        # Create answer version
        version_num = session.current_version + 1
        answer_version = AnswerVersion(
            version=version_num,
            answer=answer,
            citations=validated_citations,
            uncertainty=uncertainty,
            sub_queries=sub_queries,
            retrieved_chunks=retrieved_chunks,
            is_refinement=False,
        )

        # Update session state
        session.answer_versions.append(answer_version)
        session.citations.extend(validated_citations)
        session.sub_query_history.extend(sub_queries)
        session.all_retrieved_chunks.extend(retrieved_chunks)

        # Log answer version
        self.logger.log_answer_version(
            session_id=session_id,
            version=version_num,
            answer=answer,
            citations=validated_citations,
            uncertainty=uncertainty,
            is_refinement=False,
        )

        return SynthesisResult(
            retrieval_events=retrieval_events,
            sub_queries=sub_queries,
            answer=answer,
            citations=validated_citations,
            uncertainty=uncertainty,
            answer_version=version_num,
            session_id=session_id,
        )

    def refine_with_late_constraint(
        self,
        session_id: str,
        new_sub_queries: list[str],
        new_retrieved_chunks: list[dict],
        new_retrieval_events: list[dict],
    ) -> SynthesisResult:
        """
        Refine the current answer with a late-arriving detail.
        Does NOT restart from scratch — preserves prior facts/citations
        while updating only affected claims with the new delta.
        """
        session = self._get_or_create_session(session_id)

        if not session.latest_answer:
            # No prior answer to refine — treat as new synthesis
            return self.synthesize(session_id, new_sub_queries, new_retrieved_chunks, new_retrieval_events)

        # Combine prior chunks with new delta chunks
        prior_answer = session.latest_answer
        # Only use new chunks for the prompt — prior context carries the preserved answer
        chunks_text = self._format_chunks(new_retrieved_chunks)
        sub_queries_text = "\n".join(f"- {q}" for q in new_sub_queries)
        prior_context = (
            f"PRIOR ANSWER (Version {prior_answer.version}):\n{prior_answer.answer}\n\n"
            f"PRIOR CITATIONS: {json.dumps(prior_answer.citations)}\n\n"
            f"INSTRUCTION: Refine the prior answer by incorporating the new information below. "
            f"PRESERVE all previously cited facts and citations. Only ADD or UPDATE claims "
            f"affected by the new retrieved chunks."
        )

        prompt = self._synth_prompt.format(
            chunks=chunks_text,
            sub_queries=sub_queries_text,
            prior_context=prior_context,
        )

        try:
            response = self.adapter.complete(
                prompt=prompt,
                system_prompt="You are a grounded answer synthesis engine. Refine the prior answer with new information. Preserve all prior citations.",
                json_mode=True,
                temperature=0.1,
                max_tokens=2048,
            )
            self.logger.log_llm_call(
                session_id=session_id,
                purpose="synthesis_refinement",
                model=response.model,
                usage=response.usage,
                latency_ms=response.latency_ms,
            )
            parsed = self._parse_response(response.content)
            answer = parsed.get("answer", prior_answer.answer)
            citations = parsed.get("citations", prior_answer.citations)
            uncertainty = parsed.get("uncertainty", prior_answer.uncertainty)
        except Exception as error:
            self.logger.log_llm_failure(session_id, "synthesis_refinement", error)
            fallback_answer, fallback_citations, fallback_uncertainty = self._extractive_fallback(
                new_sub_queries, new_retrieved_chunks
            )
            answer = prior_answer.answer
            if fallback_citations:
                answer = f"{answer} {fallback_answer}"
            citations = list(prior_answer.citations) + fallback_citations
            uncertainty = fallback_uncertainty or "The model could not complete refinement; prior verified facts were preserved."

        # Validate citations against ALL chunks (prior + new)
        all_chunks = session.all_retrieved_chunks + new_retrieved_chunks
        chunks_by_key = {
            (c["doc_id"], self._normalize_section(c.get("section", ""))): c
            for c in all_chunks
        }
        valid_doc_ids = {key: chunk.get("section", "") for key, chunk in chunks_by_key.items()}
        validated_citations = []
        for cit in citations:
            cit_key = (cit.get("doc_id", ""), self._normalize_section(cit.get("section", "")))
            claim = str(cit.get("claim", "")).strip()
            chunk = chunks_by_key.get(cit_key, {})
            source = f"{chunk.get('title', '')} {chunk.get('text', '')}"
            scoped_queries = [q for q in new_sub_queries if required_named_entities(q)]
            scope_matches = (
                not scoped_queries
                or any(evidence_matches_query_scope(query, source) for query in scoped_queries)
            )
            if (cit_key in valid_doc_ids and claim and evidence_supports_claim(claim, source)
                    and scope_matches):
                validated_citations.append({**cit, "section": valid_doc_ids[cit_key]})

        prior_citations = list(prior_answer.citations)
        for citation in validated_citations:
            if citation not in prior_citations:
                prior_citations.append(citation)
        validated_citations = prior_citations
        answer, removed_unsupported = self._keep_supported_sentences(
            answer, validated_citations, chunks_by_key
        )
        if removed_unsupported:
            uncertainty = uncertainty or "Some generated statements could not be tied to retrieved corpus sections and were omitted."

        version_num = session.current_version + 1
        answer_version = AnswerVersion(
            version=version_num,
            answer=answer,
            citations=validated_citations,
            uncertainty=uncertainty,
            sub_queries=new_sub_queries,
            retrieved_chunks=all_chunks,
            is_refinement=True,
        )

        session.answer_versions.append(answer_version)
        session.citations.extend(validated_citations)
        session.sub_query_history.extend(new_sub_queries)
        session.all_retrieved_chunks.extend(new_retrieved_chunks)

        self.logger.log_answer_version(
            session_id=session_id,
            version=version_num,
            answer=answer,
            citations=validated_citations,
            uncertainty=uncertainty,
            is_refinement=True,
        )

        return SynthesisResult(
            retrieval_events=new_retrieval_events,
            sub_queries=new_sub_queries,
            answer=answer,
            citations=validated_citations,
            uncertainty=uncertainty,
            answer_version=version_num,
            session_id=session_id,
        )

    def handle_presentation_only(
        self,
        session_id: str,
        request: str,
    ) -> SynthesisResult:
        """
        Handle reformatting/shortening/translation requests.
        Transforms existing session context with ZERO new corpus queries.
        """
        session = self._get_or_create_session(session_id)

        if not session.latest_answer:
            return SynthesisResult(
                retrieval_events=[],
                sub_queries=[],
                answer="No prior answer to reformat. Please ask a question first.",
                citations=[],
                uncertainty=None,
                answer_version=0,
                session_id=session_id,
            )

        prior_answer = session.latest_answer
        citations_json = json.dumps(prior_answer.citations)

        prompt = self._reformat_prompt.format(
            previous_answer=prior_answer.answer,
            request=request,
            citations=citations_json,
        )

        try:
            response = self.adapter.complete(
                prompt=prompt,
                system_prompt="You are a text reformatting assistant. Only reformat, never add new facts.",
                json_mode=True,
                temperature=0.1,
                max_tokens=2048,
            )
            self.logger.log_llm_call(
                session_id=session_id,
                purpose="presentation_only",
                model=response.model,
                usage=response.usage,
                latency_ms=response.latency_ms,
            )
            parsed = self._parse_response(response.content)
        except Exception as error:
            self.logger.log_llm_failure(session_id, "presentation_only", error)
            parsed = {"answer": prior_answer.answer}

        self.logger.log_presentation_only(
            session_id=session_id,
            request=request,
        )

        answer = parsed.get("answer", prior_answer.answer)
        # Presentation can change wording and layout, but cannot add, remove,
        # or fabricate source references.
        citations = prior_answer.citations
        chunks_by_key = {
            (c["doc_id"], self._normalize_section(c.get("section", ""))): c
            for c in prior_answer.retrieved_chunks
        }
        valid_doc_sections = {
            (citation.get("doc_id", ""), citation.get("section", ""))
            for citation in citations
        }
        answer, removed = self._keep_supported_sentences(answer, citations, chunks_by_key)
        if removed:
            answer = prior_answer.answer

        version_num = session.current_version + 1
        answer_version = AnswerVersion(
            version=version_num,
            answer=answer,
            citations=citations,
            uncertainty=prior_answer.uncertainty,
            sub_queries=[],
            retrieved_chunks=[],
            is_refinement=False,
        )

        session.answer_versions.append(answer_version)

        self.logger.log_answer_version(
            session_id=session_id,
            version=version_num,
            answer=answer,
            citations=citations,
            uncertainty=prior_answer.uncertainty,
            is_refinement=False,
        )

        return SynthesisResult(
            retrieval_events=[],
            sub_queries=[],
            answer=answer,
            citations=citations,
            uncertainty=None,
            answer_version=version_num,
            session_id=session_id,
        )

    def _format_chunks(self, chunks: list[dict]) -> str:
        """Format retrieved chunks for inclusion in the synthesis prompt."""
        if not chunks:
            return "(No chunks retrieved.)"
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[Chunk {i}] doc_id={chunk['doc_id']}, section={chunk['section']}, "
                f"title={chunk.get('title', 'N/A')}\n{chunk['text']}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _keep_cited_sentences(
        answer: str,
        valid_doc_sections: set[tuple[str, str]] | dict[tuple[str, str], str],
    ) -> tuple[str, bool]:
        """Keep only individually cited sentences; never inherit trailing citations."""
        canonical_sections = {
            (doc_id, SessionSynthesizer._normalize_section(section)):
                (section if not isinstance(valid_doc_sections, dict)
                 else valid_doc_sections[(doc_id, section)])
            for doc_id, section in valid_doc_sections
        }
        segments = re.split(r"(?<=[.!?])\s+|\n+", answer.strip())
        citation_pattern = re.compile(r"\[(Doc_[^\]\s]+)\s+([^\]]+)\]")
        kept: list[str] = []
        removed = False
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            references = citation_pattern.findall(segment)
            normalized = [
                (doc_id, SessionSynthesizer._normalize_section(section))
                for doc_id, section in references
            ]
            if references and all(ref in canonical_sections for ref in normalized):
                canonical = citation_pattern.sub(
                    lambda match: f"[{match.group(1)} {canonical_sections[(match.group(1), SessionSynthesizer._normalize_section(match.group(2)))]}]",
                    segment,
                )
                if citation_pattern.sub("", segment).strip(" \t:;,.!?-\u2022\"'"):
                    kept.append(canonical)
                else:
                    removed = True
            else:
                removed = True
        if kept:
            return " ".join(kept), removed
        return "I could not verify an answer from the retrieved corpus sections.", True

    @staticmethod
    def _keep_supported_sentences(
        answer: str,
        citations: list[dict],
        chunks_by_key: dict[tuple[str, str], dict],
    ) -> tuple[str, bool]:
        """Drop uncited or unsupported sentences instead of inheriting citations."""
        claims: dict[tuple[str, str], list[str]] = {}
        for citation in citations:
            key = (
                str(citation.get("doc_id", "")),
                SessionSynthesizer._normalize_section(citation.get("section", "")),
            )
            claim = str(citation.get("claim", "")).strip()
            chunk = chunks_by_key.get(key, {})
            source = f"{chunk.get('title', '')} {chunk.get('text', '')}"
            if key in chunks_by_key and claim and evidence_supports_claim(claim, source):
                claims.setdefault(key, []).append(claim)

        segments = re.split(r"(?<=[.!?])\s+|\n+", answer.strip())
        pattern = re.compile(r"\[(Doc_[^\]\s]+)\s+([^\]]+)\]")
        kept: list[str] = []
        removed = False
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            references = pattern.findall(segment)
            keys = [
                (doc_id, SessionSynthesizer._normalize_section(section))
                for doc_id, section in references
            ]
            if not references or any(key not in claims for key in keys):
                removed = True
                continue
            factual_text = pattern.sub("", segment).strip(" \t:;,.!?-•\"'")
            supported = True
            for key in keys:
                chunk = chunks_by_key[key]
                source = f"{chunk.get('title', '')} {chunk.get('text', '')}"
                if not evidence_supports_claim(factual_text, source):
                    supported = False
                    break
                if not any(
                    claim_matches_answer(factual_text, claim)
                    for claim in claims[key]
                ):
                    supported = False
                    break
            if supported:
                kept.append(segment)
            else:
                removed = True

        if kept:
            return " ".join(kept), removed
        return "I could not verify an answer from the retrieved corpus sections.", True

    @staticmethod
    def _extractive_fallback(
        sub_queries: list[str], retrieved_chunks: list[dict]
    ) -> tuple[str, list[dict], Optional[str]]:
        """Return exact, query-relevant source sentences when generation fails verification."""
        stop_words = {
            "a", "an", "and", "are", "at", "be", "by", "can", "does", "for",
            "from", "how", "in", "is", "it", "of", "on", "or", "the", "their",
            "to", "what", "when", "where", "which", "who", "with", "would",
        }
        selections: list[tuple[int, int, str, dict]] = []
        missing: list[str] = []
        for query_index, query in enumerate(sub_queries):
            terms = {
                token for token in re.findall(r"[a-z0-9]+", query.casefold())
                if token not in stop_words and len(token) > 1
            }
            best: Optional[tuple[int, int, str, dict]] = None
            for chunk_index, chunk in enumerate(retrieved_chunks):
                source = f"{chunk.get('title', '')} {chunk.get('text', '')}"
                if not evidence_matches_query_scope(query, source):
                    continue
                for sentence in re.split(r"(?<=[.!?])\s+|\n+", str(chunk.get("text", ""))):
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    sentence_terms = set(re.findall(r"[a-z0-9]+", sentence.casefold()))
                    overlap = len(terms & sentence_terms)
                    if overlap == 0 or not evidence_supports_claim(sentence, source):
                        continue
                    candidate = (overlap, -chunk_index, sentence, chunk)
                    if best is None or candidate[:2] > best[:2]:
                        best = candidate
            if best is None:
                missing.append(query)
            else:
                selections.append((query_index, best[1], best[2], best[3]))

        answer_sentences: list[str] = []
        citations: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        for _, _, sentence, chunk in selections:
            doc_id = str(chunk.get("doc_id", ""))
            section = normalize_section(chunk.get("section", ""))
            key = (doc_id, section, sentence)
            if key in seen:
                continue
            seen.add(key)
            answer_sentences.append(f'"{sentence}" [{doc_id} {section}]')
            citations.append({"doc_id": doc_id, "section": section, "claim": sentence})

        if not citations:
            topic = ", ".join(missing or sub_queries) or "the requested information"
            return (
                f"I could not verify an answer about {topic} from the retrieved corpus sections.",
                [],
                f"No exact, query-relevant source excerpt was found for: {topic}.",
            )
        uncertainty = (
            f"The retrieved sections did not provide an exact supporting excerpt for: {', '.join(missing)}."
            if missing else None
        )
        return " ".join(answer_sentences), citations, uncertainty

    @staticmethod
    def _normalize_section(section: str) -> str:
        """Normalize common UTF-8/Windows mojibake while matching source markers."""
        return normalize_section(section)

    def _parse_response(self, content: str) -> dict:
        """Defensively parse LLM JSON response."""
        content = content.strip()
        # Strip code fences
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from the content
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {
                "answer": content,
                "citations": [],
                "uncertainty": "Failed to parse structured response from LLM.",
            }
