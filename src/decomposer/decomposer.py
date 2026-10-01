"""
Multi-Intent Decomposer — breaks compound utterances into independent sub-queries.
Caps at 4 sub-queries, deduplicates near-identical ones using embedding cosine similarity.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.llm.adapter import get_adapter, BaseLLMAdapter
from src.telemetry.logger import get_logger, TelemetryLogger


@dataclass
class DecompositionResult:
    """Output of multi-intent decomposition."""
    sub_queries: list[str]
    reasoning: str
    original_utterance: str
    duplicates_removed: int = 0


class MultiIntentDecomposer:
    """Decomposes compound utterances into searchable sub-queries."""

    MAX_SUB_QUERIES = 4
    SIMILARITY_THRESHOLD = 0.9
    INCREMENTAL_SIMILARITY_THRESHOLD = 0.97

    def __init__(self, embedding_model: Optional[SentenceTransformer] = None):
        self.adapter: BaseLLMAdapter = get_adapter()
        self.logger: TelemetryLogger = get_logger()
        self._prompt_template = self.adapter.load_prompt_template("prompts/decompose.txt")
        self._embedding_model = embedding_model or SentenceTransformer("all-MiniLM-L6-v2")

    def decompose(self, utterance: str, session_id: str) -> DecompositionResult:
        """
        Decompose an utterance into sub-queries.
        Synchronous — called after controller decides to RETRIEVE.
        """
        if not utterance or not utterance.strip():
            return DecompositionResult(
                sub_queries=[], reasoning="Empty utterance.", original_utterance=utterance
            )

        prompt = self._prompt_template.replace("{utterance}", utterance)


        # Attempt LLM call with one retry on parse failure
        parsed = None
        last_error = None
        for attempt in range(2):
            try:
                response = self.adapter.complete(
                    prompt=prompt,
                    system_prompt="Decompose the user utterance into sub-queries. Respond in JSON only.",
                    json_mode=True,
                    temperature=0.1,
                    max_tokens=512,
                )

                # Log LLM call
                self.logger.log_llm_call(
                    session_id=session_id,
                    purpose="decomposition",
                    model=response.model,
                    usage=response.usage,
                    latency_ms=response.latency_ms,
                )

                parsed = self._parse_json(response.content)
                if parsed and parsed.get("sub_queries"):
                    break
            except Exception as error:
                last_error = error
                if attempt == 0:
                    continue  # Retry once
                break

        if last_error is not None:
            self.logger.log_llm_failure(session_id, "decomposition", last_error)

        # Fallback: return original utterance as single sub-query
        if not parsed or not parsed.get("sub_queries"):
            result = DecompositionResult(
                sub_queries=[utterance],
                reasoning="Fallback: LLM did not return valid decomposition.",
                original_utterance=utterance,
            )
            self.logger.log_decomposition(
                session_id=session_id,
                utterance=utterance,
                sub_queries=result.sub_queries,
                reasoning=result.reasoning,
            )
            return result

        sub_queries = parsed["sub_queries"]
        reasoning = parsed.get("reasoning", "")

        # Ensure all sub_queries are strings
        sub_queries = [str(q) for q in sub_queries if q]

        # Step 1: Deduplicate near-identical queries
        sub_queries, dedup_removed = self._deduplicate_queries(sub_queries)

        # Step 2: Cap at MAX_SUB_QUERIES by merging least-distinct
        sub_queries = self._merge_excess_queries(sub_queries)

        result = DecompositionResult(
            sub_queries=sub_queries,
            reasoning=reasoning,
            original_utterance=utterance,
            duplicates_removed=dedup_removed,
        )

        # Log decomposition
        self.logger.log_decomposition(
            session_id=session_id,
            utterance=utterance,
            sub_queries=result.sub_queries,
            reasoning=result.reasoning,
            dedup_removed=dedup_removed,
        )

        return result

    @staticmethod
    def _parse_json(content: str) -> dict:
        """Defensively parse JSON, stripping code fences."""
        content = content.strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {}

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two embedding vectors."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def _deduplicate_queries(self, queries: list[str]) -> tuple[list[str], int]:
        """Remove near-duplicate queries using embedding cosine similarity > 0.9."""
        if len(queries) <= 1:
            return queries, 0

        embeddings = self._embedding_model.encode(queries)
        unique_queries: list[str] = []
        unique_embeddings: list[np.ndarray] = []
        duplicates_removed = 0

        for i, query in enumerate(queries):
            is_duplicate = False
            emb = embeddings[i]
            for u_emb in unique_embeddings:
                sim = self._cosine_similarity(emb, u_emb)
                if sim >= self.SIMILARITY_THRESHOLD:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_queries.append(query)
                unique_embeddings.append(emb)
            else:
                duplicates_removed += 1

        return unique_queries, duplicates_removed

    def only_new_queries(self, queries: list[str], previously_retrieved: list[str]) -> list[str]:
        """Return queries that add a distinct intent beyond earlier stream searches."""
        if not previously_retrieved or not queries:
            return list(queries)

        all_text = previously_retrieved + queries
        embeddings = self._embedding_model.encode(all_text)
        old_embeddings = embeddings[:len(previously_retrieved)]
        new_queries: list[str] = []
        accepted_embeddings: list[np.ndarray] = []

        for query, embedding in zip(queries, embeddings[len(previously_retrieved):]):
            prior_similarities = [
                self._cosine_similarity(embedding, old)
                for old in [*old_embeddings, *accepted_embeddings]
            ]
            if prior_similarities and max(prior_similarities) >= self.INCREMENTAL_SIMILARITY_THRESHOLD:
                continue
            new_queries.append(query)
            accepted_embeddings.append(embedding)

        return new_queries

    def _merge_excess_queries(self, queries: list[str]) -> list[str]:
        """If more than MAX_SUB_QUERIES, iteratively merge the two most similar."""
        while len(queries) > self.MAX_SUB_QUERIES:
            embeddings = self._embedding_model.encode(queries)
            max_sim = -1.0
            merge_i, merge_j = 0, 1

            for i in range(len(queries)):
                for j in range(i + 1, len(queries)):
                    sim = self._cosine_similarity(embeddings[i], embeddings[j])
                    if sim > max_sim:
                        max_sim = sim
                        merge_i, merge_j = i, j

            # Merge the two most similar
            merged = f"{queries[merge_i]} and {queries[merge_j]}"
            # Remove in reverse order to preserve indices
            queries.pop(merge_j)
            queries.pop(merge_i)
            queries.append(merged)

        return queries
