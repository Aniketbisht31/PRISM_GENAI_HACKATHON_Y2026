"""
Unit tests for Multi-Intent Decomposer (Phase 3).
Tests:
- Single intent decomposition (1 sub-query)
- Compound query decomposition (2-3 sub-queries)
- Max 4 sub-queries cap (merging least-distinct)
- Deduplication of near-identical queries using cosine similarity (>0.9)
- Defensive parsing (markdown code fences, invalid JSON fallback)
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from src.decomposer.decomposer import (
    MultiIntentDecomposer,
    DecompositionResult,
)
from src.llm.adapter import LLMResponse


@pytest.fixture
def mock_embedding_model():
    mock_model = MagicMock()
    # Return mock vectors where query 0 and query 1 can be similar or distinct
    def encode_side_effect(queries):
        embeddings = []
        for q in queries:
            if "duplicate" in q.lower():
                embeddings.append(np.array([1.0, 0.0, 0.0], dtype=np.float32))
            elif "unique_a" in q.lower():
                embeddings.append(np.array([0.0, 1.0, 0.0], dtype=np.float32))
            elif "unique_b" in q.lower():
                embeddings.append(np.array([0.0, 0.0, 1.0], dtype=np.float32))
            else:
                # Hash string into a deterministic vector
                h = abs(hash(q)) % 1000
                embeddings.append(np.array([float(h), 1.0, 0.5], dtype=np.float32))
        return embeddings
    mock_model.encode.side_effect = encode_side_effect
    return mock_model


def test_single_intent_decomposition():
    with patch("src.decomposer.decomposer.get_adapter") as mock_get_adapter:
        mock_adapter = MagicMock()
        mock_adapter.load_prompt_template.return_value = "{utterance}"
        mock_adapter.complete.return_value = LLMResponse(
            content='{"sub_queries": ["What is the seating capacity of Venue B?"], "reasoning": "Single factual lookup"}',
            model="llama-3.3-70b-versatile",
            usage={"prompt_tokens": 40, "completion_tokens": 15, "total_tokens": 55},
        )
        mock_get_adapter.return_value = mock_adapter

        decomposer = MultiIntentDecomposer(embedding_model=MagicMock())
        result = decomposer.decompose("What is the seating capacity of Venue B?", session_id="s1")

        assert isinstance(result, DecompositionResult)
        assert len(result.sub_queries) == 1
        assert result.sub_queries[0] == "What is the seating capacity of Venue B?"
        assert result.duplicates_removed == 0


def test_compound_query_decomposition():
    with patch("src.decomposer.decomposer.get_adapter") as mock_get_adapter:
        mock_adapter = MagicMock()
        mock_adapter.load_prompt_template.return_value = "{utterance}"
        mock_adapter.complete.return_value = LLMResponse(
            content='''```json
{
  "sub_queries": [
    "What is the venue capacity for a 30 person workshop in Pune?",
    "What is the cancellation policy?",
    "What catering options are available?"
  ],
  "reasoning": "Three distinct information needs"
}
```''',
            model="llama-3.3-70b-versatile",
        )
        mock_get_adapter.return_value = mock_adapter

        mock_emb = MagicMock()
        # Return orthogonal embeddings so no deduplication happens
        mock_emb.encode.return_value = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]

        decomposer = MultiIntentDecomposer(embedding_model=mock_emb)
        result = decomposer.decompose(
            "I need to plan a customer workshop in Pune for 30 people, and I need the cancellation policy and catering options.",
            session_id="s_compound"
        )

        assert len(result.sub_queries) == 3
        assert "capacity" in result.sub_queries[0].lower()
        assert "cancellation" in result.sub_queries[1].lower()
        assert "catering" in result.sub_queries[2].lower()


def test_deduplicate_near_identical_queries():
    with patch("src.decomposer.decomposer.get_adapter") as mock_get_adapter:
        mock_adapter = MagicMock()
        mock_adapter.load_prompt_template.return_value = "{utterance}"
        mock_adapter.complete.return_value = LLMResponse(
            content='{"sub_queries": ["Venue A cancellation policy", "Venue A cancellation terms", "Venue A catering"], "reasoning": "test"}',
            model="llama-3.3-70b-versatile",
        )
        mock_get_adapter.return_value = mock_adapter

        mock_emb = MagicMock()
        # First two vectors are identical (similarity = 1.0 > 0.9)
        mock_emb.encode.return_value = [
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ]

        decomposer = MultiIntentDecomposer(embedding_model=mock_emb)
        result = decomposer.decompose("Venue A details", session_id="s_dedup")

        assert len(result.sub_queries) == 2
        assert result.duplicates_removed == 1


def test_cap_max_four_sub_queries():
    with patch("src.decomposer.decomposer.get_adapter") as mock_get_adapter:
        mock_adapter = MagicMock()
        mock_adapter.load_prompt_template.return_value = "{utterance}"
        mock_adapter.complete.return_value = LLMResponse(
            content='{"sub_queries": ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"], "reasoning": "overfragmented"}',
            model="llama-3.3-70b-versatile",
        )
        mock_get_adapter.return_value = mock_adapter

        mock_emb = MagicMock()
        def mock_encode(queries):
            return [np.array([float(i), 1.0]) for i in range(len(queries))]
        mock_emb.encode.side_effect = mock_encode

        decomposer = MultiIntentDecomposer(embedding_model=mock_emb)
        result = decomposer.decompose("big compound utterance", session_id="s_cap")

        assert len(result.sub_queries) <= 4


def test_fallback_on_llm_json_error():
    with patch("src.decomposer.decomposer.get_adapter") as mock_get_adapter:
        mock_adapter = MagicMock()
        mock_adapter.load_prompt_template.return_value = "{utterance}"
        mock_adapter.complete.return_value = LLMResponse(
            content="Sorry, I cannot format this.",
            model="llama-3.3-70b-versatile",
        )
        mock_get_adapter.return_value = mock_adapter

        decomposer = MultiIntentDecomposer(embedding_model=MagicMock())
        result = decomposer.decompose("What is the capacity?", session_id="s_err")

        # Falls back to original utterance
        assert len(result.sub_queries) == 1
        assert result.sub_queries[0] == "What is the capacity?"


def test_only_new_queries_filters_already_retrieved_intents(mock_embedding_model):
    decomposer = object.__new__(MultiIntentDecomposer)
    decomposer._embedding_model = mock_embedding_model

    new_queries = decomposer.only_new_queries(
        ["unique_a capacity", "unique_b catering"],
        ["unique_a capacity"],
    )

    assert new_queries == ["unique_b catering"]
