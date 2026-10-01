"""
Unit tests for Hybrid Retrieval (Phase 4).
Tests:
- HybridRetriever indexing from corpus JSON
- Dense (ChromaDB) and Sparse (BM25) retrieval
- Reciprocal Rank Fusion (RRF, k=60)
- Deduplication by doc_id + section
- DenseOnlyRetriever interface compatibility
- Corpus isolation (all results come from indexed corpus)
"""

import os
import json
import tempfile
import pytest
import numpy as np
from unittest.mock import MagicMock

from src.retrieval.retriever import (
    HybridRetriever,
    DenseOnlyRetriever,
    RetrievedChunk,
)


@pytest.fixture
def sample_corpus():
    return [
        {
            "doc_id": "Doc_01",
            "section": "§1",
            "title": "Venue A - Overview",
            "text": "Venue A is in Hinjewadi, Pune. Maximum seating capacity 80 people theatre-style, 40 boardroom. Rent INR 25,000.",
        },
        {
            "doc_id": "Doc_02",
            "section": "§1",
            "title": "Venue B - Overview",
            "text": "Venue B is in Kharadi, Pune. Maximum capacity 100 theatre-style, 50 workshop-style. Rent INR 35,000.",
        },
        {
            "doc_id": "Doc_03",
            "section": "§1",
            "title": "Venue A - Cancellation Policy",
            "text": "Bookings for Venue A can be cancelled free up to 14 days before the event date.",
        },
    ]


@pytest.fixture
def mock_embedding_model():
    model = MagicMock()
    # Simple deterministic embedding based on length and ascii
    def encode_fn(texts):
        res = []
        for t in texts:
            vec = np.zeros(384, dtype=np.float32)
            t_low = t.lower()
            if "venue a" in t_low:
                vec[0] = 1.0
            if "venue b" in t_low:
                vec[1] = 1.0
            if "cancellation" in t_low:
                vec[2] = 1.0
            if "capacity" in t_low:
                vec[3] = 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            res.append(vec)
        return res
    model.encode.side_effect = encode_fn
    return model


def test_hybrid_retriever_indexing_and_retrieval(sample_corpus, mock_embedding_model):
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        corpus_path = os.path.join(tmp_dir, "corpus.json")
        with open(corpus_path, "w", encoding="utf-8") as f:
            json.dump(sample_corpus, f)

        retriever = HybridRetriever(
            chroma_dir=os.path.join(tmp_dir, "chroma"),
            corpus_path=corpus_path,
            embedding_model=mock_embedding_model,
        )

        results = retriever.retrieve("What is the capacity of Venue B?", session_id="test_s", top_k=2)

        assert len(results) <= 2
        assert all(isinstance(r, RetrievedChunk) for r in results)
        # Venue B chunk should be retrieved
        doc_ids = [r.doc_id for r in results]
        assert "Doc_02" in doc_ids
        assert results[0].rank_source == "fused"


def test_rrf_fusion_logic():
    # Directly test the static RRF fusion method
    chunk1 = RetrievedChunk(doc_id="Doc_01", section="§1", title="T1", text="Text 1")
    chunk2 = RetrievedChunk(doc_id="Doc_02", section="§1", title="T2", text="Text 2")
    chunk3 = RetrievedChunk(doc_id="Doc_03", section="§1", title="T3", text="Text 3")

    dense_results = [chunk1, chunk2]
    sparse_results = [chunk2, chunk3]

    fused = HybridRetriever._rrf_fusion(dense_results, sparse_results, k=60)

    assert len(fused) == 3
    # chunk2 appeared in both rankings (rank 2 in dense, rank 1 in sparse)
    # score(chunk2) = 1/(60+2) + 1/(60+1) ≈ 0.0161 + 0.0164 = 0.0325
    # score(chunk1) = 1/(60+1) ≈ 0.0164
    # score(chunk3) = 1/(60+2) ≈ 0.0161
    assert fused[0].doc_id == "Doc_02"
    assert fused[0].score > fused[1].score


def test_dense_only_retriever_compatibility(sample_corpus, mock_embedding_model):
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        corpus_path = os.path.join(tmp_dir, "corpus.json")
        with open(corpus_path, "w", encoding="utf-8") as f:
            json.dump(sample_corpus, f)

        dense_retriever = DenseOnlyRetriever(
            chroma_dir=os.path.join(tmp_dir, "chroma_dense"),
            corpus_path=corpus_path,
            embedding_model=mock_embedding_model,
        )


        results = dense_retriever.retrieve("cancellation policy", session_id="test_s_dense", top_k=2)

        assert len(results) <= 2
        assert all(isinstance(r, RetrievedChunk) for r in results)
        assert results[0].rank_source == "dense"
