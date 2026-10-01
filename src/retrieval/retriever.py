"""
Hybrid Retrieval — dense (ChromaDB) + sparse (BM25) with Reciprocal Rank Fusion.
Two variants: HybridRetriever and DenseOnlyRetriever (for ablation).
All embeddings are local via sentence-transformers — no API calls.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.telemetry.logger import get_logger, TelemetryLogger
from src.provenance import normalize_section


@dataclass
class RetrievedChunk:
    """A single retrieved document chunk with provenance metadata."""
    doc_id: str
    section: str
    title: str
    text: str
    score: float = 0.0
    rank_source: str = ""  # 'dense', 'sparse', 'fused'

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "section": self.section,
            "title": self.title,
            "text": self.text,
            "score": self.score,
            "rank_source": self.rank_source,
        }


class HybridRetriever:
    """
    Retrieves from BOTH a dense ChromaDB index AND a sparse BM25 index,
    fusing results via Reciprocal Rank Fusion (RRF, k=60).
    """

    def __init__(
        self,
        chroma_dir: Optional[str] = None,
        corpus_path: Optional[str] = None,
        embedding_model: Optional[SentenceTransformer] = None,
    ):
        self.logger: TelemetryLogger = get_logger()
        self.chroma_dir = chroma_dir or os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma_index")
        os.makedirs(self.chroma_dir, exist_ok=True)
        self._corpus_cache_path = os.path.join(self.chroma_dir, "corpus_cache.json")

        self._embedding_model = embedding_model or SentenceTransformer("all-MiniLM-L6-v2")
        self._chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
        self._collection = self._chroma_client.get_or_create_collection(
            name="workshop_corpus",
            metadata={"hnsw:space": "cosine"},
        )

        # BM25 index (in-memory)
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_data: list[dict] = []

        if corpus_path:
            self.index_corpus(corpus_path)
        else:
            self._load_bm25_from_cache()

    def index_corpus(self, corpus_path: str) -> None:
        """Index the corpus into both ChromaDB and BM25."""
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)

        for doc in corpus:
            doc["section"] = normalize_section(doc.get("section"))

        self._corpus_data = corpus

        # --- ChromaDB dense index ---
        ids = []
        documents = []
        metadatas = []

        for i, doc in enumerate(corpus):
            chunk_id = f"{doc['doc_id']}_{doc['section']}_{i}"
            ids.append(chunk_id)
            documents.append(doc["text"])
            metadatas.append({
                "doc_id": doc["doc_id"],
                "section": doc["section"],
                "title": doc["title"],
            })

        # Compute embeddings locally
        raw_embeddings = self._embedding_model.encode(documents)
        if hasattr(raw_embeddings, "tolist"):
            embeddings = raw_embeddings.tolist()
        else:
            embeddings = [e.tolist() if hasattr(e, "tolist") else list(e) for e in raw_embeddings]

        # Upsert (handles re-indexing gracefully)
        self._collection.upsert(

            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        # --- BM25 sparse index ---
        tokenized_corpus = [doc["text"].lower().split() for doc in corpus]
        self._bm25 = BM25Okapi(tokenized_corpus)

        # Cache corpus for BM25 reconstruction
        with open(self._corpus_cache_path, "w", encoding="utf-8") as f:
            json.dump(corpus, f)

    def _load_bm25_from_cache(self) -> None:
        """Rebuild BM25 index from cached corpus data."""
        if os.path.exists(self._corpus_cache_path):
            with open(self._corpus_cache_path, "r", encoding="utf-8") as f:
                self._corpus_data = json.load(f)
            for doc in self._corpus_data:
                doc["section"] = normalize_section(doc.get("section"))
            tokenized_corpus = [doc["text"].lower().split() for doc in self._corpus_data]
            self._bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(
        self,
        query: str,
        session_id: str,
        top_k: int = 5,
        timestamp_s: float = 0.0,
        trigger_reason: str = "hybrid_rrf",
    ) -> list[RetrievedChunk]:
        """
        Retrieve top-k chunks using hybrid RRF fusion of dense + sparse results.
        """
        retrieval_started = time.perf_counter()
        # 1. Dense retrieval (top-10)
        raw_q = self._embedding_model.encode([query])
        if hasattr(raw_q, "tolist"):
            query_embedding = raw_q.tolist()
        else:
            query_embedding = [e.tolist() if hasattr(e, "tolist") else list(e) for e in raw_q]
        dense_raw = self._collection.query(

            query_embeddings=query_embedding,
            n_results=min(10, self._collection.count() or 10),
        )

        dense_results: list[RetrievedChunk] = []
        if dense_raw["ids"] and dense_raw["ids"][0]:
            for i in range(len(dense_raw["ids"][0])):
                meta = dense_raw["metadatas"][0][i]
                dense_results.append(RetrievedChunk(
                    doc_id=meta["doc_id"],
                    section=normalize_section(meta["section"]),
                    title=meta["title"],
                    text=dense_raw["documents"][0][i],
                    rank_source="dense",
                ))

        # 2. Sparse retrieval (top-10)
        sparse_results: list[RetrievedChunk] = []
        if self._bm25 is not None and self._corpus_data:
            tokenized_query = query.lower().split()
            bm25_scores = self._bm25.get_scores(tokenized_query)
            top_indices = np.argsort(bm25_scores)[::-1][:10]

            for idx in top_indices:
                if bm25_scores[idx] <= 0:
                    continue
                doc = self._corpus_data[idx]
                sparse_results.append(RetrievedChunk(
                    doc_id=doc["doc_id"],
                    section=doc["section"],
                    title=doc["title"],
                    text=doc["text"],
                    rank_source="sparse",
                ))

        # 3. RRF fusion
        fused = self._rrf_fusion(dense_results, sparse_results, k=60)

        # 4. Take top-k
        results = fused[:top_k]

        # Log fusion details
        self.logger.log_fusion_rerank(
            session_id=session_id,
            query=query,
            dense_doc_ids=[f"{c.doc_id} {c.section}" for c in dense_results[:5]],
            sparse_doc_ids=[f"{c.doc_id} {c.section}" for c in sparse_results[:5]],
            fused_doc_ids=[f"{c.doc_id} {c.section}" for c in results],
        )

        # Log retrieval call
        self.logger.log_retrieval_call(
            session_id=session_id,
            query=query,
            timestamp_s=timestamp_s,
            trigger_reason=trigger_reason,
            retriever_type="hybrid",
            num_results=len(results),
            top_doc_ids=[f"{c.doc_id} {c.section}" for c in results],
            duration_ms=(time.perf_counter() - retrieval_started) * 1000,
        )

        return results

    @staticmethod
    def _rrf_fusion(
        dense_results: list[RetrievedChunk],
        sparse_results: list[RetrievedChunk],
        k: int = 60,
    ) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion: score = Σ 1/(k + rank) across all rankings."""
        scores: dict[str, float] = {}
        chunks_map: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(dense_results, start=1):
            key = f"{chunk.doc_id}_{chunk.section}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in chunks_map:
                chunks_map[key] = chunk

        for rank, chunk in enumerate(sparse_results, start=1):
            key = f"{chunk.doc_id}_{chunk.section}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in chunks_map:
                chunks_map[key] = chunk

        # Sort by fused score descending
        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        fused: list[RetrievedChunk] = []
        for key in sorted_keys:
            chunk = chunks_map[key]
            chunk.score = scores[key]
            chunk.rank_source = "fused"
            fused.append(chunk)

        return fused


class DenseOnlyRetriever:
    """
    Dense-only retriever using ChromaDB — no BM25, no fusion.
    Used for ablation study comparing against HybridRetriever.
    """

    def __init__(
        self,
        chroma_dir: Optional[str] = None,
        corpus_path: Optional[str] = None,
        embedding_model: Optional[SentenceTransformer] = None,
    ):
        self.logger: TelemetryLogger = get_logger()
        self.chroma_dir = chroma_dir or os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma_index_dense")
        os.makedirs(self.chroma_dir, exist_ok=True)

        self._embedding_model = embedding_model or SentenceTransformer("all-MiniLM-L6-v2")
        self._chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
        self._collection = self._chroma_client.get_or_create_collection(
            name="workshop_corpus_dense",
            metadata={"hnsw:space": "cosine"},
        )

        if corpus_path:
            self.index_corpus(corpus_path)

    def index_corpus(self, corpus_path: str) -> None:
        """Index corpus into ChromaDB only."""
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)

        for doc in corpus:
            doc["section"] = normalize_section(doc.get("section"))

        ids, documents, metadatas = [], [], []
        for i, doc in enumerate(corpus):
            chunk_id = f"{doc['doc_id']}_{doc['section']}_{i}"
            ids.append(chunk_id)
            documents.append(doc["text"])
            metadatas.append({
                "doc_id": doc["doc_id"],
                "section": doc["section"],
                "title": doc["title"],
            })

        raw_embeddings = self._embedding_model.encode(documents)
        if hasattr(raw_embeddings, "tolist"):
            embeddings = raw_embeddings.tolist()
        else:
            embeddings = [e.tolist() if hasattr(e, "tolist") else list(e) for e in raw_embeddings]
        self._collection.upsert(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )

    def retrieve(
        self, query: str, session_id: str, top_k: int = 5, timestamp_s: float = 0.0
    ) -> list[RetrievedChunk]:
        """Retrieve top-k chunks using dense retrieval only."""
        retrieval_started = time.perf_counter()
        raw_q = self._embedding_model.encode([query])
        if hasattr(raw_q, "tolist"):
            query_embedding = raw_q.tolist()
        else:
            query_embedding = [e.tolist() if hasattr(e, "tolist") else list(e) for e in raw_q]
        raw = self._collection.query(

            query_embeddings=query_embedding,
            n_results=min(top_k * 2, self._collection.count() or top_k * 2),
        )

        results: list[RetrievedChunk] = []
        seen: set[str] = set()
        if raw["ids"] and raw["ids"][0]:
            for i in range(len(raw["ids"][0])):
                meta = raw["metadatas"][0][i]
                key = f"{meta['doc_id']}_{meta['section']}"
                if key not in seen:
                    seen.add(key)
                    results.append(RetrievedChunk(
                        doc_id=meta["doc_id"],
                        section=normalize_section(meta["section"]),
                        title=meta["title"],
                        text=raw["documents"][0][i],
                        rank_source="dense",
                    ))
                if len(results) >= top_k:
                    break

        self.logger.log_retrieval_call(
            session_id=session_id,
            query=query,
            timestamp_s=timestamp_s,
            trigger_reason="dense_only",
            retriever_type="dense_only",
            num_results=len(results),
            top_doc_ids=[f"{c.doc_id} {c.section}" for c in results],
            duration_ms=(time.perf_counter() - retrieval_started) * 1000,
        )

        return results
