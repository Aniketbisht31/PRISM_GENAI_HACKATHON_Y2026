"""
Unit and integration tests for Streaming Live RAG API (Phase 7).
Tests:
- /health endpoint
- /api/query single-shot execution
- /api/refine late-constraint refinement without pipeline restart
- /api/presentation reformatting with 0 new retrievals
- /api/controller/evaluate direct chunk evaluation
- /api/telemetry/report
- /ws/stream WebSocket incremental streaming
"""

import json
import threading
import time
from types import SimpleNamespace
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.api.main import app, _is_presentation_request, _retrieve_subqueries
from src.controller.controller import ControllerDecision
from src.retrieval.retriever import RetrievedChunk
from src.synthesis.synthesizer import SynthesisResult
from src.llm.adapter import LLMResponse


@pytest.fixture
def client():
    return TestClient(app)


def test_parallel_subquery_retrieval_preserves_events_and_deduplicates_chunks():
    active_threads = set()
    lock = threading.Lock()

    class FakeRetriever:
        def retrieve(self, query, session_id, top_k, timestamp_s, trigger_reason):
            with lock:
                active_threads.add(threading.get_ident())
            time.sleep(0.03)
            return [RetrievedChunk(
                doc_id="Doc_01", section="§1", title="Shared", text=query
            )]

    chunks, events = _retrieve_subqueries(
        FakeRetriever(), ["capacity", "catering", "refunds"], "s_parallel", 0.8, "multi_intent"
    )

    assert len(active_threads) > 1
    assert len(chunks) == 1
    assert [event["query"] for event in events] == ["capacity", "catering", "refunds"]
    assert all(event["trigger"] == "multi_intent" for event in events)
    assert all(event["timestamp_s"] == 0.8 for event in events)


def test_acknowledgement_is_not_mistaken_for_presentation_request():
    assert not _is_presentation_request("Thanks, that's helpful.")
    assert _is_presentation_request("Can you make that shorter?")


def test_request_trace_boundaries_keep_session_id_for_filtered_reports(client):
    from src.api.main import get_components

    logger = get_components()[4]
    logger.clear()
    response = client.post(
        "/api/controller/evaluate",
        params={
            "session_id": "trace-session",
            "cumulative_text": "ok",
            "controller_type": "rule",
        },
    )
    assert response.status_code == 200
    report = logger.generate_report("trace-session")
    assert report["total_traces"] == 1
    assert report["trace_coverage_pct"] == 100.0


def test_websocket_retrieves_only_new_intents_after_provisional_search(client):
    class FakeController:
        def evaluate_chunk(self, cumulative_text, timestamp_s, session_id):
            return ControllerDecision("RETRIEVE", "clear intent", 0.95, timestamp_s)

    class FakeRetriever:
        def __init__(self):
            self.queries = []
            self.triggers = []

        def retrieve(self, query, session_id, top_k, timestamp_s, trigger_reason):
            self.queries.append(query)
            self.triggers.append(trigger_reason)
            return [RetrievedChunk("Doc_02", "§1", "Venue B", query)]

    class FakeSynthesizer:
        def __init__(self):
            self.version = 0

        def get_session(self, session_id):
            if self.version == 0:
                return None
            return SimpleNamespace(latest_answer=object())

        def synthesize(self, session_id, sub_queries, retrieved_chunks, retrieval_events):
            self.version = 1
            return SynthesisResult(
                retrieval_events, sub_queries, "Initial answer", [], None, 1, session_id
            )

        def refine_with_late_constraint(
            self, session_id, new_sub_queries, new_retrieved_chunks, new_retrieval_events
        ):
            self.version = 2
            return SynthesisResult(
                new_retrieval_events, new_sub_queries, "Refined answer", [], None, 2, session_id
            )

    decomposer = MagicMock()
    decomposer.decompose.side_effect = [
        SimpleNamespace(sub_queries=["Venue B seating capacity"], reasoning="capacity"),
        SimpleNamespace(
            sub_queries=["Venue B seating capacity", "Venue B cancellation policy"],
            reasoning="capacity and policy",
        ),
    ]
    decomposer.only_new_queries.side_effect = [
        ["Venue B seating capacity"],
        ["Venue B cancellation policy"],
    ]
    retriever = FakeRetriever()
    synthesizer = FakeSynthesizer()

    with patch("src.api.main.get_components", return_value=(
        FakeController(), decomposer, retriever, synthesizer, MagicMock()
    )):
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()  # session_start

            ws.send_json({
                "chunk_text": "What is Venue B's capacity?",
                "timestamp_s": 0.8,
                "is_final": False,
            })
            ws.receive_json()  # controller_decision
            assert ws.receive_json()["type"] == "retrieval_started"
            first_decomposition = ws.receive_json()
            first_retrieval = ws.receive_json()
            first_ready = ws.receive_json()
            assert first_ready["type"] == "provisional_context_ready"

            ws.send_json({
                "chunk_text": "And its cancellation policy?",
                "timestamp_s": 1.6,
                "is_final": True,
            })
            ws.receive_json()  # controller_decision
            second_decomposition = ws.receive_json()
            second_retrieval = ws.receive_json()
            second_answer = ws.receive_json()

    assert first_decomposition["new_sub_queries"] == ["Venue B seating capacity"]
    assert first_retrieval["retrieval_events"][0]["trigger"] == "provisional"
    assert second_decomposition["new_sub_queries"] == ["Venue B cancellation policy"]
    assert second_retrieval["retrieval_events"][0]["trigger"] == "multi_intent"
    assert second_answer["answer_version"] == 1
    assert retriever.queries == [
        "What is Venue B's capacity?",
        "Venue B seating capacity",
        "Venue B cancellation policy",
    ]


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_controller_evaluate_endpoint(client):
    response = client.post(
        "/api/controller/evaluate",
        params={
            "session_id": "test_eval",
            "cumulative_text": "Thanks, that's helpful.",
            "timestamp_s": 0.0,
            "controller_type": "rule_based",
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "NO_RETRIEVAL"


def test_api_query_and_presentation_flow(client):
    with patch("src.llm.adapter.GroqAdapter.complete") as mock_groq:
        mock_groq.side_effect = [
            # Controller
            LLMResponse(
                content='{"decision": "RETRIEVE", "reason": "question clear", "confidence": 0.9}',
                model="llama-3.3-70b-versatile",
            ),
            # Decomposer
            LLMResponse(
                content='{"sub_queries": ["What is included in the standard catering package?"], "reasoning": "single intent"}',
                model="llama-3.3-70b-versatile",
            ),
            # Synthesizer
            LLMResponse(
                content=json.dumps({
                    "answer": "\"The standard catering package for workshop events includes morning tea/coffee, a mid-morning snack, a buffet lunch with 2 vegetarian and 1 non-vegetarian option, and afternoon tea/coffee with light snacks.\" [Doc_06 \u00a71].",
                    "citations": [
                    {"doc_id": "Doc_06", "section": "\u00a71", "claim": "The standard catering package for workshop events includes morning tea/coffee, a mid-morning snack, a buffet lunch with 2 vegetarian and 1 non-vegetarian option, and afternoon tea/coffee with light snacks."},
                    ],
                    "uncertainty": None,
                }),
                model="llama-3.3-70b-versatile",
            ),
            # Presentation (reformat)
            LLMResponse(
                content=json.dumps({
                    "answer": "\u2022 \"The standard catering package for workshop events includes morning tea/coffee, a mid-morning snack, a buffet lunch with 2 vegetarian and 1 non-vegetarian option, and afternoon tea/coffee with light snacks.\" [Doc_06 \u00a71]",
                    "citations": [
                    {"doc_id": "Doc_06", "section": "\u00a71", "claim": "The standard catering package for workshop events includes morning tea/coffee, a mid-morning snack, a buffet lunch with 2 vegetarian and 1 non-vegetarian option, and afternoon tea/coffee with light snacks."},
                    ],
                    "uncertainty": None,
                }),
                model="llama-3.3-70b-versatile",
            ),
        ]

        # 1. Initial query
        q_resp = client.post(
            "/api/query",
            json={"utterance": "What is included in the standard catering package?"}
        )
        assert q_resp.status_code == 200
        q_data = q_resp.json()
        assert q_data["answer_version"] == 1
        assert len(q_data["citations"]) > 0
        session_id = q_data["session_id"]

        # 2. Presentation turn
        p_resp = client.post(
            "/api/presentation",
            json={"session_id": session_id, "request": "Make that bullet points please"}
        )
        assert p_resp.status_code == 200
        p_data = p_resp.json()
        assert p_data["answer_version"] == 2
        assert len(p_data["retrieval_events"]) == 0
        assert "•" in p_data["answer"]


def test_api_late_constraint_refinement(client):
    with patch("src.llm.adapter.GroqAdapter.complete") as mock_groq:
        mock_groq.side_effect = [
            # Controller 1
            LLMResponse(
                content='{"decision": "RETRIEVE", "reason": "ok", "confidence": 0.9}',
                model="llama-3.3-70b-versatile",
            ),
            # Decomposer 1
            LLMResponse(
                content='{"sub_queries": ["What is Venue B base rental cost?"], "reasoning": "rent"}',
                model="llama-3.3-70b-versatile",
            ),
            # Synthesis 1
            LLMResponse(
                content=json.dumps({
                    "answer": "\"Base rental cost is INR 35,000 per day.\" [Doc_02 \u00a71].",
                    "citations": [
                    {"doc_id": "Doc_02", "section": "\u00a71", "claim": "Base rental cost is INR 35,000 per day."},
                    ],
                    "uncertainty": None,
                }),
                model="llama-3.3-70b-versatile",
            ),
            # Controller 2 (Refinement)
            LLMResponse(
                content='{"decision": "RETRIEVE", "reason": "monsoon constraint", "confidence": 0.9}',
                model="llama-3.3-70b-versatile",
            ),
            # Decomposer 2 (Refinement)
            LLMResponse(
                content='{"sub_queries": ["monsoon season weather considerations Pune"], "reasoning": "weather"}',
                model="llama-3.3-70b-versatile",
            ),
            # Synthesis 2 (Refinement)
            LLMResponse(
                content=json.dumps({
                    "answer": "\"Base rental cost is INR 35,000 per day.\" [Doc_02 \u00a71]. \"Pune experiences a moderate climate for most of the year, with monsoon season from June to September bringing heavy rainfall.\" [Doc_12 \u00a71].",
                    "citations": [
                        {"doc_id": "Doc_02", "section": "\u00a71", "claim": "Base rental cost is INR 35,000 per day."},
                        {"doc_id": "Doc_12", "section": "\u00a71", "claim": "Pune experiences a moderate climate for most of the year, with monsoon season from June to September bringing heavy rainfall."},
                    ],
                    "uncertainty": None,
                }),
                model="llama-3.3-70b-versatile",
            ),
        ]

        # 1. Initial query
        r1 = client.post(
            "/api/query",
            json={"utterance": "What's the base rental cost for a workshop at Venue B?"}
        )
        assert r1.status_code == 200
        data1 = r1.json()
        assert data1["answer_version"] == 1
        session_id = data1["session_id"]

        # 2. Refinement query
        r2 = client.post(
            "/api/refine",
            json={
                "session_id": session_id,
                "new_detail": "Actually we'd need it during monsoon season, does that change anything?"
            }
        )
        assert r2.status_code == 200
        data2 = r2.json()
        assert data2["answer_version"] == 2
        doc_ids = [c["doc_id"] for c in data2["citations"]]
        assert "Doc_02" in doc_ids
        assert "Doc_12" in doc_ids


def test_websocket_streaming(client):
    with patch("src.llm.adapter.GroqAdapter.complete") as mock_groq:
        mock_groq.side_effect = [
            # Chunk 1: WAIT
            LLMResponse(
                content='{"decision": "WAIT", "reason": "fragment", "confidence": 0.8}',
                model="llama-3.3-70b-versatile",
            ),
            # Chunk 2: RETRIEVE
            LLMResponse(
                content='{"decision": "RETRIEVE", "reason": "question complete", "confidence": 0.95}',
                model="llama-3.3-70b-versatile",
            ),
            # Decompose
            LLMResponse(
                content='{"sub_queries": ["What is the seating capacity of Venue B?"], "reasoning": "capacity"}',
                model="llama-3.3-70b-versatile",
            ),
            # Synthesize
            LLMResponse(
                content=json.dumps({
                    "answer": "\"Maximum capacity is 100 people theatre-style, 50 people workshop-style with round tables.\" [Doc_02 \u00a71].",
                    "citations": [
                    {"doc_id": "Doc_02", "section": "\u00a71", "claim": "Maximum capacity is 100 people theatre-style, 50 people workshop-style with round tables."},
                    ],
                    "uncertainty": None,
                }),
                model="llama-3.3-70b-versatile",
            ),
        ]

        with client.websocket_connect("/ws/stream") as ws:
            start_msg = ws.receive_json()
            assert start_msg["type"] == "session_start"
            session_id = start_msg["session_id"]
            assert session_id

            # Send chunk 1
            ws.send_json({"chunk_text": "What is the", "timestamp_s": 0.0})
            dec1 = ws.receive_json()
            assert dec1["type"] == "controller_decision"
            assert dec1["decision"] == "WAIT"

            # Send chunk 2
            ws.send_json({
                "chunk_text": "seating capacity of Venue B?",
                "timestamp_s": 0.8,
                "is_final": True,
            })
            dec2 = ws.receive_json()
            assert dec2["type"] == "controller_decision"
            assert dec2["decision"] == "RETRIEVE"

            started_msg = ws.receive_json()
            assert started_msg["type"] == "retrieval_started"
            assert started_msg["trigger"] == "provisional"
            decomp_msg = ws.receive_json()
            assert decomp_msg["type"] == "decomposition"

            ret_msg = ws.receive_json()
            assert ret_msg["type"] == "retrieval_complete"

            ans_msg = ws.receive_json()
            assert ans_msg["type"] == "answer"
            assert ans_msg["answer_version"] == 1
            assert "Maximum capacity" in ans_msg["answer"]


def test_canonical_incremental_three_intent_stream(client):
    class FakeController:
        def evaluate_chunk(self, cumulative_text, timestamp_s, session_id):
            decision = "WAIT" if timestamp_s == 0.0 else "RETRIEVE"
            return ControllerDecision(decision, "fixture decision", 0.95, timestamp_s)

    class FakeRetriever:
        def __init__(self):
            self.calls = []

        def retrieve(self, query, session_id, top_k, timestamp_s, trigger_reason):
            self.calls.append((query, timestamp_s, trigger_reason))
            return [RetrievedChunk("Doc_01", "\u00a71", "Venue A", query)]

    class FakeSynthesizer:
        def __init__(self):
            self.result = None

        def get_session(self, session_id):
            return None

        def synthesize(self, session_id, sub_queries, retrieved_chunks, retrieval_events):
            self.result = SynthesisResult(
                retrieval_events, sub_queries,
                "Workshop choices are partly unverified [Doc_01 \u00a71].",
                [{"doc_id": "Doc_01", "section": "\u00a71", "claim": "Workshop choices"}],
                "Some requested details need confirmation.", 1, session_id,
            )
            return self.result

    decomposer = MagicMock()
    decomposition = [
        ["Pune workshop capacity for 30 people"],
        ["Pune workshop capacity for 30 people", "cancellation terms", "catering options"],
        ["Pune workshop capacity for 30 people", "cancellation terms", "catering options"],
    ]
    decomposer.decompose.side_effect = [
        SimpleNamespace(sub_queries=queries, reasoning="intent split")
        for queries in decomposition
    ]
    decomposer.only_new_queries.side_effect = [
        ["Pune workshop capacity for 30 people"],
        ["cancellation terms", "catering options"],
        [],
    ]
    retriever = FakeRetriever()
    synthesizer = FakeSynthesizer()
    with patch("src.api.main.get_components", return_value=(
        FakeController(), decomposer, retriever, synthesizer, MagicMock()
    )):
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()  # session_start
            ws.send_json({"chunk_text": "I need to plan a customer workshop in...", "timestamp_s": 0.0})
            first = ws.receive_json()
            assert first["type"] == "controller_decision" and first["decision"] == "WAIT"

            ws.send_json({"chunk_text": "...Pune for 30 people, and I need...", "timestamp_s": 0.8})
            provisional = []
            while True:
                event = ws.receive_json()
                provisional.append(event)
                if event["type"] == "provisional_context_ready":
                    break

            ws.send_json({"chunk_text": "...the cancellation policy and the catering options.", "timestamp_s": 1.6})
            multi = []
            while True:
                event = ws.receive_json()
                multi.append(event)
                if event["type"] == "provisional_context_ready":
                    break

            ws.send_json({"chunk_text": "", "timestamp_s": 2.1, "is_final": True})
            final = []
            while True:
                event = ws.receive_json()
                final.append(event)
                if event["type"] == "answer":
                    answer = event
                    break

    assert next(e for e in provisional if e["type"] == "retrieval_started")["trigger"] == "provisional"
    assert any(r.get("trigger") == "multi_intent" for e in multi if e["type"] == "retrieval_complete" for r in e["retrieval_events"])
    assert len(answer["sub_queries"]) == 3
    assert {event["trigger"] for event in answer["retrieval_events"]} == {"provisional", "multi_intent"}
    assert answer["uncertainty"] == "Some requested details need confirmation."
    assert answer["citations"] and answer["answer_version"] == 1
    assert [call[1] for call in retriever.calls if call[2] == "provisional"] == [0.8]
    assert all(call[2] == "multi_intent" for call in retriever.calls if call[1] == 1.6)


def test_presentation_variants_include_formalization():
    for request in (
        "Please repeat your last answer in two bullets.",
        "shorten that", "translate that to Hindi", "make it formal",
    ):
        assert _is_presentation_request(request)
