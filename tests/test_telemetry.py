"""
Unit tests for Telemetry Module (Phase 6).
Tests event emission, log file appending, report generation, and trace coverage.
"""

import os
import json
import tempfile
import pytest

from src.telemetry.logger import TelemetryLogger, TelemetryEvent
from src.telemetry.context import trace_id_context


@pytest.fixture
def temp_logger():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        log_path = f.name
    logger = TelemetryLogger(log_path=log_path)
    logger.clear()
    yield logger
    if os.path.exists(log_path):
        os.remove(log_path)


def test_emit_and_retrieve_events(temp_logger):
    temp_logger.log_controller_decision(
        session_id="s1",
        decision="RETRIEVE",
        reason="Clear question",
        confidence=0.9,
        transcript="What is the seating capacity of Venue B?",
        timestamp_s=0.5,
        controller_type="llm"
    )
    
    events = temp_logger.get_events("s1")
    assert len(events) == 1
    assert events[0]["event_type"] == "controller_decision"
    assert events[0]["data"]["decision"] == "RETRIEVE"
    assert events[0]["data"]["confidence"] == 0.9


def test_telemetry_file_persistence(temp_logger):
    temp_logger.log_retrieval_call(
        session_id="s2",
        query="venue b capacity",
        timestamp_s=1.0,
        trigger_reason="hybrid_rrf",
        retriever_type="hybrid",
        num_results=5,
        top_doc_ids=["Doc_02 §1", "Doc_02 §2"]
    )
    
    events_from_file = temp_logger.get_events_from_file()
    assert len(events_from_file) == 1
    assert events_from_file[0]["event_type"] == "retrieval_call"
    assert events_from_file[0]["data"]["query"] == "venue b capacity"


def test_telemetry_report_generation(temp_logger, monkeypatch):
    # Log complete trace covering all expected event types
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    session_id = "s_full"
    trace_token = trace_id_context.set("trace-test-query")
    temp_logger.log_request_boundary(session_id, "started", "POST", "/api/query")
    temp_logger.log_controller_decision(
        session_id=session_id, decision="RETRIEVE", reason="intent ready",
        confidence=0.95, transcript="capacity of venue a", timestamp_s=0.2
    )
    temp_logger.log_decomposition(
        session_id=session_id, utterance="capacity of venue a",
        sub_queries=["capacity of venue a"], reasoning="single intent"
    )
    temp_logger.log_retrieval_call(
        session_id=session_id, query="capacity of venue a", timestamp_s=0.4,
        trigger_reason="hybrid", retriever_type="hybrid", num_results=3,
        top_doc_ids=["Doc_01 §1"]
    )
    temp_logger.log_llm_call(
        session_id=session_id, purpose="synthesis", model="test-model",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        latency_ms=250.0
    )
    temp_logger.log_answer_version(
        session_id=session_id, version=1, answer="Venue A capacity is 80 [Doc_01 §1].",
        citations=[{"doc_id": "Doc_01", "section": "§1", "claim": "capacity is 80"}],
        uncertainty=None
    )
    temp_logger.log_request_boundary(session_id, "completed", "POST", "/api/query", status_code=200, duration_ms=12.0)
    trace_id_context.reset(trace_token)
    
    report = temp_logger.generate_report(session_id)
    assert report["total_events"] == 7
    assert report["trace_coverage_pct"] == 100.0
    assert report["early_retrieval_pct"] == 100.0
    assert report["citation_support_pct"] == 100.0
    assert report["total_citations"] == 1
    assert report["total_llm_calls"] == 1
    assert report["total_prompt_tokens"] == 100


def test_provider_failure_is_incomplete_trace_with_unknown_cost(temp_logger):
    trace_token = trace_id_context.set("trace-provider-failure")
    temp_logger.log_request_boundary("s_failure", "started", "POST", "/api/query")
    temp_logger.log_llm_failure("s_failure", "synthesis", RuntimeError("private provider detail"))
    temp_logger.log_request_boundary("s_failure", "completed", "POST", "/api/query", status_code=200)
    trace_id_context.reset(trace_token)

    report = temp_logger.generate_report("s_failure")
    assert report["total_llm_failures"] == 1
    assert report["llm_calls_without_cost_estimate"] == 1
    assert report["trace_coverage_pct"] == 0.0
    assert "private provider detail" not in json.dumps(temp_logger.get_events())
