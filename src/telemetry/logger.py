"""
Telemetry Logger — structured JSON logging for every pipeline stage.
Writes to stdout (JSON lines) AND a local JSONL file.
Provides report generation for evaluation gates.
"""

from __future__ import annotations

import json
import os
import sys
import time
import threading
import os
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from pathlib import Path
from src.telemetry.context import trace_id_context


@dataclass
class TelemetryEvent:
    """A single telemetry event."""
    event_type: str  # controller_decision, retrieval_call, decomposition, fusion_rerank, answer_version, llm_call
    session_id: str
    timestamp: str = ""
    data: dict = field(default_factory=dict)
    trace_id: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.trace_id:
            self.trace_id = trace_id_context.get()


class TelemetryLogger:
    """Thread-safe telemetry logger writing structured JSON events."""

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path or os.environ.get("TELEMETRY_LOG_PATH", "./data/telemetry.jsonl")
        self._lock = threading.Lock()
        self._events: list[TelemetryEvent] = []  # In-memory buffer for report generation
        # Ensure log directory exists
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: TelemetryEvent) -> None:
        """Write an event to stdout and the log file."""
        event_dict = asdict(event)
        json_line = json.dumps(event_dict, default=str)

        with self._lock:
            self._events.append(event)
            # Write to stdout
            print(json_line, file=sys.stdout, flush=True)
            # Append to JSONL file
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json_line + "\n")

    def log_controller_decision(
        self, session_id: str, decision: str, reason: str,
        confidence: float, transcript: str, timestamp_s: float,
        controller_type: str = "llm"
    ) -> None:
        self.emit(TelemetryEvent(
            event_type="controller_decision",
            session_id=session_id,
            data={
                "decision": decision,
                "reason": reason,
                "confidence": confidence,
                # Keep trace diagnostics useful without persisting raw user text.
                "transcript_char_count": len(transcript),
                "timestamp_s": timestamp_s,
                "controller_type": controller_type,
            }
        ))

    def log_decomposition(
        self, session_id: str, utterance: str,
        sub_queries: list[str], reasoning: str,
        dedup_removed: int = 0
    ) -> None:
        self.emit(TelemetryEvent(
            event_type="decomposition",
            session_id=session_id,
            data={
                "utterance": utterance,
                "sub_queries": sub_queries,
                "sub_query_count": len(sub_queries),
                "reasoning": reasoning,
                "duplicates_removed": dedup_removed,
            }
        ))

    def log_retrieval_call(
        self, session_id: str, query: str, timestamp_s: float,
        trigger_reason: str, retriever_type: str,
        num_results: int, top_doc_ids: list[str], duration_ms: Optional[float] = None,
    ) -> None:
        self.emit(TelemetryEvent(
            event_type="retrieval_call",
            session_id=session_id,
            data={
                "query": query,
                "timestamp_s": timestamp_s,
                "trigger_reason": trigger_reason,
                "retriever_type": retriever_type,
                "num_results": num_results,
                "top_doc_ids": top_doc_ids,
                "duration_ms": None if duration_ms is None else round(duration_ms, 2),
            }
        ))

    def log_fusion_rerank(
        self, session_id: str, query: str,
        dense_doc_ids: list[str], sparse_doc_ids: list[str],
        fused_doc_ids: list[str]
    ) -> None:
        self.emit(TelemetryEvent(
            event_type="fusion_rerank",
            session_id=session_id,
            data={
                "query": query,
                "dense_doc_ids": dense_doc_ids,
                "sparse_doc_ids": sparse_doc_ids,
                "fused_doc_ids": fused_doc_ids,
            }
        ))

    def log_answer_version(
        self, session_id: str, version: int,
        answer: str, citations: list[dict],
        uncertainty: Optional[str], is_refinement: bool = False
    ) -> None:
        self.emit(TelemetryEvent(
            event_type="answer_version",
            session_id=session_id,
            data={
                "version": version,
                "answer_length": len(answer),
                "citation_count": len(citations),
                "citation_sources": sorted({
                    f"{citation.get('doc_id', '')} {citation.get('section', '')}"
                    for citation in citations
                }),
                "previous_version": max(version - 1, 0),
                "has_uncertainty": uncertainty is not None,
                "uncertainty": uncertainty,
                "is_refinement": is_refinement,
            }
        ))

    def log_llm_call(
        self, session_id: str, purpose: str,
        model: str, usage: dict, latency_ms: float
    ) -> None:
        # Estimate cost for Groq (free tier, but track for reporting)
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        backend = os.environ.get("LLM_BACKEND", "groq").lower()
        if backend == "ollama":
            estimated_cost = 0.0
            cost_status = "local_inference"
        else:
            prompt_rate = os.environ.get("LLM_PROMPT_COST_PER_1M_TOKENS_USD")
            completion_rate = os.environ.get("LLM_COMPLETION_COST_PER_1M_TOKENS_USD")
            if prompt_rate is None and completion_rate is None and model == "openai/gpt-oss-20b":
                # Groq's published standard on-demand rates for GPT-OSS 20B.
                prompt_rate, completion_rate = "0.075", "0.30"
                cost_status = "published_groq_on_demand_rate"
            else:
                cost_status = "configured_rate_estimate"
            if prompt_rate is not None and completion_rate is not None:
                estimated_cost = (
                    prompt_tokens * float(prompt_rate)
                    + completion_tokens * float(completion_rate)
                ) / 1_000_000
            else:
                estimated_cost = None
                cost_status = "pricing_not_configured"

        self.emit(TelemetryEvent(
            event_type="llm_call",
            session_id=session_id,
            data={
                "purpose": purpose,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": usage.get("total_tokens", 0),
                "latency_ms": round(latency_ms, 2),
                "estimated_cost_usd": estimated_cost,
                "cost_estimate_status": cost_status,
            }
        ))

    def log_llm_failure(self, session_id: str, purpose: str, error: Exception) -> None:
        """Record failed inference without storing provider messages or credentials."""
        self.emit(TelemetryEvent(
            event_type="llm_failure",
            session_id=session_id,
            data={
                "purpose": purpose,
                "error_type": type(error).__name__,
                "cost_estimate_status": "unknown_due_provider_failure",
            },
        ))

    def log_presentation_only(
        self, session_id: str, request: str
    ) -> None:
        self.emit(TelemetryEvent(
            event_type="presentation_only",
            session_id=session_id,
            data={
                "request": request,
                "new_retrieval_count": 0,
            }
        ))

    def log_request_boundary(
        self, session_id: str, stage: str, method: str, path: str,
        status_code: Optional[int] = None, duration_ms: Optional[float] = None,
    ) -> None:
        self.emit(TelemetryEvent(
            event_type=f"request_{stage}",
            session_id=session_id,
            data={
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": duration_ms,
            },
        ))

    def get_events(self, session_id: Optional[str] = None) -> list[dict]:
        """Get all events, optionally filtered by session_id."""
        with self._lock:
            events = [asdict(e) for e in self._events]
        if session_id:
            events = [e for e in events if e["session_id"] == session_id]
        return events

    def get_events_from_file(self) -> list[dict]:
        """Read all events from the JSONL log file."""
        events = []
        if os.path.exists(self.log_path):
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        return events

    def clear(self) -> None:
        """Clear in-memory events and the log file."""
        with self._lock:
            self._events.clear()
        if os.path.exists(self.log_path):
            open(self.log_path, "w").close()

    def generate_report(self, session_id: Optional[str] = None) -> dict:
        """
        Generate a telemetry report for evaluation gates.
        Returns metrics for: early retrieval %, multi-intent accuracy,
        citation support %, hallucinated doc IDs, trace coverage.
        """
        events = self.get_events(session_id)

        # Count event types for trace coverage
        event_types = set()
        controller_decisions = []
        retrieval_calls = []
        decompositions = []
        answer_versions = []
        llm_calls = []

        for e in events:
            event_types.add(e["event_type"])
            if e["event_type"] == "controller_decision":
                controller_decisions.append(e["data"])
            elif e["event_type"] == "retrieval_call":
                retrieval_calls.append(e["data"])
            elif e["event_type"] == "decomposition":
                decompositions.append(e["data"])
            elif e["event_type"] == "answer_version":
                answer_versions.append(e["data"])
            elif e["event_type"] == "llm_call":
                llm_calls.append(e["data"])

        # Early retrieval: % of RETRIEVE decisions
        retrieve_decisions = [d for d in controller_decisions if d["decision"] == "RETRIEVE"]
        total_eligible = len([d for d in controller_decisions if d["decision"] in ("RETRIEVE", "WAIT")])
        early_retrieval_pct = (len(retrieve_decisions) / max(total_eligible, 1)) * 100

        # Multi-intent: compound decompositions
        multi_intent_decomps = [d for d in decompositions if d["sub_query_count"] > 1]

        # Citation support
        total_citations = sum(d.get("citation_count", 0) for d in answer_versions)
        answers_with_citations = len([d for d in answer_versions if d.get("citation_count", 0) > 0])
        total_answers = len(answer_versions)
        citation_support_pct = (answers_with_citations / max(total_answers, 1)) * 100

        # LLM token usage
        total_prompt_tokens = sum(d.get("prompt_tokens", 0) for d in llm_calls)
        total_completion_tokens = sum(d.get("completion_tokens", 0) for d in llm_calls)
        total_latency_ms = sum(d.get("latency_ms", 0) for d in llm_calls)

        # Per-request trace coverage; aggregate event-type presence is not enough.
        traces: dict[str, list[dict]] = {}
        for event in events:
            trace_id = event.get("trace_id")
            if trace_id:
                traces.setdefault(trace_id, []).append(event)
        complete_traces = 0
        for trace_events in traces.values():
            types = {event["event_type"] for event in trace_events}
            started = next((e for e in trace_events if e["event_type"] == "request_started"), None)
            completed = "request_completed" in types
            complete = bool(started and completed)
            path = (started or {}).get("data", {}).get("path", "")
            decisions = [
                e.get("data", {}).get("decision") for e in trace_events
                if e.get("event_type") == "controller_decision"
            ]
            if path in ("/api/query", "/api/refine", "/ws/stream"):
                complete = complete and bool(decisions)
                if "RETRIEVE" in decisions:
                    complete = complete and "decomposition" in types and "retrieval_call" in types
                    complete = complete and "answer_version" in types
            if path == "/api/controller/evaluate":
                complete = complete and "controller_decision" in types
            if path == "/api/presentation":
                complete = complete and "presentation_only" in types and "answer_version" in types
            for trace_event in trace_events:
                if trace_event.get("event_type") == "llm_failure":
                    complete = False
                if trace_event.get("event_type") == "llm_call":
                    cost = trace_event.get("data", {}).get("estimated_cost_usd")
                    if not isinstance(cost, (int, float)):
                        complete = False
                        break
            if complete:
                complete_traces += 1
        trace_coverage_pct = (complete_traces / max(len(traces), 1)) * 100

        return {
            "total_events": len(events),
            "event_type_coverage": list(event_types),
            "early_retrieval_pct": round(early_retrieval_pct, 1),
            "total_controller_decisions": len(controller_decisions),
            "retrieve_decisions": len(retrieve_decisions),
            "multi_intent_decompositions": len(multi_intent_decomps),
            "total_decompositions": len(decompositions),
            "citation_support_pct": round(citation_support_pct, 1),
            "total_citations": total_citations,
            "total_answers": total_answers,
            "total_llm_calls": len(llm_calls),
            "total_llm_failures": sum(e["event_type"] == "llm_failure" for e in events),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_latency_ms": round(total_latency_ms, 1),
            "trace_coverage_pct": round(trace_coverage_pct, 1),
            "total_traces": len(traces),
            "complete_traces": complete_traces,
            "total_retrieval_calls": len(retrieval_calls),
            "total_estimated_cost_usd": round(sum(
                d["estimated_cost_usd"] for d in llm_calls
                if isinstance(d.get("estimated_cost_usd"), (int, float))
            ), 8),
            "llm_calls_without_cost_estimate": sum(
                d.get("estimated_cost_usd") is None for d in llm_calls
            ) + sum(e["event_type"] == "llm_failure" for e in events),
        }


# ── Singleton factory ────────────────────────────────────────────────────────

_logger_instance: Optional[TelemetryLogger] = None


def get_logger() -> TelemetryLogger:
    """Return the singleton telemetry logger."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = TelemetryLogger()
    return _logger_instance


def reset_logger() -> None:
    """Reset the singleton — useful for tests."""
    global _logger_instance
    _logger_instance = None
