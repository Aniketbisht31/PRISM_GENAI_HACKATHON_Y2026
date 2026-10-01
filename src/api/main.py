"""
Streaming RAG API — FastAPI + WebSocket endpoint.
Wires Controller → Decomposer → HybridRetriever → SessionSynthesizer.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
import time
import contextvars
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel

from src.controller.controller import RetrievalController, RuleBasedController, ControllerDecision
from src.decomposer.decomposer import MultiIntentDecomposer
from src.retrieval.retriever import HybridRetriever, RetrievedChunk
from src.synthesis.synthesizer import SessionSynthesizer, SynthesisResult
from src.telemetry.logger import get_logger, TelemetryLogger
from src.telemetry.context import trace_id_context


# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Streaming Live RAG",
    description="Samsung PRISM GenAI Hackathon — Theme 4: Streaming Live RAG",
    version="1.0.0",
)


@app.middleware("http")
async def request_trace_middleware(request, call_next):
    trace_id = str(uuid.uuid4())
    token = trace_id_context.set(trace_id)
    logger = get_logger()
    started_at = time.perf_counter()
    session_id = request.query_params.get("session_id", "")
    if not session_id and request.method in {"POST", "PUT", "PATCH"}:
        try:
            body = await request.body()
            payload = json.loads(body) if body else {}
            if isinstance(payload, dict):
                session_id = str(payload.get("session_id", ""))
        except (json.JSONDecodeError, UnicodeDecodeError):
            session_id = ""
    logger.log_request_boundary(session_id, "started", request.method, request.url.path)
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-RAG-Trace-ID"] = trace_id
        return response
    finally:
        logger.log_request_boundary(
            session_id, "completed", request.method, request.url.path,
            status_code=status_code,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        )
        trace_id_context.reset(token)

# ── Lazy-initialized singletons ──────────────────────────────────────────────

_controller: Optional[RetrievalController] = None
_rule_controller: Optional[RuleBasedController] = None
_decomposer: Optional[MultiIntentDecomposer] = None
_retriever: Optional[HybridRetriever] = None
_synthesizer: Optional[SessionSynthesizer] = None
_logger: Optional[TelemetryLogger] = None


def get_components():
    """Lazy-initialize all pipeline components."""
    global _controller, _rule_controller, _decomposer, _retriever, _synthesizer, _logger

    if _logger is None:
        _logger = get_logger()

    if _retriever is None:
        corpus_path = os.environ.get("CORPUS_PATH", "data/corpus/workshop_planning_corpus.json")
        chroma_dir = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma_index")
        _retriever = HybridRetriever(chroma_dir=chroma_dir, corpus_path=corpus_path)

    if _controller is None:
        _controller = RetrievalController()

    if _rule_controller is None:
        _rule_controller = RuleBasedController()

    if _decomposer is None:
        _decomposer = MultiIntentDecomposer(embedding_model=_retriever._embedding_model)

    if _synthesizer is None:
        _synthesizer = SessionSynthesizer()

    return _controller, _decomposer, _retriever, _synthesizer, _logger


def _is_presentation_request(text: str) -> bool:
    """Separate reformat requests from conversational acknowledgements."""
    patterns = (
        r"\brepeat\b", r"\bshorter\b", r"\bshorten\b", r"\bsimpler\b", r"\bformal\b", r"\btranslate\b",
        r"\bbullet\s*points?\b", r"\bsummarize\s+that\b", r"\bsimplify\b",
        r"\breformat\b", r"\brephrase\b", r"\bmake\s+(?:it|that)\s+shorter\b",
        r"\bsay\s+that\s+again\b", r"\bput\s+that\s+in\b.*\b(list|table|bullets?)\b",
    )
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _retrieve_subqueries(
    retriever: HybridRetriever,
    queries: list[str],
    session_id: str,
    timestamp_s: float,
    trigger: str,
) -> tuple[list[dict], list[dict]]:
    """Search independent intents concurrently and preserve query order in results."""
    if not queries:
        return [], []

    def run_retrieval(context, query):
        return context.run(
            retriever.retrieve,
            query=query,
            session_id=session_id,
            top_k=5,
            timestamp_s=timestamp_s,
            trigger_reason=trigger,
        )

    with ThreadPoolExecutor(max_workers=min(4, len(queries))) as executor:
        futures = [
            executor.submit(run_retrieval, contextvars.copy_context(), query)
            for query in queries
        ]
        results = [future.result() for future in futures]

    all_chunks: list[dict] = []
    retrieval_events: list[dict] = []
    seen_keys: set[str] = set()
    for query, chunks in zip(queries, results):
        for chunk in chunks:
            key = f"{chunk.doc_id}_{chunk.section}"
            if key not in seen_keys:
                seen_keys.add(key)
                all_chunks.append(chunk.to_dict())
        retrieval_events.append({
            "timestamp_s": timestamp_s,
            "query": query,
            "sub_query": query,
            "trigger": trigger,
            "num_chunks": len(chunks),
        })

    return all_chunks, retrieval_events


# ── Pydantic models ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    utterance: str
    session_id: Optional[str] = None


class ChunkInput(BaseModel):
    chunk_text: str
    timestamp_s: float
    is_final: bool = False


class LateConstraintRequest(BaseModel):
    session_id: str
    new_detail: str


class PresentationRequest(BaseModel):
    session_id: str
    request: str


class HealthResponse(BaseModel):
    status: str
    version: str


# ── REST endpoints ───────────────────────────────────────────────────────────

@app.get("/health")
def health_check() -> HealthResponse:
    return HealthResponse(status="ok", version="1.0.0")


@app.post("/api/query")
def handle_query(req: QueryRequest) -> dict:
    """
    Single-shot query endpoint — processes a complete utterance through the full pipeline.
    Useful for testing and evaluation.
    """
    controller, decomposer, retriever, synthesizer, logger = get_components()

    session_id = req.session_id or str(uuid.uuid4())

    # Step 1: Controller decision on full utterance
    decision = controller.evaluate_chunk(
        cumulative_text=req.utterance,
        timestamp_s=0.0,
        session_id=session_id,
    )

    if decision.decision == "NO_RETRIEVAL":
        # Check if we have a prior answer for presentation-only
        session = synthesizer.get_session(session_id)
        if session and session.latest_answer and _is_presentation_request(req.utterance):
            result = synthesizer.handle_presentation_only(session_id, req.utterance)
            return _synthesis_result_to_dict(result)
        if session and session.latest_answer:
            latest = session.latest_answer
            return {
                "session_id": session_id,
                "answer": latest.answer,
                "citations": latest.citations,
                "uncertainty": latest.uncertainty,
                "answer_version": session.current_version,
                "sub_queries": [],
                "retrieval_events": [],
                "status": "no_retrieval_needed",
                "controller_decision": asdict(decision),
            }
        else:
            return {
                "session_id": session_id,
                "answer": "No prior context available and no retrieval needed.",
                "citations": [],
                "uncertainty": None,
                "answer_version": 0,
                "sub_queries": [],
                "retrieval_events": [],
                "controller_decision": asdict(decision),
            }

    if decision.decision == "WAIT":
        return {
            "session_id": session_id,
            "answer": None,
            "status": "waiting_for_more_input",
            "controller_decision": asdict(decision),
        }

    # Step 2: Decompose
    decomp = decomposer.decompose(req.utterance, session_id)

    # Step 3: Search independent intents concurrently.
    all_chunks, retrieval_events = _retrieve_subqueries(
        retriever, decomp.sub_queries, session_id, 0.0, "complete_query"
    )

    # Step 4: Synthesize
    result = synthesizer.synthesize(
        session_id=session_id,
        sub_queries=decomp.sub_queries,
        retrieved_chunks=all_chunks,
        retrieval_events=retrieval_events,
    )

    return _synthesis_result_to_dict(result)


@app.post("/api/refine")
def handle_refine(req: LateConstraintRequest) -> dict:
    """
    Late-constraint refinement — processes a new detail WITHOUT restarting the pipeline.
    Re-runs controller/decomposer only on the new detail, retrieves only the delta.
    """
    controller, decomposer, retriever, synthesizer, logger = get_components()

    # Step 1: Controller on new detail
    decision = controller.evaluate_chunk(
        cumulative_text=req.new_detail,
        timestamp_s=0.0,
        session_id=req.session_id,
    )

    if decision.decision == "NO_RETRIEVAL":
        session = synthesizer.get_session(req.session_id)
        if session and session.latest_answer and _is_presentation_request(req.new_detail):
            result = synthesizer.handle_presentation_only(req.session_id, req.new_detail)
            return _synthesis_result_to_dict(result)
        return {
            "session_id": req.session_id,
            "answer": session.latest_answer.answer if session and session.latest_answer else None,
            "citations": session.latest_answer.citations if session and session.latest_answer else [],
            "uncertainty": session.latest_answer.uncertainty if session and session.latest_answer else None,
            "answer_version": session.current_version if session else 0,
            "sub_queries": [],
            "retrieval_events": [],
            "status": "no_retrieval_needed",
        }

    if decision.decision == "WAIT":
        return {
            "session_id": req.session_id,
            "answer": None,
            "status": "waiting_for_more_input",
            "controller_decision": asdict(decision),
        }

    # Step 2: Decompose only the new detail
    decomp = decomposer.decompose(req.new_detail, req.session_id)

    # Step 3: Search the new constraint's independent intents concurrently.
    new_chunks, retrieval_events = _retrieve_subqueries(
        retriever, decomp.sub_queries, req.session_id, 0.0, "late_constraint_delta"
    )

    # Step 4: Refine (not restart)
    result = synthesizer.refine_with_late_constraint(
        session_id=req.session_id,
        new_sub_queries=decomp.sub_queries,
        new_retrieved_chunks=new_chunks,
        new_retrieval_events=retrieval_events,
    )

    return _synthesis_result_to_dict(result)


@app.post("/api/presentation")
def handle_presentation(req: PresentationRequest) -> dict:
    """Handle presentation-only requests (reformat, shorten, translate) with ZERO new retrieval."""
    _, _, _, synthesizer, _ = get_components()
    result = synthesizer.handle_presentation_only(req.session_id, req.request)
    return _synthesis_result_to_dict(result)


@app.post("/api/controller/evaluate")
def evaluate_controller(session_id: str = "", cumulative_text: str = "", timestamp_s: float = 0.0, controller_type: str = "llm") -> dict:
    """Direct controller evaluation endpoint for testing."""
    controller, _, _, _, _ = get_components()

    if not session_id:
        session_id = str(uuid.uuid4())

    if controller_type == "rule_based":
        global _rule_controller
        if _rule_controller is None:
            _rule_controller = RuleBasedController()
        decision = _rule_controller.evaluate_chunk(cumulative_text, timestamp_s, session_id)
    else:
        decision = controller.evaluate_chunk(cumulative_text, timestamp_s, session_id)

    return asdict(decision)


@app.post("/api/session/create")
def create_session() -> dict:
    """Create a new session."""
    _, _, _, synthesizer, _ = get_components()
    session_id = synthesizer.create_session()
    return {"session_id": session_id}


@app.get("/api/telemetry/report")
def get_telemetry_report(session_id: Optional[str] = None) -> dict:
    """Get telemetry report for evaluation."""
    _, _, _, _, logger = get_components()
    return logger.generate_report(session_id)


@app.get("/api/telemetry/events")
def get_telemetry_events(session_id: Optional[str] = None) -> list:
    """Get raw telemetry events."""
    _, _, _, _, logger = get_components()
    return logger.get_events(session_id)


# ── WebSocket endpoint ───────────────────────────────────────────────────────

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming transcript chunks.
    Accepts {chunk_text, timestamp_s} messages simulating incremental transcript arrival.
    """
    await websocket.accept()
    trace_id = str(uuid.uuid4())
    trace_token = trace_id_context.set(trace_id)
    controller, decomposer, retriever, synthesizer, logger = get_components()

    session_id = str(uuid.uuid4())
    cumulative_text = ""
    retrieved_sub_queries: list[str] = []
    pending_sub_queries: list[str] = []
    pending_chunks: list[dict] = []
    pending_retrieval_events: list[dict] = []

    try:
        logger.log_request_boundary(session_id, "started", "WS", "/ws/stream")
        await websocket.send_json({
            "type": "session_start",
            "session_id": session_id,
        })

        while True:
            data = await websocket.receive_json()
            chunk_text = data.get("chunk_text", "")
            timestamp_s = data.get("timestamp_s", 0.0)
            is_final = bool(data.get("is_final", False))

            # Accumulate transcript
            if cumulative_text:
                cumulative_text += " " + chunk_text
            else:
                cumulative_text = chunk_text

            # Step 1: Controller decision
            decision = await asyncio.to_thread(
                controller.evaluate_chunk,
                cumulative_text,
                timestamp_s,
                session_id,
            )

            if is_final and pending_sub_queries and decision.decision in ("WAIT", "NO_RETRIEVAL"):
                decision = ControllerDecision(
                    "RETRIEVE",
                    "Final transcript chunk received; synthesizing accumulated retrieved context.",
                    max(decision.confidence, 0.9),
                    timestamp_s,
                )
                logger.log_controller_decision(
                    session_id=session_id,
                    decision=decision.decision,
                    reason=decision.reason,
                    confidence=decision.confidence,
                    transcript=cumulative_text,
                    timestamp_s=timestamp_s,
                    controller_type="stream_finalization",
                )

            await websocket.send_json({
                "type": "controller_decision",
                "decision": decision.decision,
                "reason": decision.reason,
                "confidence": decision.confidence,
                "timestamp_s": decision.timestamp_s,
                "cumulative_text": cumulative_text,
            })

            if decision.decision == "RETRIEVE":
                # Begin a broad speculative corpus search at the first stable
                # intent while the decomposer identifies parallel subqueries.
                provisional_chunks: list[RetrievedChunk] = []
                provisional_event: Optional[dict] = None
                if not retrieved_sub_queries:
                    await websocket.send_json({
                        "type": "retrieval_started",
                        "trigger": "provisional",
                        "timestamp_s": timestamp_s,
                        "query": cumulative_text,
                    })
                    decomp_task = asyncio.to_thread(
                        decomposer.decompose, cumulative_text, session_id
                    )
                    provisional_task = asyncio.to_thread(
                        retriever.retrieve,
                        cumulative_text,
                        session_id,
                        5,
                        timestamp_s,
                        "provisional",
                    )
                    decomp, provisional_chunks = await asyncio.gather(
                        decomp_task, provisional_task
                    )
                    provisional_event = {
                        "timestamp_s": timestamp_s,
                        "query": cumulative_text,
                        "sub_query": cumulative_text,
                        "trigger": "provisional",
                        "num_chunks": len(provisional_chunks),
                    }
                else:
                    decomp = await asyncio.to_thread(
                        decomposer.decompose, cumulative_text, session_id
                    )
                new_sub_queries = decomposer.only_new_queries(
                    decomp.sub_queries, retrieved_sub_queries
                )

                await websocket.send_json({
                    "type": "decomposition",
                    "sub_queries": decomp.sub_queries,
                    "new_sub_queries": new_sub_queries,
                    "reasoning": decomp.reasoning,
                })

                if not new_sub_queries:
                    await websocket.send_json({
                        "type": "retrieval_skipped",
                        "reason": "No new intent beyond queries already retrieved.",
                    })

                # Start with a provisional search, then search only newly
                # discovered intents as more transcript arrives.
                if new_sub_queries:
                    all_chunks, retrieval_events = await asyncio.to_thread(
                        _retrieve_subqueries,
                        retriever,
                        new_sub_queries,
                        session_id,
                        timestamp_s,
                        "multi_intent",
                    )
                else:
                    all_chunks, retrieval_events = [], []
                if provisional_chunks:
                    provisional_dicts = [chunk.to_dict() for chunk in provisional_chunks]
                    seen_keys = {f"{chunk['doc_id']}_{chunk['section']}" for chunk in all_chunks}
                    for chunk in provisional_dicts:
                        key = f"{chunk['doc_id']}_{chunk['section']}"
                        if key not in seen_keys:
                            seen_keys.add(key)
                            all_chunks.append(chunk)
                if provisional_event:
                    retrieval_events.insert(0, provisional_event)
                retrieved_sub_queries.extend(new_sub_queries)
                pending_sub_queries.extend(new_sub_queries)
                pending_retrieval_events.extend(retrieval_events)
                pending_keys = {
                    f"{chunk['doc_id']}_{chunk['section']}" for chunk in pending_chunks
                }
                for chunk in all_chunks:
                    key = f"{chunk['doc_id']}_{chunk['section']}"
                    if key not in pending_keys:
                        pending_keys.add(key)
                        pending_chunks.append(chunk)

                if retrieval_events:
                    await websocket.send_json({
                        "type": "retrieval_complete",
                        "num_chunks": len(all_chunks),
                        "retrieval_events": retrieval_events,
                        "is_final": is_final,
                    })

                if not is_final:
                    await websocket.send_json({
                        "type": "provisional_context_ready",
                        "num_chunks": len(pending_chunks),
                        "pending_sub_queries": pending_sub_queries,
                    })
                    continue

                # Step 4: Synthesize
                session = synthesizer.get_session(session_id)
                if session and session.latest_answer:
                    result = synthesizer.refine_with_late_constraint(
                        session_id=session_id,
                        new_sub_queries=pending_sub_queries,
                        new_retrieved_chunks=pending_chunks,
                        new_retrieval_events=pending_retrieval_events,
                    )
                else:
                    result = synthesizer.synthesize(
                        session_id=session_id,
                        sub_queries=pending_sub_queries,
                        retrieved_chunks=pending_chunks,
                        retrieval_events=pending_retrieval_events,
                    )

                await websocket.send_json({
                    "type": "answer",
                    **_synthesis_result_to_dict(result),
                })
                cumulative_text = ""
                retrieved_sub_queries.clear()
                pending_sub_queries.clear()
                pending_chunks.clear()
                pending_retrieval_events.clear()

            elif decision.decision == "NO_RETRIEVAL":
                session = synthesizer.get_session(session_id)
                if session and session.latest_answer and _is_presentation_request(cumulative_text):
                    if not is_final:
                        await websocket.send_json({
                            "type": "presentation_waiting_for_utterance_end"
                        })
                        continue
                    result = synthesizer.handle_presentation_only(session_id, cumulative_text)
                    await websocket.send_json({
                        "type": "answer",
                        **_synthesis_result_to_dict(result),
                    })
                    cumulative_text = ""
                    retrieved_sub_queries.clear()
                    pending_sub_queries.clear()
                    pending_chunks.clear()
                    pending_retrieval_events.clear()
                else:
                    await websocket.send_json({
                        "type": "no_action",
                        "reason": decision.reason,
                    })
                    if is_final:
                        cumulative_text = ""
                        retrieved_sub_queries.clear()
                        pending_sub_queries.clear()
                        pending_chunks.clear()
                        pending_retrieval_events.clear()

            elif decision.decision == "WAIT" and is_final:
                await websocket.send_json({
                    "type": "incomplete_utterance",
                    "reason": decision.reason,
                })
                cumulative_text = ""
                retrieved_sub_queries.clear()
                pending_sub_queries.clear()
                pending_chunks.clear()
                pending_retrieval_events.clear()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        logger.log_request_boundary(
            session_id, "completed", "WS", "/ws/stream",
            status_code=200,
        )
        trace_id_context.reset(trace_token)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _synthesis_result_to_dict(result: SynthesisResult) -> dict:
    return {
        "session_id": result.session_id,
        "answer": result.answer,
        "citations": result.citations,
        "uncertainty": result.uncertainty,
        "answer_version": result.answer_version,
        "sub_queries": result.sub_queries,
        "retrieval_events": result.retrieval_events,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
