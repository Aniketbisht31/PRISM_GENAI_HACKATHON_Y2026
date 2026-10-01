"""Request-scoped telemetry context propagated through async worker threads."""

from contextvars import ContextVar

trace_id_context: ContextVar[str] = ContextVar("rag_trace_id", default="")
