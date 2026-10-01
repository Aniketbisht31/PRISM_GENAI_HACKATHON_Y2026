# Telemetry schema

Events are JSON Lines records written by `TelemetryLogger`.

Every event has:

| Field | Meaning |
| --- | --- |
| `event_type` | Pipeline stage or request boundary. |
| `session_id` | Ephemeral answer session, when the stage has one. |
| `trace_id` | UUID linking events from one HTTP request or WebSocket connection. |
| `timestamp` | UTC ISO-8601 event time. |
| `data` | Stage-specific fields. |

The trace boundary events contain HTTP method, route, response status, and elapsed request time. Controller events contain the decision, reason, confidence, transcript timestamp, and transcript character count. Decomposition events contain subqueries and duplicate counts. Retrieval events contain the query, trigger, timestamp, result count, and source IDs/sections; fusion events retain dense, sparse, and fused rankings.

Answer-version events record `version`, `previous_version`, cited source IDs/sections, uncertainty state, and refinement status. LLM events record model, purpose, prompt/completion/total tokens, latency, estimated cost, and `cost_estimate_status`.

For Groq `openai/gpt-oss-20b`, cost uses the published standard on-demand rates of $0.075 per million input tokens and $0.30 per million output tokens. Other remote models need `LLM_PROMPT_COST_PER_1M_TOKENS_USD` and `LLM_COMPLETION_COST_PER_1M_TOKENS_USD` configured to their current rates. Local Ollama cost is recorded as $0.00 with `local_inference` status. Cost estimates do not include hardware or cached-token discounts. See [Groq GPT-OSS 20B pricing](https://console.groq.com/docs/model/openai/gpt-oss-20b).

The report computes trace completeness per `trace_id`; it does not treat aggregate event-type presence as proof of complete requests. A retrieval request requires controller, decomposition, retrieval, answer-version, and request-boundary events. LLM calls without a numeric cost estimate make the trace incomplete.

Session answers are process-local and disappear on restart. Telemetry is persistent at `TELEMETRY_LOG_PATH`; set an appropriate retention policy and protect the file because retrieval queries and source references are recorded there.
