# Architecture brief

## Request paths

`src/api/main.py` exposes synchronous REST handlers and an asynchronous WebSocket stream. Components are initialized lazily and reused by the process. The streamed endpoint assigns a fresh session ID; REST callers can create and reuse a session explicitly.

The query path evaluates retrieval readiness, decomposes an utterance into at most four distinct subqueries, retrieves candidate corpus chunks, removes duplicate document sections, then synthesizes one answer with citations and uncertainty. REST and streaming subquery retrieval run concurrently with stable result order. Retrieval combines ChromaDB cosine search and BM25 through reciprocal rank fusion. Embeddings use `all-MiniLM-L6-v2` locally. LLM calls use the configured Groq or Ollama adapter. Each answer sentence must include an inline citation whose claim exactly matches text in the cited source chunk; explicitly named venue references must also match the cited evidence. Unsupported generations fall back to query-relevant exact source sentences when available, otherwise the response reports uncertainty.

The WebSocket protocol includes an `is_final` flag. While it is false, a clear intent starts a provisional corpus search and decomposition concurrently; chunks and queries are retained for that utterance. Later chunks are decomposed against prior searches and only novel intents trigger additional retrieval. Synthesis waits until `is_final:true`, then returns one answer from the accumulated retrieved evidence. Transcript and pending-query state reset after finalization while session answer history remains.

Late constraints retrieve against the new detail and pass those new chunks plus the previous answer and citations to the synthesizer. Presentation requests call a separate reformat prompt and do not call the retriever. Session answer history is process-local and ephemeral.

## Corpus boundary and provenance

The bundled workshop corpus is `data/corpus/workshop_planning_corpus.json`. Retrieval has no external web search path. The synthesizer filters citations against retrieved document and normalized section IDs and removes uncited or unsupported answer sentences; refinement checks against prior and newly retrieved chunks. Telemetry records request trace boundaries, decisions, decompositions, retrieval calls and latency, estimated LLM usage cost, and answer versions in JSONL and process memory. Remote-model cost rates are configurable; an unknown rate is recorded as unknown and makes trace cost coverage incomplete. Local Ollama inference is recorded at zero provider cost.

## Deployment layout

The Docker image installs pinned dependencies, downloads the embedding model, copies application code, corpus and prebuilt Chroma index, and exposes port 8000. Compose stores telemetry in `/var/log/rag` so its persistent volume does not mask the corpus/index under `/app/data`.

## Known scope limits

- The API's singleton components and session state are process-local; multi-worker deployments do not share session state.
- Exact source-quote checks are a conservative provenance proxy: they may reject valid paraphrases and do not establish semantic entailment beyond the quoted text.
- Session expiration/eviction and API authentication are deployment responsibilities.
- The example corpus and evaluation set are workshop-planning focused.
- A clean Docker build and replay is required before the deployment gate can be marked passed.
