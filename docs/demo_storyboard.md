# Streaming Live RAG demo storyboard (about 4 minutes)

1. **0:00-0:30 — Start and trace.** Start the API and show `/health`, the selected local corpus, and a new `X-RAG-Trace-ID`.
2. **0:30-1:20 — Early retrieval.** Stream the workshop request at 0.0s, 0.8s, and 1.6s. Show `WAIT`, provisional retrieval, three subqueries, and parallel retrieval events with timestamps and source mappings.
3. **1:20-2:00 — Unified answer.** End the utterance at 2.1s. Show the answer version, exact cited corpus excerpts, and an uncertainty note for missing facts.
4. **2:00-2:50 — Late detail.** Ask a corpus-supported question, then add a new constraint. Show a delta query, retained citations, and the transition from answer version 1 to version 2.
5. **2:50-3:30 — Suppression.** Ask to repeat, shorten, translate, and formalize the last answer. Show zero retrieval-call deltas and unchanged citation sources.
6. **3:30-4:00 — Observability and limits.** Show request trace coverage, token counts, cost-estimate status, retrieval latency, and an out-of-corpus question returning uncertainty.

The repository contains this storyboard; it is not a recorded video.
