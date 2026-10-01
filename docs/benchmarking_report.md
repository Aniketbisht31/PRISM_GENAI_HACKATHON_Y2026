# Benchmarking and evaluation

## Offline verification

Run `pytest -q` for the offline unit and integration suite. Provider calls are mocked, so these checks cover code paths and invariants without measuring live model quality or latency. The tests include source-quote provenance, per-sentence citation enforcement, request trace completeness, and gate metric behavior.

## Evaluation gates

Run `python -m eval.run_gates` against a running API configured with Groq or Ollama. The runner reads the 72 scenarios in `data/eval/test_utterances.json` and reports:

| Gate | Measurement |
| --- | --- |
| G1 | Startup and health check; a clean Docker build/replay must be run separately |
| G2 | Early retrieval recall and false-trigger rate, with minimum sample counts |
| G3 | Compound-query decomposition coverage, with minimum sample count |
| G4 | Per-sentence exact source support, valid source IDs, citation coverage, and uncertainty behavior |
| G5 | Refinement citation lineage/versioning and presentation requests with zero retrieval calls |
| G6 | Per-trace event completeness and known estimated LLM cost for every trace |

The runner fails a gate when the required minimum sample size is not met. G1 is deliberately reported as unverified by the live runner: health alone does not establish that a clean image builds and replays successfully. A live evaluation also depends on the configured provider and its current behavior. Preserve the full gate output, backend/model, corpus and fixture revisions, date, and environment configuration with benchmark results.

## Ablations

`python -m eval.ablations` compares hybrid and dense-only retrieval and rule-based and model-based control. The checked-in `data/ablation_results.json` records a previous run on 10 retrieval queries and 10 controller cases: Recall@5 was 0.950 for both retrievers; the controller accuracy was 90% for both, with recorded mean decision latency of 0.93 ms for rules and 604.30 ms for the model. Treat these as historical figures because the artifact does not preserve the run date, model/backend revision, or full output. Rerun the command and retain that metadata before presenting them as reproducible results. Model-based control uses a configured provider, so only run that portion when provider use is intended. These small fixtures do not establish broad production performance.

The project still needs a measured non-streaming baseline on the same held-out set. The checked-in ablations compare components, but do not measure the end-to-end benefit of streaming against a non-streaming pipeline.

`python -m eval.compare_streaming_baseline --limit 3` now runs paired REST and WebSocket measurements on the same simple-query fixtures, records REST latency, time to first streaming retrieval, final streaming latency, and citation support, and writes `data/streaming_baseline_results.json`. This run could not be completed in the current review because the configured Groq model exhausted its daily token quota during the live gate replay; no baseline score is asserted here.

## Edge-case findings

1. **Citation spillover:** an answer such as `Unsupported fact. Supported quoted fact [Doc_01 §1].` could let the trailing citation appear to cover both claims. The sentence validator now requires an inline citation on every factual sentence; regression coverage is in `tests/test_synthesis.py`.
2. **Corrupt section markers:** source sections containing U+FFFD could fail citation matching even when the document was correct. Corpus/cache markers are repaired and retrieval normalizes legacy metadata; regression coverage checks normalized section identity.
3. **Unknown inference pricing:** remote model telemetry without a configured rate previously risked appearing complete. Such traces now report unknown cost and fail G6 cost coverage; offline gate tests cover this condition in `tests/test_evaluation.py`.

These are code-level adversarial regressions, not a substitute for a fresh live provider evaluation.

The attempted live replays used Groq `openai/gpt-oss-20b`. One run hit the provider's daily 200,000-token limit (HTTP 429); a later run returned HTTP 400 `json_validate_failed` for some generated JSON-mode responses. Those exceptions initially surfaced as API 500s. Synthesis/refinement/presentation now fall back to verified prior or exact retrieved excerpts and log a privacy-safe `llm_failure` event; such traces remain incomplete for G6 because their cost is unknown. The replays did not produce a complete G1-G6 result table. The local Ollama smoke test exceeded its HTTP timeout before returning a full answer, so it was not used for the replay.

## Known measurement limits

- Exact quotation support is a conservative grounding proxy. It can reject faithful paraphrases and does not establish broader semantic entailment.
- Live quality, latency, and cost vary by provider, model, and configuration; offline tests do not measure them.
- The bundled corpus and evaluation fixtures are workshop-planning focused, so results do not generalize to other domains.
- G1 still requires a clean Docker build and replay to be marked passed.
