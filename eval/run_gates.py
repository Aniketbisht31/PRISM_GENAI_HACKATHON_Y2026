#!/usr/bin/env python3
"""
Evaluation Gate Runner — runs 72 test cases against the live API and
prints pass/fail results for all 6 automated evaluation gates.

Gates:
  G1: Reproducibility (docker compose up works headless)
  G2: Early Retrieval >=80% of eligible queries
  G3: Multi-Intent Identification >=70% of compound queries
  G4: Factual Grounding >=85% citation support, zero hallucinated doc IDs
  G5: Session Refinement (state continuity, no full re-search on late constraints)
  G6: Telemetry (100% trace coverage)

Usage: python -m eval.run_gates [--api-url http://localhost:8000]
"""

from __future__ import annotations

import json
import os
import sys
import time
import argparse
import re
from dataclasses import dataclass
from typing import Optional

import httpx
from src.provenance import evidence_supports_claim, normalize_section

with open("data/corpus/workshop_planning_corpus.json", encoding="utf-8") as _corpus_file:
    _corpus = json.load(_corpus_file)
VALID_SOURCES = {
    (doc["doc_id"], normalize_section(doc["section"])): f"{doc.get('title', '')} {doc['text']}"
    for doc in _corpus
}
VALID_DOC_IDS = {doc_id for doc_id, _ in VALID_SOURCES}
VALID_SECTIONS = {section for _, section in VALID_SOURCES}


@dataclass
class GateResult:
    gate: str
    description: str
    passed: bool
    score: str
    details: str


def load_test_cases(path: str = "data/eval/test_utterances.json") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def query_api(base_url: str, utterance: str, session_id: Optional[str] = None) -> dict:
    """Send a query to the API."""
    payload = {"utterance": utterance}
    if session_id:
        payload["session_id"] = session_id
    resp = httpx.post(f"{base_url}/api/query", json=payload, timeout=60.0)
    resp.raise_for_status()
    return resp.json()


def refine_api(base_url: str, session_id: str, new_detail: str) -> dict:
    """Send a late-constraint refinement."""
    resp = httpx.post(
        f"{base_url}/api/refine",
        json={"session_id": session_id, "new_detail": new_detail},
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()


def presentation_api(base_url: str, session_id: str, request: str) -> dict:
    """Send a presentation-only request."""
    resp = httpx.post(
        f"{base_url}/api/presentation",
        json={"session_id": session_id, "request": request},
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()


def controller_api(base_url: str, text: str, timestamp_s: float, session_id: str, controller_type: str = "llm") -> dict:
    """Directly evaluate the controller."""
    resp = httpx.post(
        f"{base_url}/api/controller/evaluate",
        params={
            "session_id": session_id,
            "cumulative_text": text,
            "timestamp_s": timestamp_s,
            "controller_type": controller_type,
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()


def get_telemetry_events(base_url: str, session_id: Optional[str] = None) -> list:
    params = {}
    if session_id:
        params["session_id"] = session_id
    resp = httpx.get(f"{base_url}/api/telemetry/events", params=params, timeout=30.0)
    resp.raise_for_status()
    return resp.json()


def get_telemetry_report(base_url: str, session_id: Optional[str] = None) -> dict:
    params = {}
    if session_id:
        params["session_id"] = session_id
    resp = httpx.get(f"{base_url}/api/telemetry/report", params=params, timeout=30.0)
    resp.raise_for_status()
    return resp.json()


def validate_citations(result: dict) -> tuple[int, int, list[str]]:
    """Return supported citation claims, unsupported claims, and failure details."""
    supported = 0
    unsupported = 0
    failures: list[str] = []
    for citation in result.get("citations", []):
        doc_id = str(citation.get("doc_id", ""))
        section = normalize_section(citation.get("section", ""))
        claim = str(citation.get("claim", "")).strip()
        source = VALID_SOURCES.get((doc_id, section))
        if source is not None and claim and evidence_supports_claim(claim, source):
            supported += 1
        else:
            unsupported += 1
            failures.append(f"{doc_id} {section}: missing or unsupported claim/source")

    # Do not let uncited sentences inherit a trailing citation.
    answer = str(result.get("answer", ""))
    pattern = re.compile(r"\[(Doc_[^\]\s]+)\s+([^\]]+)\]")
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer.strip()):
        if not sentence:
            continue
        references = pattern.findall(sentence)
        if not references:
            if sentence and not result.get("uncertainty"):
                unsupported += 1
                failures.append(f"uncited assertion: {sentence[:100]}")
            continue
        assertion = pattern.sub("", sentence).strip(" \t:;,.!?-?\"'")
        for doc_id, raw_section in references:
            source = VALID_SOURCES.get((doc_id, normalize_section(raw_section)))
            if source is None or not evidence_supports_claim(assertion, source):
                unsupported += 1
                failures.append(f"unsupported inline citation: {doc_id} {raw_section}")
    return supported, unsupported, failures


def run_all_tests(base_url: str, test_cases: list[dict]) -> dict:
    """Run all test cases and collect results for gate evaluation."""
    results = {
        "simple": [],
        "compound": [],
        "late_constraint": [],
        "presentation_only": [],
        "no_retrieval_needed": [],
        "wait_then_retrieve": [],
        "uncertainty_check": [],
    }

    for tc in test_cases:
        tc_id = tc["id"]
        category = tc["category"]
        print(f"  Running {tc_id} ({category})...", end=" ", flush=True)

        try:
            if category in ("simple", "compound", "uncertainty_check", "no_retrieval_needed"):
                result = query_api(base_url, tc["utterance"])
                result["_test_case"] = tc
                results[category].append(result)
                print("OK")

            elif category == "late_constraint":
                seq = tc["utterance_sequence"]
                # First utterance
                r1 = query_api(base_url, seq[0])
                session_id = r1.get("session_id", "")
                # Second utterance — refine
                r2 = refine_api(base_url, session_id, seq[1])
                r2["_test_case"] = tc
                r2["_first_result"] = r1
                results[category].append(r2)
                print("OK")

            elif category == "presentation_only":
                seq = tc["utterance_sequence"]
                r1 = query_api(base_url, seq[0])
                session_id = r1.get("session_id", "")
                before = get_telemetry_events(base_url, session_id)
                r2 = presentation_api(base_url, session_id, seq[1])
                after = get_telemetry_events(base_url, session_id)
                before_calls = sum(e.get("event_type") == "retrieval_call" for e in before)
                after_calls = sum(e.get("event_type") == "retrieval_call" for e in after)
                r2["retrieval_call_delta"] = after_calls - before_calls
                r2["_test_case"] = tc
                r2["_first_result"] = r1
                r2["_session_id"] = session_id
                results[category].append(r2)
                print("OK")

            elif category == "wait_then_retrieve":
                chunk_seq = tc["chunk_sequence"]
                session_id = f"test_{tc_id}"
                chunk_results = []
                cumulative = ""
                for chunk in chunk_seq:
                    if cumulative:
                        cumulative += " " + chunk["text"]
                    else:
                        cumulative = chunk["text"]
                    cr = controller_api(
                        base_url, cumulative, chunk["timestamp_s"], session_id
                    )
                    cr["_expected"] = chunk["expected_decision"]
                    cr["_cumulative"] = cumulative
                    chunk_results.append(cr)
                results[category].append({
                    "chunk_results": chunk_results,
                    "_test_case": tc,
                })
                print("OK")

        except Exception as e:
            print(f"FAILED ({e})")
            results[category].append({"_error": str(e), "_test_case": tc})

    return results


def evaluate_g2_early_retrieval(results: dict) -> GateResult:
    """Measure pre-final retrieval recall and premature-trigger rate separately."""
    eligible_retrievals = 0
    early_retrievals = 0
    eligible_waits = 0
    false_triggers = 0

    for test in results.get("wait_then_retrieve", []):
        if "_error" in test:
            continue
        for chunk in test.get("chunk_results", []):
            expected = chunk.get("_expected")
            actual = chunk.get("decision")
            if expected == "RETRIEVE":
                eligible_retrievals += 1
                if actual == "RETRIEVE":
                    early_retrievals += 1
            elif expected == "WAIT":
                eligible_waits += 1
                if actual == "RETRIEVE":
                    false_triggers += 1

    for category in ("no_retrieval_needed", "presentation_only"):
        for test in results.get(category, []):
            if "_error" in test:
                continue
            eligible_waits += 1
            if test.get("retrieval_call_delta", 0) != 0 or test.get("retrieval_events"):
                false_triggers += 1

    early_pct = 100.0 * early_retrievals / max(eligible_retrievals, 1)
    false_pct = 100.0 * false_triggers / max(eligible_waits, 1)
    passed = eligible_retrievals >= 20 and eligible_waits >= 20 and early_pct >= 80.0 and false_pct <= 10.0
    return GateResult(
        gate="G2",
        description="Early retrieval >=80%; false triggers <=10%",
        passed=passed,
        score=f"{early_pct:.1f}% early; {false_pct:.1f}% false triggers",
        details=(f"{early_retrievals}/{eligible_retrievals} eligible retrieval chunks triggered before final; "
                 f"{false_triggers}/{eligible_waits} no-retrieval chunks triggered retrieval"),
    )


def evaluate_g3_multi_intent(results: dict) -> GateResult:
    """G3: Multi-Intent Identification >=70% of compound queries."""
    compound_tests = results.get("compound", [])
    correct = 0
    total = 0

    for r in compound_tests:
        tc = r.get("_test_case", {})
        if "_error" in r:
            total += 1
            continue
        expected_count = tc.get("expected_sub_intent_count", 1)
        actual_queries = r.get("sub_queries", [])
        total += 1
        # Allow ±1 tolerance on sub-query count for compound queries
        if expected_count > 1 and len(actual_queries) > 1:
            correct += 1
        elif expected_count == 1 and len(actual_queries) == 1:
            correct += 1

    pct = (correct / max(total, 1)) * 100
    passed = total >= 20 and pct >= 70.0
    return GateResult(
        gate="G3",
        description="Multi-Intent >=70%",
        passed=passed,
        score=f"{pct:.1f}%",
        details=f"{correct}/{total} compound queries correctly decomposed",
    )


def evaluate_g4_grounding(results: dict) -> GateResult:
    """G4: supported assertions >=85%; no nonexistent document IDs."""
    assertion_count = 0
    supported_count = 0
    unsupported_count = 0
    uncertainty_correct = 0
    uncertainty_total = 0
    fabricated_ids: list[str] = []
    sampled_answers = 0

    for category in ("simple", "compound", "uncertainty_check", "late_constraint"):
        for result in results.get(category, []):
            sampled_answers += 1
            if "_error" in result:
                assertion_count += 1
                unsupported_count += 1
                continue
            test_case = result.get("_test_case", {})
            if test_case.get("expected_uncertainty", False):
                uncertainty_total += 1
                if result.get("uncertainty"):
                    uncertainty_correct += 1
            citations = result.get("citations", [])
            answer = str(result.get("answer", ""))
            sentence_count = len([s for s in re.split(r"(?<=[.!?])\s+|\n+", answer.strip()) if s])
            total_for_result = max(len(citations), sentence_count if not result.get("uncertainty") else 0)
            supported, unsupported, failures = validate_citations(result)
            assertion_count += total_for_result
            supported_count += max(0, total_for_result - unsupported)
            unsupported_count += unsupported
            fabricated_ids.extend(
                f"{c.get('doc_id', '')} {c.get('section', '')}"
                for c in citations
                if (c.get("doc_id", ""), normalize_section(c.get("section", ""))) not in VALID_SOURCES
            )

    support_pct = 100.0 * supported_count / max(assertion_count, 1)
    passed = sampled_answers >= 20 and support_pct >= 85.0 and not fabricated_ids and unsupported_count == 0
    detail = f"{supported_count}/{assertion_count} assertions supported across {sampled_answers} answers; {unsupported_count} unsupported; {len(fabricated_ids)} fabricated IDs"
    if uncertainty_total:
        detail += f"; {uncertainty_correct}/{uncertainty_total} uncertainty flags present"
    if fabricated_ids:
        detail += f"; invalid IDs: {fabricated_ids}"
    return GateResult(
        gate="G4",
        description="Grounding >=85%, 0 hallucinations",
        passed=passed,
        score=f"{support_pct:.1f}% supported, {len(fabricated_ids)} fabricated IDs",
        details=detail,
    )


def evaluate_g5_refinement(results: dict) -> GateResult:
    """Check answer-version lineage, retained citations, and delta-only events."""
    checks = 0
    passed_checks = 0
    for result in results.get("late_constraint", []):
        checks += 1
        if "_error" in result:
            continue
        first = result.get("_first_result", {})
        old = {
            (c.get("doc_id"), normalize_section(c.get("section", "")))
            for c in first.get("citations", [])
        }
        new = {
            (c.get("doc_id"), normalize_section(c.get("section", "")))
            for c in result.get("citations", [])
        }
        events = result.get("retrieval_events", [])
        delta_only = all(event.get("trigger") == "late_constraint_delta" for event in events)
        if result.get("answer_version", 0) >= 2 and old.issubset(new) and delta_only:
            passed_checks += 1

    for result in results.get("presentation_only", []):
        checks += 1
        if ("_error" not in result and not result.get("retrieval_events")
                and result.get("retrieval_call_delta", 0) == 0):
            passed_checks += 1

    pct = 100.0 * passed_checks / max(checks, 1)
    return GateResult(
        gate="G5",
        description="Session refinement and retrieval suppression",
        passed=checks >= 20 and pct >= 80.0,
        score=f"{pct:.1f}%",
        details=f"{passed_checks}/{checks} refinements retained citation lineage and used delta events; presentation turns had no retrieval",
    )


def evaluate_g6_telemetry(base_url: str) -> GateResult:
    """G6 requires complete request traces and known inference-cost estimates."""
    report = get_telemetry_report(base_url)
    coverage = report.get("trace_coverage_pct", 0)
    total_traces = report.get("total_traces", 0)
    complete_traces = report.get("complete_traces", 0)
    missing_cost = report.get("llm_calls_without_cost_estimate", 0)
    event_types = report.get("event_type_coverage", [])
    required = {"request_started", "request_completed", "controller_decision", "retrieval_call", "answer_version", "llm_call"}
    missing_types = sorted(required - set(event_types))
    passed = total_traces > 0 and coverage >= 100.0 and missing_cost == 0
    details = (
        f"Complete traces {complete_traces}/{total_traces} ({coverage}%), "
        f"LLM calls without cost estimates {missing_cost}, event types {sorted(event_types)}"
    )
    if missing_types:
        details += f"; event types not seen: {missing_types}"
    return GateResult(
        gate="G6",
        description="100% request trace and token cost coverage",
        passed=passed,
        score=f"{coverage:.0f}% traces; {missing_cost} unknown costs",
        details=details,
    )


def print_results_table(gates: list[GateResult]) -> None:
    """Print a formatted pass/fail table."""
    print("\n" + "=" * 80)
    print("EVALUATION GATE RESULTS")
    print("=" * 80)
    print(f"{'Gate':<6} {'Description':<35} {'Status':<8} {'Score':<25}")
    print("-" * 80)
    for g in gates:
        status = "PASS" if g.passed else "FAIL"
        print(f"{g.gate:<6} {g.description:<35} {status:<8} {g.score:<25}")
    print("-" * 80)

    passed = sum(1 for g in gates if g.passed)
    total = len(gates)
    print(f"\nOverall: {passed}/{total} gates passed")

    print("\nDetailed Results:")
    for g in gates:
        status = "PASS" if g.passed else "FAIL"
        print(f"\n  [{g.gate}] {g.description} - {status}")
        print(f"    {g.details}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run evaluation gates")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument("--test-file", default="data/eval/test_utterances.json", help="Path to test utterances")
    args = parser.parse_args()

    print(f"Evaluation Gate Runner")
    print(f"API URL: {args.api_url}")
    print(f"Test file: {args.test_file}")

    # Check API health
    print("\nChecking API health...", end=" ")
    try:
        resp = httpx.get(f"{args.api_url}/health", timeout=10.0)
        resp.raise_for_status()
        print(f"OK ({resp.json()})")
    except Exception as e:
        print(f"FAILED - {e}")
        print("Make sure the API is running (docker compose up or python -m src.api.main)")
        sys.exit(1)

    # Load test cases
    test_cases = load_test_cases(args.test_file)
    print(f"\nLoaded {len(test_cases)} test cases")

    # Run all tests
    print("\nRunning tests...")
    results = run_all_tests(args.api_url, test_cases)

    # Evaluate gates
    gates = []

    # G1 requires an isolated clean Docker build and replay. A live health
    # endpoint only proves that some API process is running.
    gates.append(GateResult(
        gate="G1",
        description="Reproducibility",
        passed=False,
        score="NOT VERIFIED",
        details="A clean Docker build and replay was not run; API health alone does not establish reproducibility.",
    ))

    # G2: Early Retrieval
    gates.append(evaluate_g2_early_retrieval(results))

    # G3: Multi-Intent
    gates.append(evaluate_g3_multi_intent(results))

    # G4: Factual Grounding
    gates.append(evaluate_g4_grounding(results))

    # G5: Session Refinement
    gates.append(evaluate_g5_refinement(results))

    # G6: Telemetry
    gates.append(evaluate_g6_telemetry(args.api_url))

    # Print results
    print_results_table(gates)

    # Exit with non-zero if any gate failed
    if not all(g.passed for g in gates):
        sys.exit(1)


if __name__ == "__main__":
    main()
