import json
from pathlib import Path
from eval import run_gates


def test_g2_reports_early_recall_and_false_triggers_separately():
    results = {
        "wait_then_retrieve": [{"chunk_results": [
            {"_expected": "WAIT", "decision": "WAIT"},
            {"_expected": "RETRIEVE", "decision": "RETRIEVE"},
        ]} for _ in range(20)],
        "no_retrieval_needed": [{"retrieval_events": []} for _ in range(20)],
    }
    gate = run_gates.evaluate_g2_early_retrieval(results)
    assert gate.passed
    assert "100.0% early" in gate.score and "0.0% false" in gate.score

    for case in results["wait_then_retrieve"]:
        case["chunk_results"][0]["decision"] = "RETRIEVE"
    assert not run_gates.evaluate_g2_early_retrieval(results).passed


def test_g3_requires_twenty_compound_cases_and_two_queries():
    results = {"compound": [
        {"_test_case": {"expected_sub_intent_count": 2}, "sub_queries": ["a", "b"]}
        for _ in range(20)
    ]}
    assert run_gates.evaluate_g3_multi_intent(results).passed
    assert not run_gates.evaluate_g3_multi_intent({"compound": results["compound"][:5]}).passed


def test_g4_checks_exact_claims_and_inline_citations():
    key = next(iter(run_gates.VALID_SOURCES))
    doc_id, section = key
    source = run_gates.VALID_SOURCES[key]
    excerpt = source.split(". ")[0].rstrip(".") + "."
    case = {
        "answer": f'"{excerpt}" [{doc_id} {section}].',
        "citations": [{"doc_id": doc_id, "section": section, "claim": excerpt}],
    }
    supported, unsupported, failures = run_gates.validate_citations(case)
    assert supported == 1 and unsupported == 0 and not failures

    bad = {**case, "answer": f'"A fabricated claim." [{doc_id} {section}].',
           "citations": [{"doc_id": doc_id, "section": section, "claim": "A fabricated claim."}]}
    _, unsupported, _ = run_gates.validate_citations(bad)
    assert unsupported > 0

    results = {category: [] for category in ("simple", "compound", "uncertainty_check", "late_constraint")}
    results["simple"] = [
        {**case, "_test_case": {"expected_uncertainty": False}}
        for _ in range(20)
    ]
    assert run_gates.evaluate_g4_grounding(results).passed


def test_g5_requires_version_lineage_and_delta_only_retrieval():
    citations = [{"doc_id": "Doc_01", "section": "\u00a71"}]
    results = {
        "late_constraint": [
            {"answer_version": 2, "citations": citations, "_first_result": {"citations": citations},
             "retrieval_events": [{"trigger": "late_constraint_delta"}]}
            for _ in range(10)
        ],
        "presentation_only": [{"retrieval_events": []} for _ in range(10)],
    }
    assert run_gates.evaluate_g5_refinement(results).passed
    for case in results["late_constraint"][:5]:
        case["retrieval_events"] = [{"trigger": "complete_query"}]
    assert not run_gates.evaluate_g5_refinement(results).passed


def test_g6_fails_when_trace_or_cost_coverage_is_incomplete(monkeypatch):
    monkeypatch.setattr(run_gates, "get_telemetry_report", lambda _: {
        "trace_coverage_pct": 100.0, "total_traces": 20, "complete_traces": 20,
        "llm_calls_without_cost_estimate": 0,
        "event_type_coverage": ["request_started", "request_completed", "controller_decision"],
    })
    assert run_gates.evaluate_g6_telemetry("http://unused").passed
    monkeypatch.setattr(run_gates, "get_telemetry_report", lambda _: {
        "trace_coverage_pct": 95.0, "total_traces": 20, "complete_traces": 19,
        "llm_calls_without_cost_estimate": 1, "event_type_coverage": [],
    })
    assert not run_gates.evaluate_g6_telemetry("http://unused").passed
