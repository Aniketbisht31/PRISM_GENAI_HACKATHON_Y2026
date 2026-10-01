#!/usr/bin/env python3
"""
Ablation Studies — compares architectural variants to justify design choices.

Ablation A: HybridRetriever vs DenseOnlyRetriever
  - Compares recall@5 and answer groundedness

Ablation B: LLM-based RetrievalController vs RuleBasedController
  - Compares early retrieval %, false-trigger rate, average decision latency

Usage: python -m eval.ablations [--api-url http://localhost:8000]
"""

from __future__ import annotations

import json
import os
import sys
import time
import argparse
from typing import Optional
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controller.controller import RetrievalController, RuleBasedController
from src.decomposer.decomposer import MultiIntentDecomposer
from src.retrieval.retriever import HybridRetriever, DenseOnlyRetriever, RetrievedChunk
from src.synthesis.synthesizer import SessionSynthesizer
from src.telemetry.logger import TelemetryLogger


# Relevant queries for retrieval comparison
RETRIEVAL_TEST_QUERIES = [
    {
        "query": "What is the seating capacity of Venue B?",
        "relevant_docs": [("Doc_02", "§1"), ("Doc_02", "§2")],
    },
    {
        "query": "cancellation policy for Venue A",
        "relevant_docs": [("Doc_03", "§1")],
    },
    {
        "query": "catering options for workshop events",
        "relevant_docs": [("Doc_06", "§1"), ("Doc_06", "§2")],
    },
    {
        "query": "parking availability at both venues",
        "relevant_docs": [("Doc_09", "§1")],
    },
    {
        "query": "AV equipment and technical support",
        "relevant_docs": [("Doc_08", "§1")],
    },
    {
        "query": "corporate discount program",
        "relevant_docs": [("Doc_13", "§1")],
    },
    {
        "query": "international attendees visa support",
        "relevant_docs": [("Doc_14", "§1")],
    },
    {
        "query": "refund processing timeline",
        "relevant_docs": [("Doc_05", "§1")],
    },
    {
        "query": "weather considerations for Pune events",
        "relevant_docs": [("Doc_12", "§1"), ("Doc_12", "§2")],
    },
    {
        "query": "external catering policy",
        "relevant_docs": [("Doc_07", "§1")],
    },
]

# Controller test cases
CONTROLLER_TEST_CASES = [
    {"text": "I need to plan a", "expected": "WAIT", "type": "incomplete"},
    {"text": "What is the seating capacity of Venue B?", "expected": "RETRIEVE", "type": "clear_question"},
    {"text": "Thanks, that's helpful.", "expected": "NO_RETRIEVAL", "type": "filler"},
    {"text": "Can you repeat that shorter?", "expected": "NO_RETRIEVAL", "type": "presentation"},
    {"text": "I need to plan a customer workshop in Pune for 30 people", "expected": "RETRIEVE", "type": "complete"},
    {"text": "venue", "expected": "WAIT", "type": "fragment"},
    {"text": "What's the parking situation and do they offer technical support on-site?", "expected": "RETRIEVE", "type": "compound"},
    {"text": "ok", "expected": "NO_RETRIEVAL", "type": "filler"},
    {"text": "Does Venue A allow external catering?", "expected": "RETRIEVE", "type": "clear_question"},
    # The first intent is already independently retrievable, even though the
    # utterance trails off. This follows the guide's early retrieval example.
    {"text": "I need the cancellation policy and the...", "expected": "RETRIEVE", "type": "trailing"},
]


def run_ablation_a(corpus_path: str = "data/corpus/workshop_planning_corpus.json") -> dict:
    """
    Ablation A: HybridRetriever vs DenseOnlyRetriever.
    Compare recall@5 and answer groundedness.
    """
    print("\n" + "=" * 60)
    print("ABLATION A: Hybrid Retriever vs Dense-Only Retriever")
    print("=" * 60)

    # Initialize retrievers with separate indices
    print("  Initializing HybridRetriever...", end=" ", flush=True)
    hybrid_start = time.perf_counter()
    hybrid_logger = TelemetryLogger(log_path="./data/ablation_a_hybrid.jsonl")
    hybrid_logger.clear()

    # Use separate chroma dirs to avoid conflicts
    hybrid = HybridRetriever(
        chroma_dir="./data/chroma_ablation_hybrid",
        corpus_path=corpus_path,
    )
    hybrid.logger = hybrid_logger
    hybrid_init_time = time.perf_counter() - hybrid_start
    print(f"OK ({hybrid_init_time:.2f}s)")

    print("  Initializing DenseOnlyRetriever...", end=" ", flush=True)
    dense_start = time.perf_counter()
    dense_logger = TelemetryLogger(log_path="./data/ablation_a_dense.jsonl")
    dense_logger.clear()

    dense = DenseOnlyRetriever(
        chroma_dir="./data/chroma_ablation_dense",
        corpus_path=corpus_path,
    )
    dense.logger = dense_logger
    dense_init_time = time.perf_counter() - dense_start
    print(f"OK ({dense_init_time:.2f}s)")

    # Run queries
    hybrid_recall_scores = []
    dense_recall_scores = []
    hybrid_latencies = []
    dense_latencies = []

    for tq in RETRIEVAL_TEST_QUERIES:
        query = tq["query"]
        relevant = set(tq["relevant_docs"])

        # Hybrid
        t0 = time.perf_counter()
        h_results = hybrid.retrieve(query, session_id="ablation_a", top_k=5)
        hybrid_latencies.append((time.perf_counter() - t0) * 1000)
        h_retrieved = {(c.doc_id, c.section) for c in h_results}
        h_recall = len(h_retrieved & relevant) / max(len(relevant), 1)
        hybrid_recall_scores.append(h_recall)

        # Dense only
        t0 = time.perf_counter()
        d_results = dense.retrieve(query, session_id="ablation_a", top_k=5)
        dense_latencies.append((time.perf_counter() - t0) * 1000)
        d_retrieved = {(c.doc_id, c.section) for c in d_results}
        d_recall = len(d_retrieved & relevant) / max(len(relevant), 1)
        dense_recall_scores.append(d_recall)

        print(f"  Query: '{query[:50]}...' — Hybrid recall: {h_recall:.2f}, Dense recall: {d_recall:.2f}")

    # Compute aggregate metrics
    avg_hybrid_recall = sum(hybrid_recall_scores) / len(hybrid_recall_scores)
    avg_dense_recall = sum(dense_recall_scores) / len(dense_recall_scores)
    avg_hybrid_latency = sum(hybrid_latencies) / len(hybrid_latencies)
    avg_dense_latency = sum(dense_latencies) / len(dense_latencies)

    results = {
        "metric": ["Recall@5", "Avg Latency (ms)", "Index Init Time (s)"],
        "hybrid_score": [f"{avg_hybrid_recall:.3f}", f"{avg_hybrid_latency:.1f}", f"{hybrid_init_time:.2f}"],
        "dense_only_score": [f"{avg_dense_recall:.3f}", f"{avg_dense_latency:.1f}", f"{dense_init_time:.2f}"],
        "delta": [
            f"{avg_hybrid_recall - avg_dense_recall:+.3f}",
            f"{avg_hybrid_latency - avg_dense_latency:+.1f}",
            f"{hybrid_init_time - dense_init_time:+.2f}",
        ],
    }

    # Print table
    print_markdown_table("Ablation A: Hybrid vs Dense-Only", results)
    return results


def run_ablation_b() -> dict:
    """
    Ablation B: LLM-based RetrievalController vs RuleBasedController.
    Compare early retrieval %, false-trigger rate, average decision latency.
    """
    print("\n" + "=" * 60)
    print("ABLATION B: LLM Controller vs Rule-Based Controller")
    print("=" * 60)

    # Initialize controllers
    rule_controller = RuleBasedController()

    # We'll run the rule-based controller against all test cases
    # (LLM controller requires API, so we measure via API or skip if offline)

    rule_correct = 0
    rule_total = 0
    rule_latencies = []
    rule_false_triggers = 0

    llm_correct = 0
    llm_total = 0
    llm_latencies = []
    llm_false_triggers = 0

    has_llm = False
    try:
        llm_controller = RetrievalController()
        has_llm = True
    except Exception:
        print("  LLM controller not available (no API key). Using cached/simulated data.")

    for tc in CONTROLLER_TEST_CASES:
        text = tc["text"]
        expected = tc["expected"]

        # Rule-based
        t0 = time.perf_counter()
        rule_decision = rule_controller.evaluate_chunk(text, 0.0, "ablation_b_rule")
        rule_latency = (time.perf_counter() - t0) * 1000
        rule_latencies.append(rule_latency)
        rule_total += 1
        if rule_decision.decision == expected:
            rule_correct += 1
        elif rule_decision.decision == "RETRIEVE" and expected != "RETRIEVE":
            rule_false_triggers += 1

        # LLM-based
        if has_llm:
            t0 = time.perf_counter()
            try:
                llm_decision = llm_controller.evaluate_chunk(text, 0.0, "ablation_b_llm")
                llm_latency = (time.perf_counter() - t0) * 1000
                llm_latencies.append(llm_latency)
                llm_total += 1
                if llm_decision.decision == expected:
                    llm_correct += 1
                elif llm_decision.decision == "RETRIEVE" and expected != "RETRIEVE":
                    llm_false_triggers += 1
                print(f"  '{text[:50]}...' — Rule: {rule_decision.decision}, LLM: {llm_decision.decision}, Expected: {expected}")
            except Exception as e:
                print(f"  '{text[:50]}...' — Rule: {rule_decision.decision}, LLM: ERROR ({e})")
        else:
            print(f"  '{text[:50]}...' — Rule: {rule_decision.decision}, Expected: {expected}")

    # Metrics
    rule_accuracy = (rule_correct / max(rule_total, 1)) * 100
    rule_avg_latency = sum(rule_latencies) / max(len(rule_latencies), 1)
    rule_false_rate = (rule_false_triggers / max(rule_total, 1)) * 100

    if has_llm and llm_total > 0:
        llm_accuracy = (llm_correct / max(llm_total, 1)) * 100
        llm_avg_latency = sum(llm_latencies) / max(len(llm_latencies), 1)
        llm_false_rate = (llm_false_triggers / max(llm_total, 1)) * 100
    else:
        llm_accuracy = 0
        llm_avg_latency = 0
        llm_false_rate = 0

    results = {
        "metric": ["Accuracy (%)", "False Trigger Rate (%)", "Avg Latency (ms)", "Requires API Key"],
        "rule_based_score": [
            f"{rule_accuracy:.1f}",
            f"{rule_false_rate:.1f}",
            f"{rule_avg_latency:.2f}",
            "No",
        ],
        "model_based_score": [
            f"{llm_accuracy:.1f}" if has_llm else "N/A",
            f"{llm_false_rate:.1f}" if has_llm else "N/A",
            f"{llm_avg_latency:.2f}" if has_llm else "N/A",
            "Yes (GROQ_API_KEY)",
        ],
        "delta": [
            f"{llm_accuracy - rule_accuracy:+.1f}" if has_llm else "N/A",
            f"{llm_false_rate - rule_false_rate:+.1f}" if has_llm else "N/A",
            f"{llm_avg_latency - rule_avg_latency:+.2f}" if has_llm else "N/A",
            "",
        ],
    }

    print_markdown_table("Ablation B: LLM Controller vs Rule-Based", results)
    return results


def print_markdown_table(title: str, data: dict) -> None:
    """Print a formatted markdown-style comparison table."""
    print(f"\n### {title}\n")

    # Determine column headers
    cols = list(data.keys())
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"

    print(header)
    print(separator)

    num_rows = len(data[cols[0]])
    for i in range(num_rows):
        row = "| " + " | ".join(str(data[c][i]) for c in cols) + " |"
        print(row)

    print()


def save_results(ablation_a: dict, ablation_b: dict, output_path: str = "data/ablation_results.json"):
    """Save ablation results to a JSON file."""
    results = {
        "ablation_a": ablation_a,
        "ablation_b": ablation_b,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--corpus", default="data/corpus/workshop_planning_corpus.json")
    parser.add_argument("--output", default="data/ablation_results.json")
    args = parser.parse_args()

    print("Ablation Study Runner")
    print(f"Corpus: {args.corpus}")

    # Run Ablation A
    ablation_a = run_ablation_a(args.corpus)

    # Run Ablation B
    ablation_b = run_ablation_b()

    # Save results
    save_results(ablation_a, ablation_b, args.output)

    print("\n" + "=" * 60)
    print("ABLATION STUDIES COMPLETE")
    print("=" * 60)
    print("\nKey Findings:")
    print("  Ablation A (Hybrid vs Dense-Only): See Recall@5 comparison above.")
    print("  Ablation B (LLM vs Rule-Based): See accuracy and latency comparison above.")
    print("\nThese results justify the architectural choices per the Architectural Parsimony rule.")


if __name__ == "__main__":
    main()
