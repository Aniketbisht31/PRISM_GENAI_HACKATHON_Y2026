"""Paired latency/grounding comparison for REST and incremental WebSocket paths."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
import websockets

from eval.run_gates import validate_citations


def load_cases(path: str, limit: int) -> list[dict]:
    with open(path, encoding="utf-8") as stream:
        cases = json.load(stream)
    selected = [case for case in cases if case.get("category") == "simple"][:limit]
    for case in selected:
        text = case["utterance"].strip()
        words = text.split()
        split_at = max(1, len(words) // 2)
        case["stream_chunks"] = [" ".join(words[:split_at]), " ".join(words[split_at:])]
    return selected


async def run_stream(base_url: str, chunks: list[str]) -> tuple[dict, float | None, float]:
    parsed = urlparse(base_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_url = f"{ws_scheme}://{parsed.netloc}/ws/stream"
    started = time.perf_counter()
    first_retrieval_ms = None
    async with websockets.connect(ws_url, open_timeout=15, close_timeout=5) as socket:
        json.loads(await asyncio.wait_for(socket.recv(), timeout=30))  # session_start
        answer = None
        for index, chunk in enumerate(chunks):
            chunk_started = time.perf_counter()
            await socket.send(json.dumps({
                "chunk_text": chunk,
                "timestamp_s": round(time.perf_counter() - started, 3),
                "is_final": index == len(chunks) - 1,
            }))
            while True:
                event = json.loads(await asyncio.wait_for(socket.recv(), timeout=120))
                if event.get("type") == "retrieval_started" and first_retrieval_ms is None:
                    first_retrieval_ms = (time.perf_counter() - started) * 1000
                if event.get("type") == "answer":
                    answer = event
                    break
                if event.get("type") in {"incomplete_utterance", "no_action"}:
                    break
            if answer is not None:
                break
            if index < len(chunks) - 1:
                # Preserve an explicit gap between transcript fragments.
                await asyncio.sleep(max(0, 0.05 - (time.perf_counter() - chunk_started)))
        if answer is None:
            raise RuntimeError("Stream ended without an answer event")
    return answer, first_retrieval_ms, (time.perf_counter() - started) * 1000


async def compare(base_url: str, cases: list[dict]) -> dict:
    paired = []
    with httpx.Client(timeout=180) as client:
        for case in cases:
            utterance = case["utterance"]
            started = time.perf_counter()
            response = client.post(f"{base_url}/api/query", json={"utterance": utterance})
            response.raise_for_status()
            rest_ms = (time.perf_counter() - started) * 1000
            rest_result = response.json()

            stream_result, first_retrieval_ms, stream_ms = await run_stream(
                base_url, case["stream_chunks"]
            )
            rest_supported, rest_unsupported, _ = validate_citations(rest_result)
            stream_supported, stream_unsupported, _ = validate_citations(stream_result)
            paired.append({
                "case_id": case["id"],
                "rest_latency_ms": round(rest_ms, 1),
                "stream_time_to_first_retrieval_ms": round(first_retrieval_ms, 1)
                    if first_retrieval_ms is not None else None,
                "stream_final_latency_ms": round(stream_ms, 1),
                "rest_supported_citations": rest_supported,
                "rest_unsupported_citations": rest_unsupported,
                "stream_supported_citations": stream_supported,
                "stream_unsupported_citations": stream_unsupported,
            })

    def mean(key: str) -> float | None:
        values = [row[key] for row in paired if row[key] is not None]
        return round(statistics.mean(values), 1) if values else None

    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_url": base_url,
        "backend": os.getenv("LLM_BACKEND", "see API environment"),
        "sample_count": len(paired),
        "summary": {
            "mean_rest_latency_ms": mean("rest_latency_ms"),
            "mean_stream_time_to_first_retrieval_ms": mean("stream_time_to_first_retrieval_ms"),
            "mean_stream_final_latency_ms": mean("stream_final_latency_ms"),
        },
        "cases": paired,
        "limitations": [
            "Run against the same API/backend and corpus revision for both paths.",
            "This paired harness measures latency and citation support; small samples do not establish general quality.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--test-file", default="data/eval/test_utterances.json")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output", default="data/streaming_baseline_results.json")
    args = parser.parse_args()
    result = asyncio.run(compare(args.api_url, load_cases(args.test_file, args.limit)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Recorded {result['sample_count']} paired cases in {output}")


if __name__ == "__main__":
    main()
