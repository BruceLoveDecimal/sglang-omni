#!/usr/bin/env python3
"""Measure Ming-Omni OpenAI text streaming metrics on GSM8K prompts."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import statistics
import time
from pathlib import Path
from typing import Any

import requests


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 95),
        "min": min(values),
        "max": max(values),
    }


def _prompt(question: str) -> str:
    return (
        "Solve the following grade school math problem. "
        "Show concise reasoning and end with the final answer.\n\n"
        f"Question: {question}\nAnswer:"
    )


def _run_one(
    *,
    idx: int,
    question: str,
    answer: str,
    url: str,
    model: str,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _prompt(question)}],
        "stream": True,
        "modalities": ["text"],
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    start = time.perf_counter()
    first_content_at: float | None = None
    last_content_at: float | None = None
    done_at: float | None = None
    non_empty_chunks = 0
    content_parts: list[str] = []
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    raw_events = 0

    try:
        with requests.post(
            f"{url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=timeout,
        ) as resp:
            status_code = resp.status_code
            if status_code != 200:
                return {
                    "idx": idx,
                    "success": False,
                    "status_code": status_code,
                    "error": resp.text[:500],
                    "question": question,
                    "expected_answer": answer,
                }

            for raw_line in resp.iter_lines(decode_unicode=True):
                now = time.perf_counter()
                if not raw_line:
                    continue
                if not raw_line.startswith("data: "):
                    continue
                data = raw_line[len("data: ") :]
                if data == "[DONE]":
                    done_at = now
                    break
                raw_events += 1
                obj = json.loads(data)
                choice = obj.get("choices", [{}])[0]
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str) and content:
                    if first_content_at is None:
                        first_content_at = now
                    last_content_at = now
                    non_empty_chunks += 1
                    content_parts.append(content)
                if choice.get("finish_reason") is not None:
                    finish_reason = choice.get("finish_reason")
                    usage = obj.get("usage") or usage
            if done_at is None:
                done_at = time.perf_counter()

    except Exception as exc:
        return {
            "idx": idx,
            "success": False,
            "error": repr(exc),
            "question": question,
            "expected_answer": answer,
        }

    latency_s = done_at - start if done_at is not None else None
    ttft_s = first_content_at - start if first_content_at is not None else None
    completion_tokens = None
    if isinstance(usage, dict):
        completion_tokens = usage.get("completion_tokens")
    if isinstance(completion_tokens, int) and completion_tokens > 1 and last_content_at and first_content_at:
        tpot_s = (last_content_at - first_content_at) / (completion_tokens - 1)
    elif non_empty_chunks > 1 and last_content_at and first_content_at:
        tpot_s = (last_content_at - first_content_at) / (non_empty_chunks - 1)
    else:
        tpot_s = None

    return {
        "idx": idx,
        "success": True,
        "status_code": 200,
        "explicit_modalities": ["text"],
        "raw_sse_events": raw_events,
        "non_empty_content_chunks": non_empty_chunks,
        "ttft_s": ttft_s,
        "tpot_s": tpot_s,
        "latency_s": latency_s,
        "finish_reason": finish_reason,
        "usage": usage,
        "question": question,
        "expected_answer": answer,
        "generated_text": "".join(content_parts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="ming-omni")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--dataset-jsonl",
        default=None,
        help="Optional local GSM8K JSONL with question/answer fields.",
    )
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_needed = args.limit + args.warmup
    if args.dataset_jsonl:
        examples = []
        with Path(args.dataset_jsonl).open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= total_needed:
                    break
                row = json.loads(line)
                examples.append(
                    {
                        "idx": i,
                        "question": row["question"],
                        "answer": row["answer"],
                    }
                )
    else:
        from datasets import load_dataset

        ds = load_dataset("gsm8k", "main", split=args.split)
        examples = [
            {
                "idx": i,
                "question": row["question"],
                "answer": row["answer"],
            }
            for i, row in enumerate(ds.select(range(min(total_needed, len(ds)))))
        ]
    warmups = examples[: args.warmup]
    measured = examples[args.warmup : args.warmup + args.limit]

    metadata = {
        "dataset": "gsm8k/main",
        "split": args.split,
        "limit": len(measured),
        "warmup": len(warmups),
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "explicit_modalities": ["text"],
        "url": args.url,
        "model": args.model,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for item in warmups:
        result = _run_one(
            idx=item["idx"],
            question=item["question"],
            answer=item["answer"],
            url=args.url,
            model=args.model,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        print("warmup", item["idx"], result.get("success"), result.get("latency_s"))

    results: list[dict[str, Any]] = []
    with futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        pending = [
            executor.submit(
                _run_one,
                idx=item["idx"],
                question=item["question"],
                answer=item["answer"],
                url=args.url,
                model=args.model,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            for item in measured
        ]
        for future in futures.as_completed(pending):
            result = future.result()
            results.append(result)
            print(
                "result",
                result.get("idx"),
                result.get("success"),
                "chunks",
                result.get("non_empty_content_chunks"),
                "ttft",
                result.get("ttft_s"),
                "tpot",
                result.get("tpot_s"),
                "lat",
                result.get("latency_s"),
                flush=True,
            )

    results.sort(key=lambda x: x["idx"])
    jsonl_path = out_dir / "requests.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    successes = [r for r in results if r.get("success")]
    summary = {
        **metadata,
        "successes": len(successes),
        "failures": len(results) - len(successes),
        "ttft_s": _stats([r["ttft_s"] for r in successes if r.get("ttft_s") is not None]),
        "tpot_s": _stats([r["tpot_s"] for r in successes if r.get("tpot_s") is not None]),
        "latency_s": _stats([r["latency_s"] for r in successes if r.get("latency_s") is not None]),
        "non_empty_content_chunks": _stats(
            [float(r["non_empty_content_chunks"]) for r in successes]
        ),
        "completion_tokens": _stats(
            [
                float((r.get("usage") or {}).get("completion_tokens"))
                for r in successes
                if isinstance((r.get("usage") or {}).get("completion_tokens"), int)
            ]
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
