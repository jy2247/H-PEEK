#!/usr/bin/env python3
"""Deterministic history-only opportunity trace for vLLM/PEEK/H-PEEK."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer


METRICS = (
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:prompt_tokens_recomputed_total",
    "vllm:prompt_tokens_by_source_total",
)


def metric_snapshot(base_url: str) -> dict[str, float]:
    response = requests.get(f"{base_url}/metrics", timeout=10)
    response.raise_for_status()
    result: dict[str, float] = {}
    for line in response.text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        series, value = line.rsplit(" ", 1)
        if not series.startswith(METRICS):
            continue
        # Preserve source labels because they distinguish compute and hit.
        if 'source="external_kv_transfer"' in series:
            continue
        result[series] = float(value)
    return result


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {
        key: value - before.get(key, 0.0)
        for key, value in after.items()
        if value - before.get(key, 0.0) != 0
    }


def complete(base_url: str, model: str, prompt: str) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_token_at = None
    usage = None
    with requests.post(
        f"{base_url}/v1/completions",
        json=payload,
        stream=True,
        timeout=120,
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():
            if not raw_line or not raw_line.startswith(b"data: "):
                continue
            body = raw_line[6:]
            if body == b"[DONE]":
                continue
            event = json.loads(body)
            choices = event.get("choices") or []
            if choices and choices[0].get("text") and first_token_at is None:
                first_token_at = time.perf_counter()
            if event.get("usage") is not None:
                usage = event["usage"]
    ended = time.perf_counter()
    return {
        "ttft_s": None if first_token_at is None else first_token_at - started,
        "e2e_s": ended - started,
        "usage": usage,
    }


def fixed_token_prompt(tokenizer, marker: str, target_tokens: int) -> str:
    prefix = f"PREFIX ID {marker}. This identifier distinguishes the cache chain. "
    filler = "Deterministic cache pressure content with stable tokenization. "
    # Token IDs are not guaranteed to survive decode/encode unchanged. Build text
    # first, then select the longest character prefix with the requested token
    # count so the server receives an exactly sized prompt prefix.
    text = prefix
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
        text += filler
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        token_count = len(
            tokenizer.encode(text[:middle], add_special_tokens=False)
        )
        if token_count <= target_tokens:
            low = middle
        else:
            high = middle - 1
    prompt = text[:low]
    actual = len(tokenizer.encode(prompt, add_special_tokens=False))
    if actual != target_tokens:
        raise ValueError(
            f"could not construct an exact {target_tokens}-token prompt: {actual}"
        )
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--model", default="qwen2.5-3b")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-tokens", type=int, default=3456)
    parser.add_argument("--history-repeats", type=int, default=4)
    parser.add_argument("--pressure-count", type=int, default=96)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    prefix_a = fixed_token_prompt(tokenizer, "HOT-A", args.prefix_tokens)
    pressure_prompts = [
        fixed_token_prompt(tokenizer, f"COLD-{index:04d}", args.prefix_tokens)
        for index in range(args.pressure_count)
    ]

    start_metrics = metric_snapshot(args.base_url)
    history_results = [
        complete(
            args.base_url,
            args.model,
            f"{prefix_a}\nA history request {index}.",
        )
        for index in range(args.history_repeats)
    ]
    after_history = metric_snapshot(args.base_url)

    pressure_results = []
    for index, prompt in enumerate(pressure_prompts):
        pressure_results.append(
            complete(
                args.base_url,
                args.model,
                f"{prompt}\nCold pressure request {index}.",
            )
        )
        if (index + 1) % 16 == 0:
            print(f"pressure progress: {index + 1}/{args.pressure_count}", flush=True)
    after_pressure = metric_snapshot(args.base_url)

    return_result = complete(
        args.base_url,
        args.model,
        f"{prefix_a}\nA return request after the absence gap.",
    )
    after_return = metric_snapshot(args.base_url)

    pressure_ttft = [r["ttft_s"] for r in pressure_results if r["ttft_s"]]
    result = {
        "config": {
            "model": args.model,
            "prefix_tokens_requested": args.prefix_tokens,
            "history_repeats": args.history_repeats,
            "pressure_count": args.pressure_count,
        },
        "history_requests": history_results,
        "pressure_summary": {
            "count": len(pressure_results),
            "ttft_mean_s": statistics.fmean(pressure_ttft),
            "ttft_p95_s": sorted(pressure_ttft)[
                max(0, int(0.95 * len(pressure_ttft)) - 1)
            ],
        },
        "return_request": return_result,
        "metric_deltas": {
            "history_build": metric_delta(start_metrics, after_history),
            "pressure": metric_delta(after_history, after_pressure),
            "return": metric_delta(after_pressure, after_return),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
