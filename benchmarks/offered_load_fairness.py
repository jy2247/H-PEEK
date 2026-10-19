#!/usr/bin/env python3
"""Open-loop offered-load and hot/cold latency-fairness benchmark."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from transformers import AutoTokenizer

from history_pressure_client import (
    complete,
    fixed_token_prompt,
    metric_delta,
    metric_snapshot,
)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def reset_prefix_cache(base_url: str) -> None:
    response = requests.post(f"{base_url}/reset_prefix_cache", timeout=30)
    response.raise_for_status()
    time.sleep(0.2)


def summarize(records: list[dict], wall_s: float) -> dict:
    ttft = [record["ttft_s"] for record in records]
    e2e = [record["e2e_s"] for record in records]
    return {
        "requests": len(records),
        "requests_per_s": len(records) / wall_s,
        "ttft_mean_s": statistics.fmean(ttft),
        "ttft_p50_s": percentile(ttft, 0.50),
        "ttft_p95_s": percentile(ttft, 0.95),
        "ttft_p99_s": percentile(ttft, 0.99),
        "e2e_p95_s": percentile(e2e, 0.95),
        "ttft_slo_1s_fraction": sum(value > 1.0 for value in ttft) / len(ttft),
    }


def delayed_complete(
    base_url: str,
    model: str,
    prompt: str,
    request_class: str,
    scheduled_at: float,
) -> dict:
    delay = scheduled_at - time.perf_counter()
    if delay > 0:
        time.sleep(delay)
    actual_start = time.perf_counter()
    result = complete(base_url, model, prompt)
    result["class"] = request_class
    result["client_start_lag_s"] = actual_start - scheduled_at
    return result


def run_load(
    tokenizer,
    base_url: str,
    model: str,
    seed: int,
    offered_qps: float,
    request_count: int,
) -> dict:
    shared = fixed_token_prompt(tokenizer, f"LOAD-{seed}-{offered_qps}-HOT", 2048)
    requests_to_send = []
    for index in range(request_count // 2):
        requests_to_send.append(
            (
                "hot",
                f"{shared}\nshared hot request {index}",
            )
        )
        cold = fixed_token_prompt(
            tokenizer, f"LOAD-{seed}-{offered_qps}-COLD-{index}", 2048
        )
        requests_to_send.append(("cold", f"{cold}\nunique cold request {index}"))
    random.Random(seed + int(offered_qps * 1000)).shuffle(requests_to_send)

    reset_prefix_cache(base_url)
    before = metric_snapshot(base_url)
    origin = time.perf_counter() + 0.5
    records = []
    with ThreadPoolExecutor(max_workers=request_count) as executor:
        futures = [
            executor.submit(
                delayed_complete,
                base_url,
                model,
                prompt,
                request_class,
                origin + index / offered_qps,
            )
            for index, (request_class, prompt) in enumerate(requests_to_send)
        ]
        for future in as_completed(futures):
            records.append(future.result())
    finished = time.perf_counter()
    after = metric_snapshot(base_url)
    wall_s = finished - origin

    hot = [record for record in records if record["class"] == "hot"]
    cold = [record for record in records if record["class"] == "cold"]
    hot_summary = summarize(hot, wall_s)
    cold_summary = summarize(cold, wall_s)
    inverse_hot_latency = 1.0 / hot_summary["ttft_mean_s"]
    inverse_cold_latency = 1.0 / cold_summary["ttft_mean_s"]
    latency_equity_jain = (
        (inverse_hot_latency + inverse_cold_latency) ** 2
        / (2 * (inverse_hot_latency**2 + inverse_cold_latency**2))
    )
    return {
        "seed": seed,
        "offered_qps": offered_qps,
        "request_count": request_count,
        "wall_s": wall_s,
        "overall": summarize(records, wall_s),
        "hot": hot_summary,
        "cold": cold_summary,
        "latency_equity_jain": latency_equity_jain,
        "client_start_lag_p95_s": percentile(
            [record["client_start_lag_s"] for record in records], 0.95
        ),
        "metrics": metric_delta(before, after),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--model", default="qwen2.5-3b")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 142, 242])
    parser.add_argument(
        "--offered-qps", nargs="+", type=float, default=[4, 8, 12, 16]
    )
    parser.add_argument("--request-count", type=int, default=64)
    args = parser.parse_args()

    if args.request_count <= 0 or args.request_count % 2:
        raise ValueError("request-count must be a positive even number")
    if any(qps <= 0 for qps in args.offered_qps):
        raise ValueError("offered QPS values must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    runs = []
    for seed in args.seeds:
        for qps in args.offered_qps:
            print(f"starting {args.policy} seed={seed} offered_qps={qps}", flush=True)
            result = run_load(
                tokenizer,
                args.base_url,
                args.model,
                seed,
                qps,
                args.request_count,
            )
            runs.append(result)
            print(
                f"finished seed={seed} qps={qps}: "
                f"throughput={result['overall']['requests_per_s']:.3f} "
                f"p95={result['overall']['ttft_p95_s']:.4f}s "
                f"fairness={result['latency_equity_jain']:.4f}",
                flush=True,
            )

    output = {
        "policy": args.policy,
        "seeds": args.seeds,
        "offered_qps": args.offered_qps,
        "request_count": args.request_count,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
