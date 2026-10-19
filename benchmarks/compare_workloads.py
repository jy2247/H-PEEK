#!/usr/bin/env python3
"""Multi-workload PEEK versus H-PEEK mechanism and regression matrix."""

from __future__ import annotations

import argparse
import json
import math
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


def summarize_requests(results: list[dict], wall_s: float) -> dict:
    ttft = [item["ttft_s"] for item in results if item["ttft_s"] is not None]
    e2e = [item["e2e_s"] for item in results]
    prompt_tokens = sum(
        (item.get("usage") or {}).get("prompt_tokens", 0) for item in results
    )
    return {
        "requests": len(results),
        "wall_s": wall_s,
        "requests_per_s": len(results) / wall_s,
        "prompt_tokens_per_s": prompt_tokens / wall_s,
        "ttft_mean_s": statistics.fmean(ttft),
        "ttft_p50_s": percentile(ttft, 0.50),
        "ttft_p95_s": percentile(ttft, 0.95),
        "ttft_p99_s": percentile(ttft, 0.99),
        "e2e_p95_s": percentile(e2e, 0.95),
    }


def run_sequential(base_url: str, model: str, prompts: list[str]) -> tuple[list[dict], float]:
    started = time.perf_counter()
    results = [complete(base_url, model, prompt) for prompt in prompts]
    return results, time.perf_counter() - started


def run_concurrent(
    base_url: str,
    model: str,
    prompts: list[str],
    concurrency: int,
) -> tuple[list[dict], float]:
    started = time.perf_counter()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(complete, base_url, model, prompt)
            for prompt in prompts
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return results, time.perf_counter() - started


def reset_prefix_cache(base_url: str) -> None:
    response = requests.post(f"{base_url}/reset_prefix_cache", timeout=30)
    response.raise_for_status()
    # Let the metrics/logging thread observe the idle boundary before snapshots.
    time.sleep(0.2)


def measured(
    base_url: str,
    runner,
) -> tuple[list[dict], float, dict[str, float]]:
    before = metric_snapshot(base_url)
    results, wall_s = runner()
    after = metric_snapshot(base_url)
    return results, wall_s, metric_delta(before, after)


def periodic_reuse(tokenizer, base_url: str, model: str, seed: int) -> dict:
    hot = fixed_token_prompt(tokenizer, f"PERIODIC-{seed}-HOT", 3000)
    history = [f"{hot}\nhistory occurrence {index}" for index in range(4)]
    pressure = [
        fixed_token_prompt(tokenizer, f"PERIODIC-{seed}-COLD-{index}", 3000)
        + f"\npressure {index}"
        for index in range(24)
    ]
    return_prompt = f"{hot}\nreturn after zero waiting demand"

    def runner():
        started = time.perf_counter()
        history_results = [complete(base_url, model, prompt) for prompt in history]
        pressure_results = [complete(base_url, model, prompt) for prompt in pressure]
        return_result = complete(base_url, model, return_prompt)
        return (
            history_results + pressure_results + [return_result],
            time.perf_counter() - started,
        )

    results, wall_s, metrics = measured(base_url, runner)
    return {
        "summary": summarize_requests(results, wall_s),
        "return_request": results[-1],
        "metrics": metrics,
    }


def stale_history_pollution(tokenizer, base_url: str, model: str, seed: int) -> dict:
    # Nineteen formerly-hot prefixes nearly fill the 65,536-token cache. A
    # newer target B is repeated less often. PEEK/LRU should prefer recent B,
    # whereas unnormalized historical frequency can overprotect stale A blocks.
    stale_prefixes = [
        fixed_token_prompt(tokenizer, f"DRIFT-{seed}-STALE-{index}", 3000)
        for index in range(19)
    ]
    target = fixed_token_prompt(tokenizer, f"DRIFT-{seed}-TARGET-B", 3000)
    pressure = [
        fixed_token_prompt(tokenizer, f"DRIFT-{seed}-PRESSURE-{index}", 3000)
        for index in range(4)
    ]
    prompts = []
    for stale_index, prefix in enumerate(stale_prefixes):
        prompts.extend(
            f"{prefix}\nstale {stale_index} occurrence {repeat}"
            for repeat in range(4)
        )
    prompts.extend(f"{target}\nnew target occurrence {repeat}" for repeat in range(2))
    prompts.extend(
        f"{prefix}\npost-switch pressure {index}"
        for index, prefix in enumerate(pressure)
    )
    return_prompt = f"{target}\nnew target return"

    def runner():
        started = time.perf_counter()
        body = [complete(base_url, model, prompt) for prompt in prompts]
        returned = complete(base_url, model, return_prompt)
        return body + [returned], time.perf_counter() - started

    results, wall_s, metrics = measured(base_url, runner)
    return {
        "summary": summarize_requests(results, wall_s),
        "return_request": results[-1],
        "metrics": metrics,
    }


def singleton_overhead(tokenizer, base_url: str, model: str, seed: int) -> dict:
    prompts = [
        fixed_token_prompt(tokenizer, f"SINGLETON-{seed}-{index}", 1024)
        + f"\nunique request {index}"
        for index in range(96)
    ]
    results, wall_s, metrics = measured(
        base_url,
        lambda: run_concurrent(base_url, model, prompts, concurrency=8),
    )
    return {
        "summary": summarize_requests(results, wall_s),
        "metrics": metrics,
    }


def hotspot_switch(tokenizer, base_url: str, model: str, seed: int) -> dict:
    """Oversubscribe cache with lower-frequency members of a new hot set.

    This is intentionally adversarial to the unnormalized MVP score. Nineteen
    old prefixes have frequency four, then nineteen new prefixes have frequency
    two. The cache cannot hold both sets, so a history-only preference for the
    old set can reduce hits when every member of the new set returns.
    """
    old_prefixes = [
        fixed_token_prompt(tokenizer, f"SWITCH-{seed}-OLD-{index}", 3000)
        for index in range(19)
    ]
    new_prefixes = [
        fixed_token_prompt(tokenizer, f"SWITCH-{seed}-NEW-{index}", 3000)
        for index in range(19)
    ]
    build_prompts = []
    for prefix_index, prefix in enumerate(old_prefixes):
        build_prompts.extend(
            f"{prefix}\nold set {prefix_index} occurrence {repeat}"
            for repeat in range(4)
        )
    for prefix_index, prefix in enumerate(new_prefixes):
        build_prompts.extend(
            f"{prefix}\nnew set {prefix_index} occurrence {repeat}"
            for repeat in range(2)
        )
    return_prompts = [
        f"{prefix}\nnew set return {index}"
        for index, prefix in enumerate(new_prefixes)
    ]

    before = metric_snapshot(base_url)
    started = time.perf_counter()
    build_results = [complete(base_url, model, prompt) for prompt in build_prompts]
    before_return = metric_snapshot(base_url)
    return_started = time.perf_counter()
    return_results = [
        complete(base_url, model, prompt) for prompt in return_prompts
    ]
    return_wall_s = time.perf_counter() - return_started
    wall_s = time.perf_counter() - started
    after = metric_snapshot(base_url)
    all_results = build_results + return_results
    return {
        "summary": summarize_requests(all_results, wall_s),
        "return_summary": summarize_requests(return_results, return_wall_s),
        "return_metrics": metric_delta(before_return, after),
        "metrics": metric_delta(before, after),
    }


def history_gap_80(tokenizer, base_url: str, model: str, seed: int) -> dict:
    """Return a hot prefix after 80 admitted, unique cold requests."""
    hot = fixed_token_prompt(tokenizer, f"GAP80-{seed}-HOT", 3000)
    history_prompts = [f"{hot}\nhot occurrence {index}" for index in range(4)]
    gap_prompts = [
        fixed_token_prompt(tokenizer, f"GAP80-{seed}-COLD-{index}", 1024)
        + f"\nabsence-gap request {index}"
        for index in range(80)
    ]
    return_prompt = f"{hot}\nreturn after 80 admitted requests"

    before = metric_snapshot(base_url)
    started = time.perf_counter()
    history_results = [
        complete(base_url, model, prompt) for prompt in history_prompts
    ]
    gap_results = [complete(base_url, model, prompt) for prompt in gap_prompts]
    before_return = metric_snapshot(base_url)
    returned = complete(base_url, model, return_prompt)
    wall_s = time.perf_counter() - started
    after = metric_snapshot(base_url)
    all_results = history_results + gap_results + [returned]
    return {
        "summary": summarize_requests(all_results, wall_s),
        "return_request": returned,
        "return_metrics": metric_delta(before_return, after),
        "metrics": metric_delta(before, after),
    }


def queue_share(tokenizer, base_url: str, model: str, seed: int) -> dict:
    shared = fixed_token_prompt(tokenizer, f"QUEUE-SHARE-{seed}", 2048)
    prompts = [f"{shared}\nconcurrent sibling {index}" for index in range(32)]
    results, wall_s, metrics = measured(
        base_url,
        lambda: run_concurrent(base_url, model, prompts, concurrency=16),
    )
    return {
        "summary": summarize_requests(results, wall_s),
        "metrics": metrics,
    }


SCENARIOS = {
    "periodic_reuse": periodic_reuse,
    "stale_history_pollution": stale_history_pollution,
    "hotspot_switch": hotspot_switch,
    "history_gap_80": history_gap_80,
    "singleton_overhead": singleton_overhead,
    "queue_share": queue_share,
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
        "--scenarios", nargs="+", choices=tuple(SCENARIOS), default=list(SCENARIOS)
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    runs = []
    for seed in args.seeds:
        for scenario_name in args.scenarios:
            reset_prefix_cache(args.base_url)
            print(f"starting {args.policy} {scenario_name} seed={seed}", flush=True)
            result = SCENARIOS[scenario_name](
                tokenizer, args.base_url, args.model, seed
            )
            result.update({"scenario": scenario_name, "seed": seed})
            runs.append(result)
            summary = result["summary"]
            print(
                f"finished {scenario_name} seed={seed}: "
                f"rps={summary['requests_per_s']:.3f} "
                f"p95={summary['ttft_p95_s']:.4f}s",
                flush=True,
            )

    output = {
        "policy": args.policy,
        "seeds": args.seeds,
        "scenarios": args.scenarios,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
