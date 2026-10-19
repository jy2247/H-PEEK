#!/usr/bin/env python3
"""Small, deterministic OpenAI-compatible prefix-cache smoke test."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests


def relevant_metrics(base_url: str) -> list[str]:
    response = requests.get(f"{base_url}/metrics", timeout=10)
    response.raise_for_status()
    needles = ("prefix_cache", "prompt_tokens", "cache_hit", "request_prefill")
    return [
        line
        for line in response.text.splitlines()
        if not line.startswith("#") and any(needle in line for needle in needles)
    ]


def run_one(base_url: str, model: str, prompt: str) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": 16,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first_token_at = None
    text_parts: list[str] = []
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
            if choices:
                piece = choices[0].get("text", "")
                if piece and first_token_at is None:
                    first_token_at = time.perf_counter()
                text_parts.append(piece)
            if event.get("usage") is not None:
                usage = event["usage"]
    ended = time.perf_counter()
    return {
        "ttft_s": None if first_token_at is None else first_token_at - started,
        "e2e_s": ended - started,
        "usage": usage,
        "text": "".join(text_parts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18000")
    parser.add_argument("--model", default="qwen2.5-3b")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shared = (
        "H-PEEK cache experiment context. The following paragraph is deliberately "
        "repeated to create a block-aligned reusable prefix. "
    ) * 180
    prompts = [
        f"{shared}\nRequest {index}: return only the integer {index}."
        for index in range(4)
    ]

    result = {
        "base_url": args.base_url,
        "model": args.model,
        "metrics_before": relevant_metrics(args.base_url),
        "requests": [run_one(args.base_url, args.model, prompt) for prompt in prompts],
        "metrics_after": relevant_metrics(args.base_url),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
