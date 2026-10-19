# Baseline Checkpoint

## Configuration

All accepted smoke runs used:

```text
Model: Qwen/Qwen2.5-3B-Instruct
Device: CUDA_VISIBLE_DEVICES=0
vLLM: 0.19.1, V1 engine
Prefix caching: enabled
GPU memory utilization: 0.80
Maximum model length: 4096
Generation config: vllm
Requests: 4 sequential prompts, 3972 prompt tokens each
Shared prefix: approximately 3952 tokens
Output: 16 tokens, temperature 0
```

These runs validate wiring only. They contain no eviction pressure and are not
evidence that one policy outperforms another.

## Smoke results

| Policy | TTFT seconds, requests 1-4 | Cache hits / queries | Recomputed tokens |
| --- | --- | --- | --- |
| Clean Vanilla vLLM | 1.0186, 0.0522, 0.0508, 0.0519 | 11,856 / 15,888 | 0 |
| Original PEEK `cluster` preset | 0.3897, 0.0610, 0.0517, 0.0475 | 11,856 / 15,888 | 0 |
| H-PEEK `history` integration | 0.3851, 0.0550, 0.0549, 0.0535 | 11,856 / 15,888 | 0 |

The first-request TTFT differs because compile/cache warmup was not controlled
for this wiring test. It must not be used as a comparison.

H-PEEK's profile after four requests reported:

```text
history_observe_calls: 4
history_requests_in_window: 4
history_tracked_keys: 251
history_key_references_in_window: 992
```

That verifies one history observation per first admission on the real vLLM
scheduler. No history lookup was expected because free uncached blocks remained
and the smoke trace did not create eviction pressure.

## Stored artifacts

- `results/baseline/vanilla_clean_smoke.json`
- `results/baseline/peek_cluster_smoke.json`
- `results/baseline/hpeek_history_joint_smoke.json`

An earlier `vanilla_smoke.json` was produced before discovering that an offline
test had patched site-packages. It is diagnostically useful but invalid as a
Vanilla result and must not enter analysis.

## Preliminary eviction-pressure result

This deterministic mechanism check used a 65,536-token KV cache
(`--num-gpu-blocks-override 4096`, 16 tokens per block). The trace first issued
four requests with an exactly 3,456-token Prefix A, then 24 unique 3,456-token
prefixes (83,126 prompt tokens total) while no A request was waiting, and
finally requested A again. Every policy received the same sequential trace.

| Policy | A-return cache hit / query tokens | A-return compute tokens | A-return TTFT (s) | Pressure mean TTFT (s) |
| --- | ---: | ---: | ---: | ---: |
| Vanilla vLLM | 0 / 3,465 | 3,465 | 0.3140 | 0.3138 |
| PEEK `cluster` preset | 0 / 3,465 | 3,465 | 0.3210 | 0.3192 |
| PEEK `recency` | 0 / 3,465 | 3,465 | 0.3140 | 0.3139 |
| PEEK `decay` | 0 / 3,465 | 3,465 | 0.3175 | 0.3163 |
| H-PEEK `history_only` | 3,456 / 3,465 | 9 | 0.0518 | 0.3197 |
| H-PEEK `history` (joint) | 3,456 / 3,465 | 9 | 0.0526 | 0.3187 |

In this single run, joint H-PEEK reduced A's recomputed prompt tokens by 99.7%
and its return TTFT by 83.2% versus Vanilla. This is the intended positive-case
trace, with one return request and no repetitions, so these numbers establish
that the eviction mechanism works but are not statistical performance results.
The equal history-only and joint outcomes are expected because waiting demand
for cached A was zero throughout the pressure phase.

The joint H-PEEK engine profile over all 29 requests reported:

```text
history observations: 29 (8.774 ms total)
history lookups: 5,685 (9.010 ms total, about 1.59 us/lookup)
zero-waiting candidates elevated by history: 5,685
eviction calls: 54 (37.316 ms total; 9,792 blocks scanned)
history state: 5,400 unique keys / 6,264 request-key references
```

The tracker state stayed within the configured 256-request window. The
corresponding PEEK `cluster` run performed 55 eviction calls and scanned 3,331
blocks, confirming that this trace exercised real cache pressure. H-PEEK's
larger scan count and about 0.30 ms profiled admission-update cost per request
are optimization targets for the formal throughput experiment.

Pressure artifacts:

- `results/baseline/vanilla_pressure_4096blocks.json`
- `results/baseline/peek_cluster_pressure_4096blocks.json`
- `results/baseline/peek_recency_pressure_4096blocks.json`
- `results/baseline/peek_decay_pressure_4096blocks.json`
- `results/baseline/hpeek_history_only_pressure_4096blocks.json`
- `results/baseline/hpeek_history_pressure_4096blocks.json`
- `results/baseline/peek_cluster_pressure_profile.json`
- `results/baseline/hpeek_history_pressure_profile.json`

The replay driver is `benchmarks/history_pressure_client.py`. It constructs
prefix text by verifying its re-tokenized length, rather than assuming that a
token-ID decode/encode round trip preserves token count.

## Observed upstream issue

Original PEEK emitted a harmless shutdown warning when vLLM called
`finish_requests(None, ...)`. H-PEEK now treats `None` as "finish all tracked
requests". This does not affect measured request execution.

## Extended three-seed checkpoint

The paired comparison was extended with the same PEEK cLPM scheduler on both
sides; only `cluster` versus `history` eviction differs. All values below are
means over seeds 42, 142, and 242.

| Experiment | PEEK | H-PEEK | Finding |
| --- | ---: | ---: | --- |
| Periodic return TTFT, 4,096 blocks | 272.5 ms | 50.9 ms | H-PEEK -81.3%; +2,992 hit tokens |
| Periodic throughput, 4,096 blocks | 4.015 req/s | 4.118 req/s | H-PEEK +2.57% |
| Singleton throughput | 12.908 req/s | 12.895 req/s | H-PEEK -0.10% |
| Queue-sharing throughput | 65.704 req/s | 65.179 req/s | H-PEEK -0.80% |
| Strong-switch new-set hits | 56,848 | 56,848 | No stale-history hit regression |
| Strong-switch return throughput | 21.887 req/s | 21.140 req/s | H-PEEK -3.41% overhead |
| Gap-80 return, window 64 | 0 hit tokens | 0 hit tokens | Exact request-window expiry |
| Gap-80 return, window 256 | 0 hit tokens | 2,992 hit tokens | H-PEEK TTFT -83.8% |
| Periodic return, 2,048 blocks | 0 hit tokens | 2,992 hit tokens | Benefit survives stronger pressure |
| Periodic return, 8,192 blocks | 2,992 hit tokens | 2,992 hit tokens | No hit benefit; H-PEEK throughput -0.63% |
| Open-loop throughput, offered 12 req/s | 11.426 req/s | 11.406 req/s | H-PEEK -0.18% |
| Open-loop throughput, offered 16 req/s | 12.020 req/s | 11.994 req/s | H-PEEK -0.21% |
| Open-loop TTFT P95, offered 16 req/s | 2.089 s | 2.100 s | H-PEEK +0.51% |
| Open-loop TTFT $>1$ s, offered 16 req/s | 23.4% | 23.4% | No observed SLO regression |

The aggregate four-workload profile shows 2.38x more scanned blocks and 3.36x
more eviction time for H-PEEK. The observed risk is therefore CPU-side scan
overhead; the strong-switch test did not reveal cache pollution under the
default scan cap. Full statistics, standard deviations, limitations, and raw
artifact mapping are maintained in `paper/main.tex` and `results/matrix/`.

The open-loop experiment used a randomized 50/50 mixture of one shared hot
2,048-token prefix and unique cold 2,048-token prefixes, with 64 requests per
run and offered loads of 4, 8, 12, and 16 req/s. Cache-hit and recompute-token
counts were identical between policies at every load, so it acts as an overhead
comparison rather than a positive history-reuse case. Across the sweep,
H-PEEK throughput changed by +0.02% to -0.21%. Its largest mean P95 increase
was 6.57% at 12 req/s, where the three-seed standard deviation was large; at
16 req/s the P95 increase was 0.51% and both policies had 23.4% of requests
over one second. The matching engine profiles are
`results/matrix/peek_cluster_offered_load_profile.json` and
`results/matrix/hpeek_history_offered_load_profile.json`.
