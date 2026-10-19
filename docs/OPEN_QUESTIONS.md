# Open Decisions

The plan can start without these answers, but they should be resolved before the
first formal benchmark.

## Resolved

1. **Upstream pin** — PEEK is a Git submodule pinned at
   `3aca72b63c1dd6433193c0857b965acecdbfbb8a`; changes live on the nested
   `h-peek-history` branch until a fork remote is created.
2. **Environment** — The isolated vLLM environment, CUDA access, disk capacity,
   and Qwen2.5-3B-Instruct weights are ready.
3. **History unit** — One first admission of one request is one history event;
   scheduler iterations, preemptions, and resumes do not add observations.

## Important experiment choices

1. **Workload data** — Are there project-specific production traces, or should
   P1 use deterministic synthetic traces plus a public repeated-prefix dataset?
2. **Cache pressure control** — Resolved for synthetic runs: vLLM 0.19.1 exposes
   `--num-gpu-blocks-override`; the first pressure trace used 4,096 blocks
   (65,536 tokens). Formal runs still need the capacity sweep values locked.
3. **Preliminary vs. formal repetitions** — Two days is realistic for a working
   prototype, one-seed coverage of all five policies, and three-seed coverage of
   the core V/P-C/H comparison. Full three-seed sweeps of all parameters and all
   policies may require additional GPU time.

## Deferred by default

- History persistence across restarts.
- Multi-tenant isolation and per-tenant history.
- Approximate/semantic prefix matching.
- History-aware scheduling.
- Distributed or multi-GPU history coordination.
