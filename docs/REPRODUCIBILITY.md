# Reproducibility

## Source layout

The parent repository records official PEEK as a Git submodule at the exact
upstream commit below:

```text
repository: https://github.com/xiexbing/peek.git
commit: 3aca72b63c1dd6433193c0857b965acecdbfbb8a
upstream commit date: 2026-08-11
local development branch: h-peek-history
```

Until a public H-PEEK fork exists, the parent repository also stores every
nested-source change in `patches/h-peek-history.patch`. Its current SHA-256 is:

```text
26cf88b40617aea79e8cfddd9ef8e75f63d7a805bfe811aedd0205055b0c0389
```

This avoids depending on an untracked or unreachable nested Git commit. From a
fresh parent clone:

```bash
git submodule update --init --recursive
scripts/apply_hpeek_patch.sh
```

The script refuses to apply if the PEEK checkout is at a different commit or
already contains local changes. After a GitHub fork is available, the patch can
become a normal fork commit and the submodule URL/revision can be advanced.

## Runtime versions

The complete observed component and host matrix is in `docs/ENVIRONMENT.md`.
The critical Python package pins for the prototype are:

```text
Python 3.10.12
vllm 0.19.1
torch 2.10.0+cu128
transformers 5.20.1
peek 0.1.0 from the pinned source tree
```

All packages, Rust tools, and model files live inside the project directory;
the system Python environment is not modified.

## Pressure replay

The preliminary positive-case client is
`benchmarks/history_pressure_client.py`. Each policy server must be restarted
with an empty cache and the same vLLM options:

```text
CUDA_VISIBLE_DEVICES=0
--enable-prefix-caching
--gpu-memory-utilization 0.8
--max-model-len 4096
--num-gpu-blocks-override 4096
--generation-config vllm
```

Policy-specific environment variables are:

```text
Vanilla: no PEEK variables or injection path
PEEK cluster: PEEK_PRESET=peek-online
PEEK recency: PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=recency
PEEK decay: PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=decay
H-PEEK history-only: PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=history_only
H-PEEK joint: PEEK_ONLINE_EVICTION=1 PEEK_ONLINE_EVICTION_MODE=history
```

PEEK modes also require
`PYTHONPATH=third_party/peek/scripts/peek_sitecustomize` so the vLLM spawned
engine process imports the patch. H-PEEK used `HPEEK_HISTORY_WINDOW=256` and
`HPEEK_HISTORY_WEIGHT=1.0`. The minimum-evidence gate ablation additionally
sets `HPEEK_HISTORY_MIN_FREQUENCY=2`; the default value is 1.

## Open-loop load and fairness replay

With either the PEEK cluster or H-PEEK joint server listening on loopback port
18000, run the matched 4/8/12/16 req/s sweep as follows:

```bash
PYTHONPATH=benchmarks .venv/bin/python benchmarks/offered_load_fairness.py \
  --policy POLICY_NAME \
  --tokenizer models/Qwen2.5-3B-Instruct \
  --output results/matrix/POLICY_NAME_offered_load_4096blocks.json
```

Defaults are seeds 42/142/242, 64 requests per point, a 50/50 hot/cold mix,
and one generated token per request. The driver clears vLLM's prefix cache
before each point. The current development endpoint does not clear H-PEEK's
tracker; Prefix IDs are disjoint across points, so prior points cannot supply
cache reuse, while the tracker remains at a steady-state bounded footprint.
For a publication cold-start comparison, restart the server before every point
or add an explicit tracker-reset hook.
