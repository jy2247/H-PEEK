# H-PEEK

History-Enhanced Predictive KV Cache Management for PEEK Online + vLLM V1.

## P1 hypothesis

PEEK protects cached prefixes when the current waiting queue already exposes
future demand. H-PEEK tests a different signal: whether reuse observed across
past admitted requests can predict that a currently absent prefix will return.

The P1 policy combines two independent terms:

```text
protection(block) = queue_score(block) + history_weight * history_score(block)
```

`queue_score` preserves PEEK's online demand signal. `history_score` is derived
from repeated appearances in a bounded window of prior requests, not merely
from the block's last-access time. This distinction is essential because PEEK
already provides `cluster`, `recency`, and `decay` eviction modes.

## Fixed scope

- Framework: PEEK Online on vLLM V1
- Development model: `Qwen/Qwen2.5-3B-Instruct`
- Main performance model: `Qwen/Qwen2.5-7B-Instruct`
- Hardware target: 1 x RTX A5000 24 GB
- P1 deliverable: Historical Prefix Tracker + history-aware eviction
- Stretch goal: history-aware scheduling only after eviction is validated
- Initial target: a working prototype and preliminary benchmark in two days

## Documents

- [Implementation plan](docs/PLAN.md)
- [Technical design](docs/DESIGN.md)
- [Benchmark protocol](docs/BENCHMARK.md)
- [Baseline checkpoint](docs/BASELINE.md)
- [Locked environment](docs/ENVIRONMENT.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [LaTeX paper-style report](paper/main.tex)
- [Open decisions](docs/OPEN_QUESTIONS.md)

## Repository status

- Upstream PEEK is pinned as `third_party/peek` submodule at
  `3aca72b63c1dd6433193c0857b965acecdbfbb8a`.
- Parent and submodule development branches are named `h-peek-history`.
- Isolated project environment is installed under `.venv`, `.cargo`, and
  `.rustup`; none modifies the system Python environment.
- Vanilla vLLM, original PEEK, and H-PEEK integration smoke runs pass on one
  RTX A5000 with Qwen2.5-3B-Instruct.
- The bounded admitted-request tracker and the two required history eviction
  branches are implemented and covered by CPU integration tests.
- A first 65,536-token pressure run shows that both H-PEEK modes retain the
  absent 3,456-token hot prefix, while Vanilla, PEEK `cluster`, `recency`, and
  `decay` recompute it. This is a one-run mechanism check, not a performance
  claim; see [Baseline checkpoint](docs/BASELINE.md).
- A paired three-seed matrix with the same PEEK scheduler now covers periodic
  reuse, single-target drift, no-sharing concurrency, queue sharing, a strong
  hotspot switch, history windows 64/256/512, and KV capacities
  2,048/4,096/8,192 blocks. It also covers open-loop 4/8/12/16 req/s hot/cold
  fairness and overload behavior. The continuously updated tables are in
  [paper/main.tex](paper/main.tex).
- The first overhead ablation, `HPEEK_HISTORY_MIN_FREQUENCY=2`, reduces matched
  three-workload history scans by 37.6% and eviction time by 44.4% while
  preserving the periodic reuse hit count. Its raw results and profile are in
  `results/matrix/`.
- The window sweep now includes `W=32` on the 80-request gap trace: it expires
  the hot prefix exactly as expected, while `W=256` retains it.

## Upstream references

- PEEK repository: <https://github.com/xiexbing/peek>
- PEEK paper: <https://arxiv.org/abs/2607.02525>
- vLLM repository: <https://github.com/vllm-project/vllm>
