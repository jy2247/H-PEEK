# Two-Day P1 Plan

## Definition of done

P1 is complete when all of the following are true:

1. H-PEEK runs on the pinned PEEK Online + vLLM V1 stack with the scheduler
   unchanged.
2. A bounded Historical Prefix Tracker records cross-request prefix frequency.
3. Eviction combines current waiting-queue demand and historical demand.
4. Unit tests prove that a prefix with zero current demand can be protected by
   history and that stale/cold history is eventually removed.
5. One controlled trace demonstrates the intended history-only opportunity.
6. Preliminary results include vanilla vLLM, original PEEK `cluster`, PEEK
   `recency`, PEEK `decay`, and H-PEEK under identical conditions.
7. Results report performance, policy overhead, and a no-reuse negative case.

## Day 1: environment and prototype

### Morning — pin and reproduce

- Clone PEEK into this project and record the upstream commit SHA.
- Use PEEK's tested vLLM/Python/Torch versions as the initial compatibility
  target; do not silently upgrade components during P1.
- Build the Rust extension and run CPU tests before installing the GPU stack.
- Start vanilla vLLM and PEEK Online with Qwen2.5-3B-Instruct.
- Capture a short baseline run and verify that prefix-cache metrics are visible.

Gate: do not modify eviction until the original PEEK baseline can be reproduced.

### Midday — tracker in isolation

- Implement `HistoricalPrefixTracker` as a bounded sliding window over prior
  admitted requests.
- Index by vLLM raw block hash, matching PEEK's existing eviction index.
- Count each block hash at most once per request.
- Store only hashes and counters; never retain prompt text.
- Add deterministic unit tests for insert, expiry, deduplication, capacity,
  prefix depth, and reset.

Gate: tracker cost is O(number of full prefix blocks in the admitted request),
and retained state is bounded by the configured history window.

### Afternoon — eviction integration

- Add a new opt-in mode such as `PEEK_ONLINE_EVICTION_MODE=history`.
- Reuse PEEK's vLLM `BlockPool.get_new_blocks` scan rather than rewriting the
  scheduler or block allocator.
- Update history when requests leave waiting state for admission, after the
  current waiting-demand snapshot is adjusted.
- Score every cached candidate, including candidates with current demand zero.
- Preserve stock behavior when history is disabled or empty.
- Add counters for history lookup hits, history-protected blocks, scan length,
  tracker memory, and decision overhead.

Gate: CPU/integration tests pass and disabling H-PEEK is behaviorally identical
to the pinned upstream PEEK mode.

### Evening — synthetic trace and smoke benchmark

- Generate a deterministic wave trace with hot prefixes, a queue-absence gap,
  cache pressure from one-off prefixes, and later hot-prefix return.
- Run one seed on the 3B model for all five required policies.
- Inspect event logs to confirm that H-PEEK protects the intended block while
  its current queue demand is zero.

## Day 2: tune once, then measure

### Morning — bounded tuning on 3B

- Tune only two parameters: history window and history weight.
- Use a small preregistered grid and one development seed.
- Select parameters using cache-hit improvement subject to no material latency
  regression on the singleton negative trace.
- Freeze parameters before running the 7B evaluation.

Suggested initial grid:

| Parameter | Values |
| --- | --- |
| History window (admitted requests) | 128, 512, 2048 |
| History weight | 0.5, 1.0, 2.0 |

### Afternoon — 7B preliminary benchmark

- First run one seed for every policy to catch configuration failures.
- Then run three seeds for vanilla vLLM, PEEK `cluster`, and H-PEEK.
- If time permits, run three seeds for `recency` and `decay`; otherwise label
  their one-seed numbers as smoke results and schedule the formal repetitions.
- Reset the server and cache between runs, preserve request traces, and randomize
  policy execution order within each seed.

### End of day — analysis

- Produce a machine-readable result table plus plots for prefix-cache hit ratio,
  TTFT, E2E latency, and throughput.
- Report p50/p95/p99, not only averages.
- Report scheduler/eviction overhead and tracker memory.
- Separate "history signal was active" runs from accidental wins caused by
  changed arrival timing.
- Write a go/no-go note for the optional scheduling extension.

## Priority order under time pressure

1. Correct historical signal and instrumentation.
2. History-only positive trace plus no-reuse negative trace.
3. Required policy comparison on 3B.
4. Core three-policy comparison on 7B with three seeds.
5. Full `recency`/`decay` repetitions.
6. Additional workloads and history-aware scheduling.

## Explicit non-goals for P1

- Reimplementing PEEK's cLPM scheduler.
- Training a neural predictor.
- Multi-GPU or distributed history.
- Persisting history across server restarts.
- Semantic similarity or approximate text matching.
- Claiming novelty from recency or exponential time decay alone.
