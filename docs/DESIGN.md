# P1 Technical Design

## Upstream behavior relevant to H-PEEK

PEEK Online's vLLM hook maintains an inverted index from raw vLLM block hash to
the number of waiting requests whose prefixes contain that block. Its patched
`BlockPool.get_new_blocks` scans free cached blocks and protects blocks with
higher queue-derived priority.

Existing modes already cover:

- `plain`: current waiting demand;
- `cluster`: current demand weighted by prefix-chain depth;
- `recency`: current demand minus a last-access age penalty;
- `decay`: current demand multiplied by exponential last-access decay.

Therefore H-PEEK must use multiple prior request observations. A renamed access
timestamp or a different half-life is not sufficient.

## Historical Prefix Tracker

### Event definition

For P1, one observation is one request admitted from the waiting queue. History
is updated at admission, not initial arrival, so the same request is not counted
simultaneously as both current queue demand and past demand.

For each admitted request:

1. Read its complete, reusable vLLM block-hash prefix.
2. Deduplicate hashes within the request defensively.
3. Append the hash list to a fixed-size FIFO window.
4. Increment per-hash request frequency.
5. Expire the oldest request and decrement its hashes when the window is full.

State:

```text
request_window: deque[list[BlockHash]]
history_count: dict[BlockHash, int]
```

The actual vLLM integration keys history by `BlockHashWithGroupId`, not raw
`BlockHash`. When a cache group uses a larger block size, hashes are converted
with vLLM's own `BlockHashListWithBlockSize` before the group ID is packed.

The exact counter is preferred over a sketch for the two-day prototype because
it makes correctness and attribution straightforward. A Count-Min Sketch or
SpaceSaving table is a later scalability option only if measured memory demands
it.

### Prediction

Let:

- `q_b` be current PEEK waiting demand for block `b`;
- `h_b` be the number of prior admitted requests containing `b` within the last
  `W` admitted requests;
- `alpha` be a configured conversion from historical hits to request-equivalent
  predicted demand.

The minimal P1 score is:

```text
priority(b) = q_b + alpha * h_b
```

Higher priority means harder to evict. Ties retain the existing free-queue/LRU
order. The bounded request window prevents an old dominant prefix from becoming
permanent. Normalization, caps, and time-based TTL are intentionally deferred
until the unnormalized MVP has been measured.

This formula is deliberately simple. It tests whether cross-request frequency
adds signal before introducing inter-arrival models, learned weights, or a
history-aware scheduler.

### Critical behavioral change

Original PEEK immediately considers a cached block evictable when its current
waiting demand is zero. H-PEEK must instead calculate history priority for such
blocks. Otherwise the new signal can never solve the stated A-prefix absence
case.

## Lifecycle and integration

Recommended implementation boundaries:

```text
python/peek/online/history.py
    HistoricalPrefixTracker
    HistoryConfig

python/peek/online/engines/vllm/patch_hook.py
    admission lifecycle hook -> tracker.observe(...)
    eviction scan -> joint priority
    diagnostics -> JSON/profile output
```

Keep the scheduler configuration identical between original PEEK and H-PEEK.
Only the eviction mode changes in the primary experiment.

## Configuration proposal

```text
PEEK_ONLINE_EVICTION_MODE=history
HPEEK_HISTORY_WINDOW=512
HPEEK_HISTORY_WEIGHT=1.0
HPEEK_HISTORY_MAX_KEYS_PER_REQUEST=4096
PEEK_ONLINE_PROFILE=1
PEEK_ONLINE_PROFILE_PATH=/tmp/hpeek_profile_{pid}.json
```

`history_only` ignores waiting demand; `history` uses the joint formula.

Invalid values must fail fast rather than silently falling back during an
experiment.

## Required diagnostics

- admitted requests observed;
- current number and peak number of tracked hashes;
- history evictions/expirations;
- tracker update and lookup nanoseconds;
- eviction calls and blocks scanned;
- candidates with `queue_demand == 0 && history_count > 0`;
- blocks protected specifically because of history;
- selected victims' queue score and history score;
- fallback count when the scan cap prevents a complete choice.

Diagnostics should support a sampled decision log so results can distinguish
mechanism success from timing noise without producing unbounded output.

## Correctness tests

1. Repeated prefix observations increase protection.
2. Duplicate hashes inside one request count once.
3. A zero-current-demand historical prefix outranks a cold singleton.
4. Current queue demand still protects a never-before-seen prefix.
5. Window expiry removes old influence exactly.
6. Request window and per-request key cap bound retained history.
7. Empty/disabled tracker reproduces upstream order.
8. Tracker state stays within the configured bound.
9. Cancelled or rejected requests are not recorded as admitted history.
10. Prefix hash identity includes vLLM's cache-relevant extras/group semantics;
    raw hashes are used only where upstream PEEK already establishes safety.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Historical pollution after workload drift | Bounded request window and score cap |
| Large CPU memory from long prefixes | Fixed window, counters removed at zero, measure peak bytes |
| O(window) updates | O(prefix blocks) incremental FIFO updates only |
| Double-counting current requests | Record at admission, after removal from waiting demand |
| Hash/version mismatch | Reuse PEEK/vLLM hash extraction and pin exact versions |
| Scan overhead | Preserve scan cap and measure decision latency |
| Apparent gain from scheduler changes | Do not modify scheduling in P1 |
| Existing recency explains the gain | Compare `recency` and `decay` on identical traces |

## P1-to-P2 boundary

Only consider history-aware scheduling after eviction demonstrates a causal
benefit. A P2 scheduler could use predicted prefix demand to admit a pioneer
before its siblings arrive, but it introduces fairness and queueing confounders
that are unnecessary for the P1 hypothesis.
