# Benchmark Protocol

## Policies

Use the same model, request trace, cache budget, engine settings, output length,
and scheduler for every policy.

| ID | Policy | Purpose |
| --- | --- | --- |
| V | Vanilla vLLM V1 + APC | Engine baseline |
| P-C | PEEK Online `cluster` | Primary PEEK baseline |
| P-R | PEEK Online `recency` | Existing linear age control |
| P-D | PEEK Online `decay` | Existing exponential age control |
| H-O | H-PEEK `history_only` eviction | History ablation |
| H-J | H-PEEK `history` eviction | Proposed joint method |
| H-G | H-PEEK `history` with `HPEEK_HISTORY_MIN_FREQUENCY=2` | Minimum-evidence gate ablation |

An optional `plain` PEEK row is useful for diagnosis but does not replace any
required baseline above.

## Workloads

### W-History: intended positive case

Create multiple request waves over reusable prefixes A, B, and C:

1. History-building phase: A recurs often enough to become historically hot.
2. Absence phase: no A request is waiting.
3. Pressure phase: unique long prefixes force cache eviction while A has zero
   current queue demand.
4. Return phase: A requests arrive after the eviction decision.

The pressure and return timings must be fixed in a saved trace, not adapted to
the policy. This workload directly tests the project hypothesis.

### W-Zipf: stationary recurring prefixes

Sample prefix groups with a fixed Zipf distribution. Vary skew and offered load
to test whether historical frequency helps under realistic repeated popularity.

### W-Drift: negative/adaptation case

Make A hot early, then permanently switch demand to B/C. This measures false
protection and how quickly the bounded window forgets A.

### W-Singleton: no-reuse negative case

Every request has a unique prefix. H-PEEK should stay close to the strongest
baseline and its tracker/scan overhead must be visible.

### W-QueueShare: original PEEK strength

Keep several siblings of the same prefix in the current waiting queue. H-PEEK
must not erase the value of PEEK's current-demand signal.

## Pressure levels

At minimum, run a low-pressure control and a high-pressure setting that causes
repeatable eviction. Calibrate pressure on the 3B model before the formal 7B
runs. Record the effective number of GPU KV blocks, not only the configured
`gpu_memory_utilization`.

## Models and run sequence

- Development: `Qwen/Qwen2.5-3B-Instruct`
- Primary: `Qwen/Qwen2.5-7B-Instruct`
- Hardware: one RTX A5000 24 GB
- Seeds: 42, 142, 242
- Warmup: exclude a fixed 10-20% prefix of each trace, declared before analysis
- Decoding: deterministic settings and fixed maximum output tokens

For each seed, randomize policy order. Restart the server and verify an empty
prefix cache before every measured run. Save the exact request trace so all
policies receive identical arrival times and prompts.

## Primary metrics

- cached prompt tokens / cacheable prompt tokens;
- prefix-cache hit rate;
- TTFT p50, p95, p99;
- end-to-end latency p50, p95, p99;
- requests/second and output tokens/second;
- completed requests and errors/timeouts.

## Mechanism and overhead metrics

- fraction of eviction decisions where history changes the victim;
- reuse-survival rate for historically protected blocks;
- useful versus false history protections using future trace ground truth;
- eviction decision time and number of scanned blocks;
- tracker update time, lookup time, current/peak entries, and estimated bytes;
- PEEK fallback frequency caused by scan caps.

## Reporting rules

- Report each seed and the mean with standard deviation.
- Never mix one-seed and three-seed results without labeling them.
- Treat cache-hit improvement without TTFT/throughput improvement as a partial
  result, not an overall win.
- Report negative results and workload drift explicitly.
- Use the same PEEK scheduler flags for P-C, P-R, P-D, and H.
- Freeze H-PEEK parameters after 3B tuning; do not retune on 7B test traces.

## Preliminary success criteria

These are engineering gates, not publication claims:

1. On W-History under pressure, H-PEEK improves cached-token ratio over P-C and
   both existing age-based modes, with the decision log showing protection while
   current queue demand is zero.
2. The improvement translates into lower TTFT or higher throughput on 7B.
3. On W-Singleton, throughput regression stays within 3% and p95 TTFT within 5%
   of P-C; otherwise optimize/short-circuit the tracker before expanding scope.
4. Tracker memory remains bounded and eviction overhead does not dominate a
   scheduler iteration.

## Current preliminary coverage

The first one-run W-History mechanism check covers V, P-C, P-R, P-D, H-O, and
H-J at 4,096 KV blocks. A subsequent paired, three-seed matrix covers W-History,
W-Drift, W-Singleton, W-QueueShare, a strong hotspot switch, history-window
expiry, capacities of 2,048/4,096/8,192 blocks, and a 4/8/12/16 req/s
open-loop hot/cold fairness sweep. Results are recorded in
`docs/BASELINE.md`, `paper/main.tex`, and `results/matrix/`.

Still pending are more than three seeds, policy-order randomization,
public/production traces, and the 7B model. The open-loop study supplies an
initial class-level Jain latency-equity metric, but request-level fairness and
long-duration overload remain future work. Current results remain prototype
evidence rather than a full PEEK paper reproduction.

The first implementation ablation adds `HPEEK_HISTORY_MIN_FREQUENCY`. With the
value set to 2, one-request historical evidence contributes no protection
score. On the matched periodic, stale-history, and singleton traces, this gate
reduced H-PEEK block scans by 37.6%, eviction time by 44.4%, and historical
lookups by 63.6% while preserving the periodic cached-token count. Raw outputs
are `results/matrix/hpeek_history_minfreq1_matched_4096blocks.json` and
`results/matrix/hpeek_history_minfreq2_4096blocks.json`.

The next window point is now complete: with an 80-request absence gap,
`HPEEK_HISTORY_WINDOW=32` produces zero return hits and recomputes 3,007 tokens,
matching the previously measured `W=64` behavior. `W=256` retains 2,992 tokens
on the same trace. The open-loop `min_frequency=2` run also preserved all cache
hit counts while reducing its matched profile's history lookups by 89.9% versus
default H-PEEK.
