# Benchmark and claims contract

## Baseline

All target-checkpoint comparisons use the release's native mixed-precision representation as the baseline. StrataFold evaluates structural compression without additional quantization. A dtype or bit-width change is disallowed as a compression result. Synthetic experiments use one equal dtype throughout each matched comparison and label that dtype explicitly.

## Separate ledgers

Every experiment reports these quantities independently; none is substituted for another:

1. logical parameter count;
2. serialized tensor bytes and total artifact bytes;
3. peak and steady-state CPU RAM/HBM;
4. KV/cache/state bytes;
5. active parameters and estimated/measured FLOPs;
6. latency distribution and throughput;
7. quality metrics and domain sensitivity.

Sparse routing, KV compression, streaming/offload, and lossless archive compression are controls or runtime techniques. They are not structural stored-parameter compression.

## Evidence tags

- `measured`: produced by a committed command with raw machine-readable evidence on named hardware/software.
- `source-reproduced`: independently recomputed from a pinned primary source or official artifact.
- `derived`: transparent arithmetic over tagged inputs, with formula and units.
- `projected`: model-based estimate whose assumptions and uncertainty are explicit.
- `unverified`: bootstrap input or report not yet independently checked.

No performance, quality, memory, or compression number enters a report unless it is in `CLAIMS.json` and points to committed raw JSON. Failed and negative runs remain in the evidence set.

## Fair comparisons

- Dense, independent SVD, uniform whole-expert pruning, REAP-style saliency pruning, REAM-style merging, and StrataFold use identical source tensors, dtype, calibration/held-out splits, and evaluation code.
- Comparisons are matched by exact serialized tensor bytes. Container overhead is additionally reported.
- Budgets begin at 10%, 20%, and 25% structural reduction. More aggressive settings are exploratory.
- Random seeds, topology, input generators, domain mixture, warmups, repetitions, thread count, and software revision are recorded.
- Calibration domains are reported separately: agentic/tool use, code, math, multilingual/general text, and long-context patterns. Aggregate quality never hides a failed slice.
- The planner may keep a layer unchanged or return `INFEASIBLE`; a requested compression rate is not forced.

## Metrics

The micro oracle reports route-weighted output error, maximum absolute error, held-out KL divergence, top-1 agreement, exact bytes, and per-domain sensitivity. Runtime evidence includes median/p95 latency, throughput, peak RSS, repetitions, and cold/warm status. Full-model quality suites are specified in the hardware runbook and remain `not run` on this host.

## Evidence levels

Schema compatibility, official metadata inspection, exact-topology scaled synthetic behavior, and full-checkpoint experiments are four distinct evidence levels. Every output prints the achieved level verbatim.
