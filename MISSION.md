# Mission: StrataFold

> **Fold experts, not precision.**

## Outcome

Create and publicly ship an original portfolio-grade laboratory for **structural MoE compression without additional quantization**, owned by Omar Ibrahim. It must demonstrate senior/high-level ML-systems work through honest research design, safe artifact handling, byte-exact accounting, deterministic experiments, an original compression hypothesis, a compact runtime, and a polished evidence-backed release.

The primary target adapter is the official `deepseek-ai/DeepSeek-V4-Flash-0731`, but this small AWS host is an orchestration/CI machine, not a full-model machine. The project must be genuinely runnable here through pinned metadata and generated micro-checkpoints, while the full 166.9-GB campaign is guarded behind explicit hardware requirements.

## Required first actions

1. Create one persistent Goal for this entire mission, with no token budget, and begin immediately.
2. Inventory the workspace, disk/RAM/CPU, installed toolchains, GitHub identity/scopes, and configured Codex features without exposing secrets. Do not touch other agents.
3. Re-check the target's official Hugging Face page, API, config, LICENSE, weight index, inference reference, and technical report. Pin exact revisions and record retrieval dates/hashes.
4. Search current primary literature and official implementations as of the execution date. Start from the research ledger below, verify every claim, and explicitly distinguish peer-reviewed papers from preprints.
5. Inspect `https://github.com/FareedKhan-dev/kimi-k3-in-c` read-only only to extract product lessons such as a resource doctor, offline demo, validation ladder, and clear resource ledger. Record the studied commit and license; copy nothing.
6. Write the clean-room/provenance boundary, claims taxonomy, architecture decision, benchmark contract, disk threat model, and name review before implementation claims.
7. Confirm the name and create a public GitHub repository under `omar07ibrahim` early. Use `stratafold` if the name review is clean; otherwise select and document a distinct alternative. Add an honest description and useful topics.
8. Build and push a green vertical slice quickly: `doctor -> inspect pinned metadata -> synth fixture -> compress baseline -> verify -> benchmark JSON`. Then deepen it milestone by milestone.

## Baseline facts that must remain explicit

- Official model: `https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731`.
- Bootstrap revision: `7872f01b1d1fe23eabc4c98b48bffcef5a386062`; verify before pinning.
- Published repository: 48 weight shards, approximately 166,898,661,074 bytes total; tensor payload index approximately 166,878,536,440 bytes.
- Final model page: roughly 304B parameters. The earlier technical report describes a 284B/13B-activated preview/core, so do not substitute that number for the final 0731 artifact.
- Native release already contains FP4 experts and an FP8 configuration. All StrataFold compression claims are relative to the native artifact and/or an explicitly equal-dtype micro baseline. No additional quantization counts as a result.
- Official full serving examples expect serious GPU hardware (for example a four-GPU GB300-class node). This AWS host must never claim full-model inference.

## Research ledger to verify and cite

Prioritize primary sources and official code. Create a dated comparison table covering objective, true parameter reduction, runtime needs, calibration/training cost, target architectures, license, and known failure modes.

- DeepSeek-V4 technical report: `https://arxiv.org/abs/2606.19348`.
- REAP expert pruning (ICLR 2026): `https://arxiv.org/abs/2510.13999`, official Apache-2.0 code `https://github.com/CerebrasResearch/reap`.
- REAM expert merging (2026 preprint): `https://arxiv.org/abs/2604.04356`, official MIT code `https://github.com/SamsungSAILMontreal/ream`.
- EvoESAP non-uniform cross-layer allocation: `https://arxiv.org/abs/2603.06003`.
- GRAPE global redundancy-aware allocation: `https://arxiv.org/abs/2604.06542`.
- MoBE shared expert bases: `https://arxiv.org/abs/2508.05257`; treat its code as read-only research if a usable license is absent.
- D2-MoE shared base plus low-rank deltas: `https://arxiv.org/abs/2502.17298`.
- TD-MoE cross-expert Tucker decomposition (ICLR 2026): `https://openreview.net/forum?id=D9cnZNZfxX`.
- Sub-MoE subspace expert merging: `https://arxiv.org/abs/2506.23266`.
- RFID-MoE routing-frequency/information-density rank allocation: `https://arxiv.org/abs/2602.09316`.
- MoE-I2 inter-expert pruning plus intra-expert low-rank decomposition: `https://arxiv.org/abs/2411.01016`.
- Lillama local feature distillation (NAACL 2025): `https://aclanthology.org/2025.naacl-long.291/`.
- MoE-to-dense pruning/distillation: `https://arxiv.org/abs/2605.28207`.
- FlashMemory for DeepSeek-V4: `https://arxiv.org/abs/2606.09079`; classify it as KV/runtime residency, not weight compression.
- ZipNN: `https://arxiv.org/abs/2411.05239`; classify it as lossless disk/network compression, not parameter or steady-state memory reduction.

Do not assemble an unprincipled grab bag. The first production baseline is REAP-style whole-expert pruning, the mandatory comparison is REAM-style merging, and the original line is Route-Stratified Factorization. Add rank allocation/decomposition ideas only behind fair matched-byte evaluations.

## Technical architecture

```text
pinned official metadata
          |
          v
disk doctor + target/claims ledger
          |
          v
deterministic exact-topology scaled fixtures
          |
          v
router traces + activation/output sketches
          |
          +--> equal-dtype baselines (SVD / REAP / REAM)
          |
          v
co-routing-constrained functional clustering
          |
          v
route-stratified shared factors + sparse residual repair
          |
          v
global byte/error planner (may abstain)
          |
          v
checksummed .sfc artifact + Python oracle + Rust mmap runtime
          |
          v
raw JSON evidence -> reproducible dashboard/report
```

### Original hypothesis: Route-Stratified Factorization

For each MoE layer, collect route frequency, co-routing, and output/activation sketches on a documented calibration mixture. Cluster experts that are functionally similar while penalizing merges of experts that commonly co-activate and supply complementary outputs. Learn shared factors within hot/warm/cold or learned strata, allocate ranks globally under a byte and quality budget, and recover sensitive errors with block-sparse per-expert residuals.

Optimize route-weighted output error, not only static weight reconstruction error. Always compare at equal serialized-byte budget. The planner must preserve a layer or return `INFEASIBLE` if held-out error is unsupported. This is a hypothesis to test against strong baselines, not a pre-announced breakthrough.

### Components

- Python control plane with no heavyweight dependency on this host: doctor, official metadata snapshot, schema/byte estimator, fixture generation, calibration/sketches, baselines, factorization, planner, evaluation, plots.
- Rust runtime and `.sfc` container: checksummed metadata, bounded parser, mmap-friendly tensors, factorized FFN plus sparse residual execution without reconstructing full weights.
- Fixtures:
  - algebraic low-rank: should reconstruct almost exactly;
  - structured/trained micro: primary compression/quality benchmark;
  - adversarial full-rank: must make the planner abstain.
- Evidence pipeline: all tables, Pareto curves, routing heatmaps, Sankey/strata diagrams, and README badges derive from committed JSON, never hand-entered results.

## Milestones and gates

### M0 — provenance, truth contract, and resource safety

- Publish name review, source/license ledger, clean-room ADR, `CLAIMS.json`, `PROVENANCE.yaml`, `CITATIONS.bib`, third-party notices, benchmarking contract, threat model, and exact target ledger.
- Implement a disk/download doctor that checks free space and rejects forbidden extensions, LFS/Xet weight redirects, oversized objects, and unpinned metadata.
- Gate: a simulated `.safetensors` request fails before the first payload byte; secrets/license/claims checks pass; the README distinguishes measured, derived, projected, and unverified evidence.

### M1 — pinned target inspection and byte-exact schema

- Snapshot only permitted official metadata with hashes and an offline manifest.
- Parse the config/schema and report topology, dtype classes, optional DSpark tensors if identifiable, and separate parameter/artifact/state/active-compute ledgers.
- Gate: estimators match generated artifact byte counts exactly and schema changes fail closed with a clear diff.

### M2 — deterministic micro model and oracle

- Generate a scaled fixture preserving the 43-layer schedule, 256 routed + one shared expert topology, top-6 routing, hash-layer markers, and DSpark metadata while reducing widths and stored data enough for CI.
- Implement separate Python reference routing/FFN behavior and deterministic calibration traces.
- Gate: clean checkout reproduces fixture hashes; algebraic, structured, and adversarial cases are covered; the demo is offline after the metadata snapshot.

### M3 — strong equal-dtype baselines

- Implement independent SVD, uniform expert pruning, REAP-style saliency, and REAM-style alignment/merging at 10/20/25% structural budgets on the micro fixture.
- Implement multi-domain calibration slices and sensitivity reporting. Add non-uniform layer allocation only after uniform results are correct.
- Gate: matched-byte Pareto results, held-out KL/top-1/output error, and negative results are reproducible from raw JSON; no method is assumed superior.

### M4 — StrataFold method and planner

- Implement functional sketches, co-routing-aware clustering, shared factorization, rank allocation, sparse residual repair, and a constraint-aware global planner.
- Gate: algebraic fixture max reconstruction error `<1e-5`; the adversarial fixture returns `INFEASIBLE`; on the structured micro fixture the release target is at least `1.5x` equal-dtype byte reduction with KL `<=0.02`, top-1 agreement `>=95%`, and better held-out quality than independent SVD at the same byte budget. These are release criteria, not claims until measured.

### M5 — safe artifact and runtime

- Define a versioned, checksummed `.sfc` format and bounded parser. Implement scalar Python loading and a Rust mmap execution path that evaluates factors/residuals without materializing full expert weights.
- Gate: malformed/corrupt/truncated cases fail safely; Python/Rust logits have max absolute difference `<=1e-4`; serialized byte accounting is exact; peak RSS and latency are measured repeatedly.

### M6 — polished reproducible release

- Provide `make doctor`, `make demo`, `make test`, and `make report` or equally simple cross-platform commands.
- Create an original restrained identity: midnight navy `#08111F`, electric cyan `#2DE2E6`, amber `#FFB000`, off-white `#EAF2F8`; an SVG ribbon/fold mark; architecture diagrams from source; responsive static report/dashboard.
- Use Apache-2.0 for code and document the chosen docs/assets license. Add security, contributing, code of conduct, citation, issue/PR templates, CI, SBOM, and a tagged release.
- Gate on this host: `make demo` in under 10 minutes on 4 vCPU, peak RSS under 4 GiB, total workspace/cache/artifacts under 750 MiB, and at least 2 GiB disk left. Clean clone is green and prints exactly the evidence level, including:

```text
STRUCTURAL COMPRESSION: VERIFIED ON MICRO FIXTURE
DEEPSEEK-V4-0731 SCHEMA: VALIDATED
FULL 166.9-GB CHECKPOINT: NOT DOWNLOADED / NOT RUN
```

## Full-model campaign, explicitly outside this host

Create a dry-run-only adapter and hardware runbook that calculates required disk, CPU RAM, HBM, temporary conversion space, calibration data, wall time, and cost before any run. Pin all artifacts. Compare REAP, REAM, non-uniform allocation, and StrataFold at equal native-dtype bytes and identical evaluation/calibration matrices. Test 10/20/25% before aggressive ratios. Audit whether removing the optional DSpark attachment changes base logits and record its exact byte contribution instead of guessing.

Do not rent hardware, spend money, download weights here, or claim full-model validation. Those require separate explicit user authorization and suitable infrastructure.

## Improvement loop

After the first honest release, keep the Goal active and improve the highest-impact defensible backlog: stronger baselines, calibration robustness, planner quality, factorized kernels, format fuzzing, real smaller open MoE reproduction that fits authorized hardware, adapter generality, documentation gaps, or review findings. Avoid endless cosmetic churn. Close work only with evidence.

## Non-negotiables

- Never download target weight shards on this AWS host.
- Never count additional quantization, routing sparsity, KV compression, offload, or archive compression as structural parameter reduction.
- Never copy or cosmetically rewrite the Kimi prior-art repository or research code without a verified license and explicit provenance.
- Never fabricate benchmarks, hardware runs, quality retention, users, stars, affiliations, dates, or support.
- Never expose credentials, publish a network service, alter other agents, or spend money.
- Never sacrifice correctness, provenance, or honest limitations for an impressive headline.
