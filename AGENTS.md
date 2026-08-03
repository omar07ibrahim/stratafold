# StrataFold autonomous engineering rules

These rules apply to this workspace and to the public repository created from it.

## Mission and ownership

- Build **StrataFold — Fold experts, not precision**, an original clean-room research and systems project for structural Mixture-of-Experts compression without any additional quantization.
- The target adapter is `deepseek-ai/DeepSeek-V4-Flash-0731`; the core must remain model-agnostic and must be testable with generated micro-checkpoints.
- The work and Git history belong to Omar Ibrahim (`omar07ibrahim`). Configure commits as `Omar Ibrahim <31526072+omar07ibrahim@users.noreply.github.com>`.
- Check the working name across GitHub, package registries, and obvious trademarks before the first public push. If it is clear, create the public repository `omar07ibrahim/stratafold`; otherwise document the collision and choose a distinct name before publishing.
- Keep `main` releasable. Push reviewed, green, atomic commits. Never force-push or rewrite published history.

## Non-quantization contract

- The released target checkpoint is already mixed precision: `expert_dtype=fp4`, an FP8 E4M3 quantization configuration for other paths, and BF16/F32 metadata. Treat that native representation as the baseline.
- **Never claim the target is unquantized.** The exact phrase is: `structural compression without additional quantization`.
- Do not change tensor bit width or dtype to create a compression result. Preserve each source tensor's native dtype unless an explicitly labeled equal-dtype synthetic experiment uses BF16/F32 throughout.
- Structural methods may remove experts, merge compatible experts, share/factor bases, allocate ranks, preserve sparse residuals, prune structures, or distill a smaller student.
- Report logical parameter count, serialized bytes, steady-state RAM/HBM, KV/state bytes, active parameters/FLOPs, latency, throughput, and quality as separate quantities. Streaming/offload and lossless archive compression are useful controls, but are not parameter compression.
- Every public number must be tagged `measured`, `source-reproduced`, `derived`, `projected`, or `unverified`. No benchmark claim without committed raw machine-readable evidence and a reproducible command.

## Exact target reality

- Pin the official Hugging Face revision before using metadata. The bootstrap research snapshot is commit `7872f01b1d1fe23eabc4c98b48bffcef5a386062` from 2026-08-01; re-check and record any movement rather than silently changing the baseline.
- The published artifact has 48 weight shards and is approximately 166.9 GB. The model page reports about 304B parameters. The config has 43 hidden layers, hidden size 4096, 256 routed plus one shared expert, top-6 routing, three hash layers, one next-token-prediction layer, and DSpark target layer IDs `[40, 41, 42]`.
- CSA/HCA reduce KV/state and attention work; sparse routing reduces activated compute; DSpark targets decoding speed. None of these alone reduces the stored expert weights.
- Schema compatibility, metadata inspection, a scaled synthetic topology, and a full-checkpoint experiment are four different evidence levels. State the achieved level explicitly.

## Clean-room and attribution

- Treat `https://github.com/FareedKhan-dev/kimi-k3-in-c` only as credited product/validation prior art. Do not fork, clone into the project, rename, copy, translate, or lightly paraphrase its code, prose, file layout, fixtures, diagrams, assets, branding, or benchmark claims.
- Independently derive the implementation from primary papers and official model specifications. Maintain `PROVENANCE.yaml`, `CITATIONS.bib`, `THIRD_PARTY_NOTICES.md`, `docs/PRIOR_ART.md`, and a clean-room ADR.
- Preserve all required notices for any intentionally reused dependency. The default is no source-code reuse from research repositories; reimplement from papers/specifications and document the boundary.
- REAP is Apache-2.0; REAM and TD-MoE are MIT; D2-MoE is Apache-2.0 at this snapshot. The MoBE repository appeared to lack a standard license, so do not copy from it without verified permission.
- Do not imply endorsement or affiliation with DeepSeek, Hugging Face, Moonshot, or any paper authors. Synthetic weights must be locally generated and not derived from DeepSeek weights.

## Engineering and research bar

- Implement a deterministic, exact-topology-but-scaled `dsv4-0731-micro` fixture and separate algebraic-low-rank, structured/trained, and adversarial-full-rank fixtures.
- Maintain a simple independent Python oracle and a safe Rust runtime/format reader. The optimized path must never need to reconstruct all full expert matrices in memory.
- Baselines must include equal-dtype dense storage, independent per-expert SVD, uniform whole-expert pruning, REAP-style saliency pruning, REAM-style merging, and a storage-only lossless control. Add methods only when the comparison remains fair and reproducible.
- The original research hypothesis is **Route-Stratified Factorization**: functional output sketches plus routing/co-routing constraints, shared bases within compatible strata, global byte/error rank allocation, and output-sensitive block-sparse residual repair. Present it as a falsifiable hypothesis, never as SOTA before evidence.
- Calibration must span agentic/tool use, code, math, multilingual/general text, and long-context patterns when real data is used. Report domain composition and sensitivity; never hide calibration collapse behind an aggregate.
- The planner must be allowed to return `INFEASIBLE` or preserve a layer unchanged. It must not force compression when the quality budget is unsupported.
- Add property tests, malformed-input tests, deterministic fixtures, differential tests, license/secret checks, and CI. Generate every chart from committed JSON evidence.

## Host and download safety

- This AWS host has about 4 vCPU, 30 GiB RAM, and only about 2.6 GiB free at bootstrap. Re-check before every dependency install, build, cache, or remote fetch.
- Maintain at least `2 GiB` free on `/`. Keep the complete workspace, build cache, and generated artifacts under `750 MiB`; keep download/cache data under `128 MiB`.
- Fetch only pinned small metadata after a HEAD/API size check. Refuse any single remote object over `32 MiB`, any aggregate fetch over `128 MiB`, all `.safetensors`, model `.bin`/weight blobs, Xet/LFS shard redirects, and any unbounded recursive download.
- Never download the 166.9-GB target checkpoint on this host. Never install PyTorch, CUDA, vLLM, SGLang, Docker images, or multi-gigabyte toolchains here. Do not purchase cloud resources or change AWS/IAM/firewall/network settings.
- A full V4 compression campaign must be a dry-run manifest plus a hardware-gated script and written runbook for a suitable multi-GPU machine. Do not pretend it ran here.
- Do not alter `agent-01`, unrelated services, other workspaces, credentials, or user files. Never print or commit secrets. Localhost-only test services are allowed; public deployment needs a separate instruction.

## Persistent execution loop

- The initial instruction explicitly requests a persistent Goal for the whole mission. Create it immediately without a token budget and keep it active until the acceptance gates are genuinely complete.
- Work in vertical milestones. For each: define the gate, implement, test, measure, run a bounded independent subagent review, fix findings, update docs/claims/backlog, commit, and push.
- Do not stop at research notes, a plan, a scaffold, or a polished README. Build the runnable micro path first, then improve the highest-impact defensible unfinished work.
- Keep cosmetic changes subordinate to correctness and evidence. Once a release gate is met, continue only with meaningful measurable improvements or well-scoped adapters.
- If suitable GPU hardware or licensed data becomes essential, preserve a complete runnable CPU/micro result, create a reproducible campaign plan, keep the Goal active, and report the exact human-only blocker rather than fabricating results.
