# StrataFold

> Fold experts, not precision.

StrataFold is an independent clean-room laboratory for **structural Mixture-of-Experts compression without additional quantization**. It investigates whether routing-aware shared structure can reduce stored expert weights while preserving each source tensor's native dtype. The primary adapter targets `deepseek-ai/DeepSeek-V4-Flash-0731` at the exact pinned revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

The target baseline is described from bounded official metadata. Its config declares `expert_dtype=fp4`, `torch_dtype=bfloat16`, and an FP8 `e4m3` quantization configuration. These are configuration declarations—not shard-header or payload observations. StrataFold does not call the target unquantized and does not count a dtype or bit-width change as structural compression.

## Validated M1 target genome

M1 is now a validated, pinned-official-metadata snapshot:

- **Topology declarations:** 43 hidden layers, hidden size 4096, 256 routed experts plus one shared expert per layer, top-6 routing, three hash layers, one next-token-prediction layer, and DSpark target layers 40–42.
- **Index declarations:** 72,317 tensor names, 48 listed shard filenames, and 166,878,536,440 declared tensor payload bytes.
- **Verified API projection:** 166,886,535,336 weight-shard bytes, 12,125,738 non-weight-file bytes, and 166,898,661,074 total repository-file bytes.
- **API-reported parameter classes:** 304,180,418,494 total, reproduced from the committed receipt and not recomputed from shard headers.
- **Derived difference:** 7,998,896 bytes between API-reported weight-shard bytes and index-declared tensor payload bytes. It is unattributed metadata—not measured compression or proven container overhead.

Evidence surfaces are committed directly:

- [raw M1 JSON](evidence/raw/m1_target_genome.json)
- [raw M1 CLI transcript](evidence/raw/m1_target_genome.txt)
- [provenance ledger](PROVENANCE.yaml)
- [snapshot manifest](metadata/deepseek-v4-flash-0731/7872f01b1d1fe23eabc4c98b48bffcef5a386062/manifest.json)
- [repository projection receipt](metadata/deepseek-v4-flash-0731/7872f01b1d1fe23eabc4c98b48bffcef5a386062/repository.receipt.json)
- [claim registry](CLAIMS.json)

Every quantitative claim is registered in `CLAIMS.json` with an allowed evidence tag, exact committed evidence paths, and a reproduce command.

## Setup and verification

The control plane is dependency-free and requires Python 3.10+ and `make`:

```bash
git clone https://github.com/omar07ibrahim/stratafold.git
cd stratafold
make doctor
make test
make evidence-m1-check
make evidence-m1-verify
PYTHONPATH=src python3 -m stratafold inspect-target
```

`make doctor` enforces the workspace, cache, object-size, revision, and metadata-only request boundaries before any evidence command runs. `make evidence-m1-check` reconstructs the expected target-genome surfaces in memory and compares them byte-for-byte without rewriting; `make evidence-m1-verify` validates the adopted evidence identities and semantics.

## Evidence status and safety

Evidence labels remain deliberately separate:

- `[source-reproduced]` config, topology, index, repository byte ledgers, and API-reported parameter classes were reproduced from the committed metadata and post-capture receipt;
- `[derived]` the 7,998,896-byte difference is arithmetic with no causal attribution;
- `[measured]` the validated decoder invocation found zero shard files in the snapshot, opened zero shard files, executed no target code, and performed no full-checkpoint operation;
- `[unverified]` the capture-time attestation is not an independent audit, and the hostwide download state is unaudited;
- `[not-run]` the full checkpoint is **NOT DOWNLOADED / NOT RUN** for the M1 project record.

This host supports generated micro-checkpoints and metadata-only validation. No M1 result is a compression-ratio, quality, throughput, active-compute, performance, or state-of-the-art claim.

See the [target ledger](docs/TARGET_LEDGER.md), [benchmark contract](docs/BENCHMARK_CONTRACT.md), [clean-room decision](docs/adr/0001-clean-room.md), [threat model](docs/THREAT_MODEL.md), and [prior-art boundary](docs/PRIOR_ART.md) before interpreting future results.

StrataFold is not affiliated with or endorsed by DeepSeek, Hugging Face, Moonshot AI, or any cited researcher or institution.

## License

Code is licensed under Apache-2.0. Documentation and original project artwork are licensed under CC BY 4.0; see [`LICENSE-DOCS`](LICENSE-DOCS) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
