# StrataFold

> Fold experts, not precision.

StrataFold is an independent clean-room laboratory for **structural Mixture-of-Experts compression without additional quantization**. It investigates whether routing-aware shared structure can reduce stored expert weights while preserving each source tensor's native dtype.

The primary adapter targets `deepseek-ai/DeepSeek-V4-Flash-0731`. That released checkpoint is already mixed precision: its experts use a native FP4 representation and other paths include an FP8 E4M3 configuration. StrataFold uses that representation as the target baseline; it does not call the release unquantized and does not count a dtype or bit-width change as compression.

## Evidence state

The repository is under active milestone development. Current evidence levels are intentionally separate:

- `[measured]` the dependency-free resource doctor and its request-policy unit tests run on this host;
- `[unverified]` target topology and artifact facts remain bootstrap inputs until the pinned official metadata snapshot and hashes pass M1;
- `[not-run]` no DeepSeek weight shard has been downloaded, opened, compressed, or served here;
- `[not-yet-claimed]` Route-Stratified Factorization is a falsifiable research hypothesis, not a performance or SOTA claim.

All quantitative claims must appear in [`CLAIMS.json`](CLAIMS.json) with one of `measured`, `source-reproduced`, `derived`, `projected`, or `unverified`, plus a reproducible evidence reference where applicable.

## Safety-first start

The doctor is standard-library-only, so it can run before an install, build, cache fill, or metadata fetch:

```bash
make doctor
make test
```

It enforces at least 2 GiB free on `/`, a 750 MiB complete-workspace ceiling, a 128 MiB cache/fetch ceiling, a 32 MiB per-object ceiling, exact revision pins, and metadata-only URLs. Weight blobs, `.safetensors`, model `.bin` files, and LFS/Xet delivery paths fail closed before an opener is called.

## Scope and honesty

This small CPU host supports generated micro-checkpoints and metadata-only schema validation. It is not a full-model machine. Full-checkpoint work is limited to a hardware-gated dry-run manifest and runbook; the approximately 166.9 GB target artifact will not be downloaded here. `[unverified bootstrap]`

See the [benchmark contract](docs/BENCHMARK_CONTRACT.md), [clean-room decision](docs/adr/0001-clean-room.md), [threat model](docs/THREAT_MODEL.md), and [prior-art boundary](docs/PRIOR_ART.md) before interpreting results.

StrataFold is not affiliated with or endorsed by DeepSeek, Hugging Face, Moonshot AI, or any cited researcher or institution.

## License

Code is licensed under Apache-2.0. Documentation and original project artwork are licensed under CC BY 4.0; see [`LICENSE-DOCS`](LICENSE-DOCS) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
