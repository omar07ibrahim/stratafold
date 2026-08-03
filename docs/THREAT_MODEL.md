# Host, artifact, and download threat model

## Protected assets

- at least 2 GiB free space on `/`;
- unrelated workspaces, services, credentials, and user files;
- the integrity of native target dtypes and public claims;
- CI availability on a 4-vCPU, low-disk orchestration host.

## Threats

1. A Hugging Face model URL redirects through LFS/Xet to a 166.9-GB weight shard.
2. A small-looking index triggers recursive or unbounded fetching.
3. Build/dependency caches consume the small margin above the disk floor.
4. A malformed `.sfc` requests out-of-bounds allocation, integer overflow, path traversal, or checksum bypass.
5. Logs or commits expose GitHub/AWS credentials.
6. Compression accounting silently changes dtype, ignores container overhead, or conflates stored, resident, active, and archived bytes.
7. Schema drift is accepted as if the pinned target were unchanged.

## Controls

- The dependency-free doctor runs before installs, builds, cache fills, metadata fetches, and large generation.
- Remote requests require HTTPS, an allowlisted host, a 40-hex revision embedded in the URL, a recognized metadata suffix, a known content length no larger than 32 MiB, and aggregate accounting no larger than 128 MiB.
- Weight extensions and LFS/Xet paths fail before opener construction. Redirect and streaming limits receive dedicated M1 tests before network transport is enabled.
- The complete workspace/build/artifact tree must remain at or below 750 MiB; cache/download data must remain at or below 128 MiB.
- No recursive model download, `.safetensors`, model `.bin`, PyTorch, CUDA, vLLM, SGLang, Docker image, public service, cloud purchase, or infrastructure/IAM change is allowed here.
- Metadata snapshots are content-hashed and offline by default. Schema differences fail closed and require a reviewed pin update.
- The format reader uses checked arithmetic, explicit maximums, checksums, and bounded allocation. Fuzz/malformed cases precede release.
- Secret and forbidden-claim scans run in CI. Only variable names and masked authentication status may appear in diagnostics.

## Residual risk and response

HEAD responses can lie or omit lengths, redirects can change, and local free space can change between checks. The fetcher therefore revalidates the final URL and byte count while streaming, aborts on missing/invalid lengths, writes only bounded temporary data, and re-runs the doctor after completion. Any uncertainty fails closed.
