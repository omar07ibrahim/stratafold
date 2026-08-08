# DeepSeek-V4-Flash-0731 target ledger

## Pinned status

- Official repository: `deepseek-ai/DeepSeek-V4-Flash-0731`
- Exact revision: `7872f01b1d1fe23eabc4c98b48bffcef5a386062`
- M1 status: `validated`
- Evidence level: `pinned-official-metadata`
- Original bounded metadata retrieval window: `2026-08-06T12:24:23Z` to `2026-08-06T12:33:44Z`
- Post-capture repository projection verification: `2026-08-08T15:49:13Z`
- Full checkpoint: **NOT DOWNLOADED / NOT RUN**

The committed [manifest](../metadata/deepseek-v4-flash-0731/7872f01b1d1fe23eabc4c98b48bffcef5a386062/manifest.json) pins five bounded metadata files. The [raw M1 JSON](../evidence/raw/m1_target_genome.json), [CLI transcript](../evidence/raw/m1_target_genome.txt), and [repository projection receipt](../metadata/deepseek-v4-flash-0731/7872f01b1d1fe23eabc4c98b48bffcef5a386062/repository.receipt.json) are the review surfaces. Reproduce their semantics and identities with:

```bash
make evidence-m1-check
make evidence-m1-verify
```

## Source-reproduced declarations

The pinned `config.json` declares:

- `expert_dtype=fp4`;
- `torch_dtype=bfloat16`;
- `quantization_config.quant_method=fp8` and `fmt=e4m3`;
- 43 hidden layers and hidden size 4096;
- 256 routed experts and one shared expert per layer;
- top-6 routing;
- three hash layers;
- one next-token-prediction layer;
- DSpark target layer IDs 40, 41, and 42.

These are configuration declarations only. No shard header, shard payload, tensor value, or runtime activation was inspected.

## Source-reproduced index and API projection

The pinned safetensors index lists 72,317 tensor names across 48 shard filenames and declares 166,878,536,440 tensor payload bytes.

The repository API projection, reconstructed offline from its committed post-capture receipt, reports:

| Artifact ledger | Bytes |
| --- | ---: |
| Weight-shard files | 166,886,535,336 |
| Non-weight files | 12,125,738 |
| All repository files | 166,898,661,074 |

Its API-reported parameter classes are:

| Storage class | Parameters |
| --- | ---: |
| BF16 | 1,483,567,488 |
| F32 | 37,741,630 |
| F8_E4M3 | 6,304,038,912 |
| I64 | 2,327,040 |
| I8 | 296,352,743,424 |
| **Total** | **304,180,418,494** |

The parameter classes are API-reported metadata reproduced from the receipt; they were not recomputed from shard headers.

## Derived ledger

`166,886,535,336 - 166,878,536,440 = 7,998,896` bytes.

This is an unattributed difference between two metadata ledgers. It is not measured compression, proven container overhead, padding, or any other causal attribution.

## Safety observation and limits

For the validated decoder invocation and committed snapshot, the decoder found zero weight-shard files, opened zero weight-shard files, executed no target code, and performed no full-checkpoint operation. This measured statement is limited to that invocation and snapshot.

The capture-time safety statement remains `unverified`, and the hostwide download state remains `not_audited`. Neither is promoted into a hostwide claim.

## Explicit non-claims

M1 does not establish:

- shard-header or payload dtypes beyond the configuration declarations;
- a compression ratio, quality result, throughput result, active parameter count, or decoding-speed result;
- that the 7,998,896-byte difference is compression or container overhead;
- parameter counts independently recomputed from checkpoint contents;
- provenance for the uncommitted full API response body;
- performance leadership or a state-of-the-art result.

The earlier report's 284B/13B-activated preview/core is not interchangeable with the final 0731 metadata snapshot. CSA/HCA, sparse routing, and DSpark may alter execution behavior, but M1 makes no active-compute or performance claim from those declarations.
