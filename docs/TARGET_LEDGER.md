# DeepSeek-V4-Flash-0731 target ledger

- Official repository: `deepseek-ai/DeepSeek-V4-Flash-0731`
- Bootstrap revision: `7872f01b1d1fe23eabc4c98b48bffcef5a386062`
- Ledger status: `unverified` until the M1 bounded snapshot is hashed and reviewed
- Retrieval date: pending

## Bootstrap facts—not yet release claims

The mission supplies the following inputs for verification: 48 weight shards; approximately 166,898,661,074 artifact bytes; approximately 166,878,536,440 indexed tensor payload bytes; roughly 304B parameters on the final model page; 43 hidden layers; hidden size 4096; 256 routed and one shared expert; top-6 routing; three hash layers; one next-token-prediction layer; and DSpark target layer IDs 40, 41, and 42. Every value in this paragraph is tagged `unverified` until reproduced from a pinned official source.

The earlier technical report's 284B/13B-activated preview/core is not interchangeable with the final 0731 artifact. CSA/HCA, sparse routing, and DSpark may change state, active work, or decoding speed; none alone removes stored expert tensors.

## Native representation contract

The released artifact is expected to declare `expert_dtype=fp4`, an FP8 E4M3 quantization configuration for other paths, and BF16/F32 metadata. StrataFold preserves native tensor dtypes and describes any result only as structural compression without additional quantization. It never treats the target as unquantized.

## Evidence ladder

1. schema compatibility;
2. pinned official metadata inspection;
3. exact-topology scaled synthetic validation;
4. full-checkpoint experiment on separately authorized hardware.

Only levels 1–3 are in scope on this host. The target weights are `NOT DOWNLOADED / NOT RUN`.
