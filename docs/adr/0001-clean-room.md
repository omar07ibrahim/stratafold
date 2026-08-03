# ADR 0001: Clean-room implementation boundary

- Status: accepted
- Date: 2026-08-03
- Decision owner: Omar Ibrahim

## Context

StrataFold must test structural MoE compression ideas while avoiding accidental source, prose, fixture, or claim inheritance from research implementations and product prior art. Several useful papers have official code under different licenses; at least one referenced repository has no verified standard license. The primary checkpoint is too large and is forbidden on this host.

## Decision

The implementation is derived independently from mathematical descriptions in primary papers and official model specifications. Research repositories may be inspected only to verify authorship, release status, license, and high-level experimental protocol. Their code is not copied, translated, adapted, vendored, executed as a dependency, or used as a file-layout template.

`FareedKhan-dev/kimi-k3-in-c` is isolated as product-discipline prior art. Read-only inspection may identify abstract lessons such as making resource checks visible, offering an offline demo, separating validation levels, and publishing a resource ledger. No code, prose, layout, fixture, diagram, asset, branding, or benchmark claim crosses the boundary.

Only official, revision-pinned, small metadata may be snapshotted for the target adapter. Synthetic checkpoint weights are generated locally from documented seeds and never from target weights. The core remains model-agnostic.

## Verification controls

- `PROVENANCE.yaml` records every studied source and its permitted use.
- `THIRD_PARTY_NOTICES.md` distinguishes citations from incorporated components.
- CI scans for unexpected large/binary weight files, common secret patterns, and forbidden target claims.
- Every implementation PR states whether source code was consulted; the default answer is no.
- Paper equations are re-derived in design notes and tested against an independent Python oracle.

## Consequences

The project may take longer than adapting an existing repository, but its design history and evidence remain attributable. A reference without verified permission can inform comparison questions but cannot contribute implementation material.
