# Changelog

## 0.3.0 — 2026-07-31

- Require `count_split>=1.0.0` for full graph/count-split/DE installations.
- Adopt the corrected simultaneous multi-fold molecule allocation, integer
  dtype preservation, sparse canonicalization, and deterministic seed contract
  delivered by `count_split` 1.0.0.
- Retain the existing cells × genes adapter, exact conservation checks, and
  public `split_counts` interface.
