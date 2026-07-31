# Changelog

## 0.3.1 — 2026-07-31

- Initialize the vendored PyDESeq2 `refitted` gene mask from the existing
  `replaced` mask before Cook's-distance outlier refitting. This fixes real
  sparse Tahoe contrasts that contain at least one replaceable outlier.
- Normalize the inherited outlier-refit size-factor vector across AnnData's
  one- and two-dimensional `obsm` representations.
- Initialize the normalized refit mask before copying refitted gene estimates
  back into the parent dataset.
- Add a full synthetic DE fit that forces the Cook's outlier-refit path.

## 0.3.0 — 2026-07-31

- Require `count_split>=1.0.1` for full graph/count-split/DE installations.
- Adopt the corrected simultaneous multi-fold molecule allocation, integer
  dtype preservation, sparse canonicalization, and deterministic seed contract
  delivered by the `count_split` 1.0 release series.
- Retain the existing cells × genes adapter, exact conservation checks, and
  public `split_counts` interface.
- Restore Python 3.10 and 3.11 parser compatibility in the vendored PyDESeq2
  outlier-refit status message.
