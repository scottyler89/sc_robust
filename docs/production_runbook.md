# sc_robust Production Handoff

This runbook defines the package contract before downstream orchestration.
It does not define project-specific storage, scheduling, QC thresholds, control
matching, or contrast eligibility.

## Install

From a tagged checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

The package requires cells x genes count matrices. The initial immutable cell IDs
are the input AnnData observation names; gene IDs are the ordered variable names.
Do not regenerate IDs from row positions after ingestion.

## Required Inputs

- Counts: two-dimensional, nonnegative, finite, integral cells x genes values.
- Metadata: one row per cell, indexed by the exact ordered cell IDs.
- Graph: cells x cells, aligned to the same ordered cell IDs.
- Boundary factors: identifiable metadata columns such as `sample`, `cluster`, or
  a joint `sample:cluster` key.
- DE condition: a categorical metadata column with an explicit formula and
  explicitly selected treatment/control pairs.

## Supported Sequence

1. Call `split_counts` and retain the returned provenance and conservation status.
2. Call `build_pseudobulk(..., partition_by=[...], cell_ids=...)`. Every declared
   joint boundary is enforced; METIS target size is advisory and may warn.
3. Call `prepare_deseq_dataset` with an explicit formula such as
   `~ 0 + condition` and annotation-only columns separately.
4. Call `fit_deseq_dataset`, then `run_pairwise_de` with only requested pairs.
5. If held-out counts are available, call `apply_pseudobulk_membership` with
   the learned result before fitting the held-out DE dataset.
6. Export result provenance and diagnostics before writing result tables.
7. Optionally call pathway helpers on exported DE tables, using
   `stat_col_by_contrast` when score columns differ.

## Required Artifacts

Persist counts/metadata identifiers or hashes, split parameters and conservation,
boundary factors and source membership hashes, formula and design matrix metadata,
contrast direction and coefficient mapping, fit diagnostics, dependency versions,
and the serialized provenance envelope. Paths alone are not artifact identity.

## Failure Policy

Treat invalid counts, ID/order mismatches, mixed boundaries, zero libraries,
rank-deficient designs, unknown contrasts, and requested seed failures as hard
errors. Treat usable per-gene fallback or non-convergence as structured diagnostics
and warnings. Never rely on stdout or a runtime monkeypatch.

## Scope Boundary

The downstream project owns source decoding, global QC, shard ordering,
distributed storage, scheduling, retries, control matching, and scientific
contrast manifests. Validate this contract with a frozen integration pilot
before production execution.
