# Tahoe Analysis Hardening Plan

**Status:** Proposed implementation plan

**Branch:** `feature/tahoe-hardening`

**Worktree:** `.worktrees/tahoe-hardening`

**Reference analysis:** `../bfx-luad-scrnaseq-GSE131907`

**Request source:** `.tmp/sc_robust_dev_requests.md` in the primary worktree

## Objective

Make the supported `sc_robust` path safe and auditable for the Tahoe-100M
pilot without moving Tahoe-specific storage, scheduling, or scientific policy
into this package. The package must provide conserved count splits, strict
pseudobulk boundaries, reconstructable DE designs and contrasts, structured
fit diagnostics, reusable artifact checks, and deterministic Leiden calls.

Production DE is additionally gated on recovering and validating the intended
sparse-regime PyDESeq2 behavior. The dirty sibling `../PyDESeq2` checkout is an
audit input, not a dependency or a source from which code may be copied.

## Audit Summary

The existing package already contains most of the required computational
path. The hardening work is concentrated at these boundaries:

- `sc_robust/sc_robust.py`: count splitting, numba seed bridging, and robust
  run provenance.
- `sc_robust/de/pseudobulk.py` and
  `sc_robust/process_de_test_split.py`: graph partitioning, aggregation, source
  membership, and pseudobulk metadata.
- `sc_robust/de/differential_expression.py`: PyDESeq2 dataset construction,
  fitting stages, coefficient lookup, and contrast execution.
- `sc_robust/de/base.py` and `sc_robust/de/workflow.py`: public result models,
  orchestration, provenance, and CPU behavior.
- `sc_robust/utils.py` and `sc_robust/gene_modules.py`: Leiden invocation and
  higher-level seed propagation.
- `sc_robust/qc.py`: existing metric, threshold, classification, and plotting
  seams; streaming generalization is deferred.
- `setup.py`, `requirements.txt`, and `.github/workflows/ci.yml`: supported
  PyDESeq2 pin and mandatory DE test environment.

The LUAD repository demonstrates the intended calling sequence and useful
artifact guards, but currently compensates for package gaps with local Leiden
helpers, joblib monkeypatches, repeated ID/order checks, and script-level JSON
summaries. Those patterns should become package APIs where they are generic.

Baseline on the feature worktree is `38 passed, 12 warnings`. PyDESeq2 is not
installed in the current test environment, so that baseline does not validate
the DE fit path. CI must not silently omit the new P0 DE tests.

## Public Contract Decisions

### Matrix orientation

Every public package count API accepts cells x genes. The adapter around
`count_split.count_split.multi_split` is responsible for the genes x cells
transpose required by the dependency. Public docstrings and errors must state
the orientation. Internal helper names should include the expected orientation
when ambiguity remains.

### Annotation and partition labels

`cluster_labels` describes the analytical grouping used by `within_cluster`.
`sample_labels` remains descriptive metadata and must be documented as such.
A new `partition_labels` argument declares a hard boundary. The implementation
will partition each boundary group independently, remap source indices to the
global input, and verify after construction that no pseudobulk mixes groups.
Filtering cross-group edges alone is insufficient because a graph partitioner
can reuse one partition across disconnected components.

`group_labels` may be accepted as a compatibility alias only if one canonical
name is selected before release. Supplying both names must fail.

### DE design and contrasts

The canonical DE entry point will accept an explicit formula, for example
`design="~ 0 + condition"`, plus `annotation_columns` that are retained for
reporting but never silently added to the model. Formula terms determine the
design matrix. Existing `design_columns` remains a supported shorthand for a
no-intercept additive design. Existing `metadata_columns` is deprecated in
favor of `annotation_columns`; its legacy annotation-only behavior is retained
during the deprecation window.

Coefficient identity must not be inferred from ad hoc prefixes. Dataset
preparation records the mapping from original categorical labels to sanitized
coefficient names, formula, design columns, matrix rank, and reference level.
`run_pairwise_de(..., pairs=[(numerator, denominator)])` is canonical;
`cluster_pairs` remains a mutually exclusive compatibility alias. Only the
requested pairs are evaluated.

### CPU behavior

Fit-level and contrast-level parallelism will be separate, named settings.
Defaults will avoid nested use of every host CPU. Resolved worker counts are
stored in provenance, and a request for parallelism at both levels emits an
explicit warning or fails under strict policy. The restricted-joblib backend
workaround remains separate from numerical fallback behavior.

### Structured results

Compatibility findings, provenance, and diagnostics use JSON-safe dataclasses
or typed records with stable `to_dict()`/`to_json()` methods. DataFrame export
is provided for naturally tabular findings and per-contrast diagnostics.
Fitted PyDESeq2 objects may remain attached for interactive work, but a batch
runner must not need to pickle or inspect them to audit a result.

## Requester Decisions Needed

The following answers should be recorded before the affected workstream is
considered production-ready. Only the first two block initial implementation;
the others have safe provisional defaults but affect scientific policy or
backward compatibility.

1. What exact PyDESeq2 artifact produced the successful LUAD sparse fits: an
   environment lock, wheel, container digest, fork commit, or archived source?
   Please also identify one known successful sparse fit that can become a
   regression fixture or numerical reference.
2. What was the intended statistical behavior of the local ridge changes, and
   who will approve the fallback trigger, penalty, terminal-failure policy, and
   numerical tolerances? The current dirty checkout is not executable evidence
   of that behavior.
3. What is Tahoe's canonical hard pseudobulk boundary key? Confirm whether it
   is well, sample, or a composite key, and specify what to do when a boundary
   has too few cells for the requested pseudobulk size. The package will never
   merge undersized groups across a declared boundary.
4. Confirm the initial Tahoe formula and condition encoding. The proposed
   default is `~ 0 + condition`, with drug-dose labels as numerators and the
   same-plate DMSO label as denominator; Tahoe orchestration remains
   responsible for eligible pair selection.
5. Which cell identifier is canonical across counts, metadata, graph, and
   split artifacts? Stable source hashes require a unique, immutable ID rather
   than row positions when artifacts may be reconstructed independently.
6. Should rank deficiency, zero libraries, seed-bridge failure, and mixed
   pseudobulk boundaries always be hard failures in production? The proposed
   strict default is error, with explicit warning modes only for backward
   compatibility.
7. May CPU defaults change from implicit all-host-CPUs behavior to one worker,
   with explicit fit-level and contrast-level overrides? This is safer for
   batch scheduling but is a visible behavior change.
8. What minimal LUAD input and expected outputs may be committed or generated
   for the compatibility smoke test? If no redistributable fixture exists, the
   requester should provide an immutable artifact location and checksum.

## Workstreams

### 0. Recover the supported PyDESeq2 inference path

This is the first production gate and should be isolated from normal package
API changes.

1. Locate the exact PyDESeq2 artifact used by the successful LUAD environment:
   lock files, environment exports, wheel caches, image metadata, commits, or
   archived result provenance. Record hashes and version information.
2. Reproduce the ordinary fit and the two intended sparse/singular failures in
   a clean environment. Preserve the failing fixtures before changing code.
3. Reconstruct the intended IRLS and dispersion fallback behavior as a
   reviewed inference implementation or upstream/fork patch. Do not use the
   dirty sibling diff directly; it currently has an unresolved `la` reference
   and passes an unsupported `ridge_penalty` argument.
4. Emit per-gene convergence, fallback activation, and terminal failure as
   machine-readable records rather than stdout messages.
5. Compare coefficients, standard errors, dispersions, Cook's distances, and
   null behavior against stock PyDESeq2 on well-conditioned fixtures.
6. Pin an immutable reviewed release or fork commit in package metadata, LUAD,
   and CI. No production import may resolve through the sibling checkout.

The result of this workstream is a short decision record identifying the
supported artifact, fallback trigger and mathematics, numerical tolerances,
and ownership strategy. If the historical artifact cannot be recovered, the
replacement must be validated as new behavior rather than presented as an
exact reconstruction.

### 1. Clarify the DE design and add diagnostics

Refactor `prepare_deseq_dataset` around an explicit design specification while
preserving its current shorthand. Validate required metadata columns, category
levels, coefficient uniqueness after sanitization, full-rank design, zero
libraries, and resolved worker counts before fitting.

Add stable fit and contrast identifiers derived from declared inputs rather
than Python object identity. Extend `DEAnalysisResult` with JSON-safe design,
contrast, provenance, and diagnostic records. Each contrast records original
labels, coefficient labels, vector, numerator, denominator, reference,
direction, status, and error.

Capture fit-stage diagnostics around the existing explicit PyDESeq2 sequence:

- input, filtered, and modeled gene counts;
- observations per condition and library-size summaries;
- formula, design shape, rank, condition number, and aliased columns;
- size-factor and dispersion summaries;
- convergence, fallback, and terminal failure counts;
- Cook's outlier/refit counts and independent-filter threshold;
- non-finite coefficient, standard error, statistic, and p-value counts; and
- fit-level and contrast-level status/error fields.

Failed and partially filtered fits must still return or raise with a complete
diagnostic record. Define one documented exception that carries the record for
fail-fast callers; orchestration may convert it into a failed result.

### 2. Add artifact compatibility and provenance APIs

Introduce one public validator, tentatively `validate_compatibility`, that
accepts the available counts, metadata, graph, split, and identifier records.
It returns a `CompatibilityReport` containing deterministic findings with
severity, field, expected value, observed value, and remediation. A caller
policy selects `report`, `warn`, or `error`; policy never changes what was
checked.

Checks include:

- matrix and graph shapes;
- exact ordered cell and gene IDs, including shape-equal reorder failures;
- counts-to-metadata index alignment;
- deterministic ID and artifact hashes;
- split proportions and exact count conservation when split matrices exist;
- pseudobulk source hashes and boundary labels; and
- relevant algorithm parameters and dependency versions.

Lift the generic behavior of LUAD's guards and SHA-256 manifest helpers, not
its file layout. Reuse the same hash implementation in robust, pseudobulk, and
DE provenance. Extend `PseudobulkResult` and `DEAnalysisResult` with consistent
JSON-safe provenance; do not introduce a count-split result hierarchy.

### 3. Enforce pseudobulk boundaries and source provenance

Add `partition_labels`, optional `cell_ids`, and
`retain_source_cells: bool = True` to `build_pseudobulk`. Existing calls retain
their output shape and full `source_cells` lists by default.

When a boundary is supplied:

1. Validate graph, count, cluster, sample, boundary, and cell-ID lengths.
2. Process boundary groups in stable declared/input order.
3. Run the existing topology or within-cluster builder independently per
   group; do not fork a second aggregation algorithm.
4. Remap local source positions to global input positions.
5. Generate IDs from a stable boundary key plus local pseudobulk ordinal.
6. Hard-fail if any output contains more than one boundary value.

Metadata always includes `source_cell_count`, a versioned hash of sorted stable
cell IDs (or global integer positions when IDs are unavailable), mode, seed,
boundary key, and whether source lists were retained. Document that hash
membership is order-insensitive while pseudobulk IDs are stable only for the
declared group and cell ordering.

No graph-free aggregation API is included in this delivery unless the Tahoe
pilot demonstrates that independent per-well graph partitioning is unsuitable.

### 4. Harden the existing count-split adapter

Add a thin cells x genes public adapter that delegates to `multi_split` and is
also used by `robust.do_splits`. It validates:

- dense or sparse two-dimensional counts;
- finite, nonnegative, integral values without silently coercing them;
- finite nonnegative proportions with the documented sum policy;
- output count, shape, and sparse/dense expectations; and
- exact element-wise conservation using sparse differences where possible.

The adapter exposes a seed-bridge failure policy. A requested deterministic
run defaults to an error if the numba RNG bridge cannot be seeded; `warn` is an
explicit opt-in compatibility mode. Errors identify an affected cell/gene or
provide a compact sparse mismatch summary.

Document and test the actual guarantee: same seed plus same ordered matrix
repeats. `multi_split` is order-dependent, so shard/order invariance remains a
Tahoe adapter responsibility and is not claimed by `sc_robust`.

### 5. Expose seeded Leiden consistently

Add one canonical `random_state` parameter to
`perform_leiden_clustering`, accepting `seed` as a compatibility alias if
needed, and forward it to `leidenalg.find_partition`. Preserve the current
return tuple. Propagate the parameter through `single_graph_and_leiden` and
gene-module workflows, recording the resolved seed and package versions in
higher-level provenance.

The same graph, resolution, dependency versions, and seed must produce exactly
the same labels in a regression fixture. After release, LUAD's local helpers
and monkeypatches can be removed in a separate LUAD commit.

### 6. Keep streaming QC deferred

Do not add a speculative streaming abstraction in this branch. Document the
existing separation among metric calculation, threshold derivation,
classification, and plotting, and capture Tahoe pilot measurements that would
drive a future API. Tahoe retains source scanning, mergeable summaries,
stable-ID sampling, and threshold policy until that follow-up is approved.

## Tests and CI

Add focused test modules rather than expanding only the end-to-end fixture:

- `tests/test_count_split.py`: dense/sparse conservation, invalid counts and
  proportions, fixed-order repeatability, and seed-bridge policy.
- `tests/test_pseudobulk.py`: both existing modes, label types, strict boundary
  isolation, source hashes, optional source lists, and stable IDs.
- `tests/test_de_design.py`: categorical formula, annotation/design separation,
  coefficient mapping, selected pair orientation, and legacy aliases.
- `tests/test_de_diagnostics.py`: success, filtering, rank failure, fallback,
  terminal failure, JSON export, and stable IDs.
- `tests/test_compatibility.py`: shape, ordered-ID, hash, policy, remediation,
  and deterministic serialization checks.
- `tests/test_leiden_seed.py`: fixed-seed repeatability and propagation.
- `tests/test_sparse_inference.py`: IRLS and dispersion fallback activation plus
  ordinary-fit numerical comparisons.

Keep a fast core CI job for supported Python versions and add a required DE job
with the immutable PyDESeq2 dependency. The DE job must fail on dependency
installation or collection failure; it must not skip all DE tests. Store the
resolved dependency versions in test output or an artifact. Add one LUAD smoke
fixture or script invocation after package tests pass, using small committed or
generated data and no sibling monkeypatches.

## Delivery and Commit Sequence

Each numbered item is independently reviewed and committed; tests and docs ship
with the API they cover.

1. Commit this plan and the PyDESeq2 recovery decision record template.
2. Recover/pin PyDESeq2 behavior and land sparse inference fixtures.
3. Land the explicit DE design contract, generic `pairs`, and diagnostics.
4. Land compatibility reports and shared provenance hashing.
5. Land strict pseudobulk boundaries and source provenance.
6. Land count validation, seed policy, and conservation tests.
7. Land seeded Leiden propagation and remove package-level nondeterminism.
8. Update the tutorial/API documentation and CI, then run the LUAD smoke path.
9. Tag a Tahoe pilot candidate only after all release gates pass.

Commits should remain scoped to one workstream. The primary worktree's
unrelated `.gitignore` modification is not part of this branch.

## Release Gates

- The supported PyDESeq2 artifact is immutable, reviewed, and CI-tested.
- Sparse IRLS and dispersion fallbacks are triggered by dedicated fixtures and
  reported per gene.
- Split outputs exactly conserve counts and fixed-order reproducibility is
  explicit.
- Declared pseudobulk boundaries cannot be crossed silently.
- Formula, coefficient mapping, reference, and every contrast are
  reconstructable from serialized result metadata.
- Every fit emits diagnostics, including failed and partially filtered fits.
- Shape-equal cell or gene reorder mismatches fail compatibility validation.
- Fixed-seed Leiden repeats without a LUAD monkeypatch.
- The package suite and LUAD smoke path pass in clean environments.
- Tahoe pins released package/dependency versions and imports no sibling
  working copy.

## Explicit Non-goals

- Tahoe source decoding, canonical shard ordering, distributed split storage,
  task scheduling, retry policy, DMSO eligibility, atlas-wide correction, and
  publication schemas.
- A second count-splitting algorithm or count-split result hierarchy.
- Arbitrary numerical DE contrasts in the first one-factor Tahoe delivery.
- A production graph-free pseudobulk API before pilot evidence requires it.
- Streaming QC before measured Tahoe access patterns are available.
- Broad performance grids or simulation campaigns beyond focused release
  regressions.

## Follow-up Packaging Note

The current `setup.py` installs the monolithic `requirements.txt`, so there is
no pathway-analysis-only PyPI dependency route today. A `pathway` optional
extra and packaging metadata cleanup remain useful, but they should be handled
as a separate scoped packaging change so the Tahoe dependency pin and release
gates are not conflated with optional-install redesign.
