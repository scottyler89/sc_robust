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
sparse-regime PyDESeq2 behavior from the inline implementation and repository
history in `../PyDESeq2`. That checkout is the implementation source to audit,
but production must pin reviewed, committed code rather than import a dirty
sibling working tree.

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

`cluster_labels` describes the analytical grouping used by the legacy
`within_cluster` mode. `sample_labels` remains descriptive metadata and must be
documented as such. The canonical hard-boundary API will instead use metadata
factor names: `partition_by="sample"`, `partition_by="cluster"`, or
`partition_by=["sample", "cluster"]`. Multiple factors form an exact joint key.
The implementation partitions each key independently, remaps source indices to
the global input, and verifies after construction that no pseudobulk mixes
keys. Filtering cross-group edges alone is insufficient because a graph
partitioner can reuse one partition across disconnected components.

The low-level array API may accept a named mapping of factor arrays when a
metadata frame is unavailable. Anonymous `partition_labels` is retained only
as a compatibility adapter; new provenance must always record factor names and
levels. Partition factors remain available as pseudobulk annotations and enter
the DE design only when explicitly named in the formula.

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
When no worker count is supplied, the fit uses half of the CPUs visible to the
process or scheduler, rounded down with a minimum of one. Only one layer is
parallel by default; contrast-level parallelism remains one unless explicitly
requested. Resolved worker counts are stored in provenance, and nested
parallel requests emit an explicit warning or fail under strict policy. The
restricted-joblib backend workaround remains separate from numerical fallback
behavior.

### Structured results

Compatibility findings, provenance, and diagnostics use JSON-safe dataclasses
or typed records with stable `to_dict()`/`to_json()` methods. DataFrame export
is provided for naturally tabular findings and per-contrast diagnostics.
Fitted PyDESeq2 objects may remain attached for interactive work, but a batch
runner must not need to pickle or inspect them to audit a result.

## Clarified Decisions

The requester clarified:

- The PyDESeq2 behavior is inline software logic, not a serialized fit
  artifact. Audit the local repository history and current implementation.
  Approved behavior includes both safeguards: the committed IRLS `1e-6`
  diagonal ridge and the Cox-Reid dispersion retry with `1e-6 * I` after a
  singular Fisher-information inversion. Both remain narrow numerical
  protections for poor fits rather than a new general regularized estimator.
- Pseudobulk boundaries are user-selectable metadata factors, supporting keys
  such as sample, cluster, or the exact joint sample-and-cluster key. Those
  factors must remain identifiable in output metadata and DE provenance.
- Canonical cell IDs are the AnnData observation IDs captured when data are
  first loaded. Validate uniqueness at ingestion and propagate those IDs
  unchanged through split, graph, pseudobulk, and DE artifacts.
- Validation should be strict by default for invariant-breaking conditions;
  failures and warning-only numerical outcomes are enumerated below rather than
  controlled by one ambiguous global strictness flag.
- Default fit parallelism uses half of the scheduler-visible CPUs. Explicit
  overrides remain available, resolved counts are recorded, and only one layer
  is parallel by default.
- LUAD is a code template, not a dataset re-analysis target. No LUAD data
  fixture, expected-result recreation, or re-analysis smoke run is required.

- `cells_per_pb` is an advisory METIS target, not a drop threshold. METIS may
  produce larger groups when exact target sizes are unavailable. Preserve every
  cell, never merge across a declared boundary, and emit a compact size warning
  with the boundary key and observed range instead of dropping or failing.
- Tahoe's final formula and condition encoding are deferred until data ingest
  stabilizes. Package design tests use a generic one-factor condition fixture
  without asserting Tahoe-specific scientific policy.
- Both the existing IRLS epsilon and the uncommitted dispersion retry are in
  scope for validation, diagnostics, and immutable versioning.

## Implementation Tracker

The table below is the single source of truth for milestone status. Detailed
sections define scope and evidence requirements but do not maintain a second
status value.

Status values are `not started`, `in progress`, `blocked`, `done`, or `deferred`.

| ID | Priority | Status | Depends on | Deliverable | Completion evidence |
| --- | --- | --- | --- | --- | --- |
| M0 | P0 | done | none | Provenance, SSoT, compatibility foundation | commit and M0 validation output |
| M1 | P0 | in progress | none | Reviewed PyDESeq2 safeguards | immutable commit and numerical report |
| M2 | P0 | in progress | M0, M1 | Explicit DE design and diagnostics | commit and M2 validation output |
| M3 | P1 | done | M0 | Metadata-driven pseudobulk boundaries | commit and M3 validation output |
| M4 | P1 | done | M0 | Validated count-split adapter | commit and M4 validation output |
| M5 | P1 | done | M0 | Deterministic Leiden propagation | commit and M5 validation output |
| M6 | P0 release gate | in progress | M0-M5 | Integrated API, docs, CI, and compatibility review | release-candidate commit and CI URL |
| D1 | P2 | deferred | M6 and Tahoe pilot | Streaming-friendly QC proposal | approved follow-up plan |

Execution follows dependency order first and priority second. M0 and M1 may run
in parallel; among otherwise ready milestones, P0 precedes P1. M6 remains last
because it integrates every preceding milestone despite being a P0 release gate.

Tracking rules:

- Change current status only in this table; append each transition as an event
  in the progress log.
- Mark a milestone `done` only after its acceptance criteria and validation pass.
- Record commit hashes and validation evidence in the progress log immediately.
- Use one scoped implementation commit, or a short reviewable series, per milestone.

### M0 Establish provenance, SSoT, and compatibility foundations (P0)

**Depends on:** none

Establish one provenance vocabulary and compatibility layer before result APIs
begin recording additional fields.

Implement one public provenance module that owns canonical JSON conversion,
versioned hashing, dependency capture, stable fit/contrast IDs, immutable input
snapshots, and compatibility findings. Replace private hash implementations and
result-specific provenance conventions with calls into that module.

Current audit findings: `_hash_strings` is private to `robust`; identifier
hashing silently becomes `None` on failure; `_write_json(default=str)` can hide
non-JSON values; timestamps are mixed with identity data; and mutable
`parameters` mappings duplicate fields without a schema or ownership rule.

#### SSoT ownership

| Concern | Authoritative source | Result snapshot rule |
| --- | --- | --- |
| Cell IDs | unique `adata.obs_names` captured at first load | ordered IDs or versioned ordered hash; never regenerated from row number |
| Gene IDs | ordered `adata.var_names` at the count boundary | ordered IDs or versioned ordered hash |
| Matrix orientation | public API contract: cells x genes | record shape and orientation; transpose only inside the count-split adapter |
| Run configuration | validated explicit function arguments | store normalized values once in `algorithm` provenance |
| Partition factors | named columns in aligned cell metadata | store names, ordered levels, and exact joint-key rule |
| DE design | validated formula/design specification | store formula, terms, matrix columns, rank, coefficient map, and reference |
| Contrasts | explicit contrast specification | store numerator, denominator, vector, coefficient names, and direction |
| Provenance schema | exported schema constants and typed records in the provenance module | no result-specific ad hoc schema versions |
| Supported dependency constraints | package build metadata backed by one reviewed constraint source | CI consumes the same source; projects record their exact resolution |
| Dependency versions | installed package metadata and reviewed commit IDs | capture once per run environment |
| Execution settings | resolved scheduler-visible CPU count, seed, and backend | record requested and resolved values separately |
| Artifact identity | canonical content hash plus schema version | paths are locations, never artifact identity |

Authoritative inputs remain authoritative while a computation is live. Result
provenance is an immutable snapshot sufficient to audit that computation; it
must not become a second mutable configuration object.

#### Stage lineage

- Ingestion captures immutable cell/gene axis snapshots from initial AnnData.
- Split provenance references the ingestion ID and records normalized
  proportions, seed request/result, orientation, and conservation status.
- Graph provenance references the compatible split/axis ID and records graph
  parameters, seed, dependency versions, shape, and alignment hash.
- Pseudobulk provenance references graph/count/axis IDs and records partition
  factors, source membership hashes, mode, seed, and METIS size warnings.
- DE provenance references pseudobulk/design IDs and records filters, fit
  diagnostics, execution settings, and explicit contrast IDs.
- Pathway provenance references contrast IDs and records library versions and
  the per-contrast statistic-column mapping.

Each stage stores parent stable IDs and compact snapshots, not mutable copies of
the complete parent provenance tree. Compatibility validation runs before each
expensive transition and its report ID is recorded on the child result.

Hash semantics are distinct and named:

- ordered-axis hashes detect cell/gene reordering;
- sorted-membership hashes identify pseudobulk source membership; and
- file/content checksums identify persisted artifacts independently of paths.

Compatibility fields remain views over the canonical envelope. Existing
`parameters` attributes become read-only views of `provenance.algorithm` rather
than separate mappings. `design_columns`, graph summaries, and dependency
versions are derived from their canonical design, diagnostic, or environment
records. `artifacts` stores locations plus content identities and never becomes
the source of scientific configuration.

#### Provenance contract

- Every serialized record carries a provenance `schema_version`.
- Canonical serialization accepts JSON-native values only and fails on unknown
  objects; do not use `default=str` as a silent conversion path.
- Hash records include algorithm, schema/domain tag, encoding, and whether
  ordering is significant. Shared hashing code is used everywhere.
- Stable run, fit, and contrast IDs exclude timestamps, filesystem paths, and
  mutable display labels. They derive only from canonical scientific inputs.
- `created_utc` is event metadata and is never part of a stable identity hash.
- Result records expose defensive/immutable mappings and deterministic export.

`validate_compatibility` returns one `CompatibilityReport` for counts,
metadata, graph, split, pseudobulk, and DE boundaries. Caller policy controls
reporting (`report`, `warn`, or `error`) but never changes which checks run.

Findings contain severity, stable code, field, expected value, observed value,
expected/observed hash or shape where relevant, and a remediation hint.

Required checks cover matrix/graph shapes, exact ordered cell and gene IDs,
counts-to-metadata index alignment, split proportions and conservation,
pseudobulk source hashes and partition factors, design/contrast identity, and
relevant algorithm parameters and dependency versions.

Strict defaults are scoped by failure type:

- Hard errors cover invalid counts, failed conservation, duplicate or missing
  IDs, shape/order mismatches, mixed partition keys, zero libraries,
  rank-deficient designs, unknown contrasts, and requested seed failures.
- Per-gene numerical fallback or non-convergence remains a structured warning
  and diagnostic when the fit is usable. No modeled genes or a terminal model
  stage failure is a hard fit failure.
- Version or optional provenance differences are findings whose severity
  depends on whether they invalidate reconstruction, not a global strict flag.

#### Acceptance criteria

- [x] One shared hash/serializer implementation replaces private variants.
- [x] Repeated equivalent inputs produce byte-identical canonical JSON and IDs.
- [x] Shape-equal reordered cell/gene fixtures fail with actionable findings.
- [x] All provenance records serialize without pickle or `default=str`.
- [x] Result fields reference one canonical design/config snapshot rather than
  duplicating independently mutable values.
- [x] Robust, pseudobulk, DE, and pathway results export the same versioned
  provenance envelope with stage-specific payloads.

#### Validation

```bash
python -m pytest -q tests/test_provenance.py tests/test_compatibility.py
python -m pytest -q tests/test_axis_conventions.py
```

Record the implementation commit, test output, and one redacted example JSON.

### M1 Recover the supported PyDESeq2 inference path (P0)

**Depends on:** none

This is the first production gate and should be isolated from normal package
API changes.

Audit finding: `sc_robust.fit_deseq_dataset` reaches PyDESeq2 dispersion fitting
through `fit_genewise_dispersions()` and `fit_MAP_dispersions()`, and reaches
IRLS through initial mean estimation and `fit_LFC()`. Committed PyDESeq2 IRLS
already adds a `1e-6` diagonal ridge (`780b48ec`, 2023-01-09). This committed
safeguard is approved and must remain covered. The uncommitted IRLS wrapper
duplicates that protection incompletely and will not be adopted. The approved
novel behavior is in the Cox-Reid dispersion gradient: after a singular
Fisher-information inversion, retry with `1e-6 * I`. Both safeguards require
focused regression coverage and machine-readable diagnostics.

1. Trace the fallback through the complete local PyDESeq2 history, all refs,
   repository metadata, and the current inline changes. Record the exact base
   version and every relevant code path.
2. Isolate the intended safeguard from incomplete experimental changes. The
   current working tree has an unresolved `la` reference and passes an
   unsupported `ridge_penalty` argument, so it cannot be pinned as-is.
3. Preserve and test the committed IRLS `1e-6` ridge, then formalize the
   dispersion retry with `1e-6 * I` at the singular Fisher-information inverse.
   Preserve stock behavior for well-conditioned fits, avoid a general
   regularized estimator, and report each safeguard activation separately.
4. Reproduce ordinary fits and all observed sparse/singular failure sites in a
   clean environment. Preserve the failing fixtures before changing code.
5. Emit per-gene convergence, fallback activation, and terminal failure as
   machine-readable records rather than stdout messages.
6. Compare coefficients, standard errors, dispersions, Cook's distances, and
   null behavior against stock PyDESeq2 on well-conditioned fixtures.
7. Pin an immutable reviewed release or fork commit in package metadata, LUAD,
   and CI. No production import may resolve through the sibling checkout.

The result of this milestone is a short decision record identifying the
supported commit, fallback location and epsilon, numerical tolerances, and
ownership strategy. If no historical commit contains the working behavior, the
implementation is reviewed and recorded as a new formalization of inline code.

#### Acceptance criteria

- [x] A sparse IRLS fixture exercises and reports the committed `1e-6` ridge.
- [x] A singular Cox-Reid dispersion fixture triggers exactly one `1e-6 * I`
  retry and records the affected gene.
- [x] Ordinary-fit coefficients, standard errors, dispersions, Cook's distances,
  and null behavior remain within approved tolerances.
- [x] Fallback, convergence, and terminal failure are machine-readable and are
  not available only through stdout.
- [ ] A reviewed immutable commit is pinned; production imports no dirty sibling.

#### Validation

```bash
(cd ../PyDESeq2 && python -m pytest -q tests/test_sparse_fallbacks.py)
python -m pytest -q tests/test_sparse_inference.py
python -m pytest -q tests/test_de_diagnostics.py -k fallback
```

Record the PyDESeq2 commit, dependency lock diff, numerical comparison report,
and test output.

### M2 Clarify the DE design and add diagnostics (P0)

**Depends on:** M0, M1

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

#### Acceptance criteria

- [x] A generic categorical condition fixture produces the intended explicit
  no-intercept design; Tahoe-specific encoding remains undecided.
- [x] Annotation-only columns remain outside the model and requested formula
  terms enter it.
- [x] One fit evaluates only requested `pairs`; `cluster_pairs` remains a tested
  compatibility alias.
- [x] Formula, coefficient mapping, reference, design matrix, contrast vector,
  direction, and stable IDs serialize from one canonical design snapshot.
- [x] Success, filtering, rank failure, fallback, and terminal failure emit
  stable diagnostics; resolved CPU settings are visible and non-nested.

#### Validation

```bash
python -m pytest -q tests/test_de_design.py
python -m pytest -q tests/test_de_diagnostics.py
```

Record the implementation commit, example fit/contrast JSON, and test output.

### M3 Enforce pseudobulk boundaries and source provenance (P1)

**Depends on:** M0

Add `cell_metadata`, `partition_by: str | Sequence[str] | None`, optional
`cell_ids`, and `retain_source_cells: bool = True` to `build_pseudobulk`.
Existing calls retain their output shape and full `source_cells` lists by
default. A named mapping of arrays supports low-level callers without a frame.

AnnData-facing workflows capture unique `adata.obs_names` at first loading and
propagate them as `cell_ids`. Low-level callers may use positional IDs only for
legacy compatibility, with weaker provenance recorded explicitly.

When a boundary is supplied:

1. Validate graph, counts, metadata index, labels, and cell-ID alignment.
2. Resolve every `partition_by` factor and reject missing or null key values.
3. Form exact joint keys for multiple factors in stable input order.
4. Run the existing topology or within-cluster builder independently per
   group; do not fork a second aggregation algorithm.
5. Preserve every cell; treat `cells_per_pb` as advisory and warn with the
   boundary key and observed size range when METIS cannot meet it exactly.
6. Remap local source positions and generate IDs from factor keys plus a local
   pseudobulk ordinal.
7. Hard-fail if any output contains more than one value for any boundary factor.

Metadata always includes `source_cell_count`, a versioned hash of sorted stable
cell IDs, mode, seed, factor names and values, boundary key, ID source, and
whether source lists were retained. Hash membership is order-insensitive while
pseudobulk IDs are stable only for the declared group and cell ordering.

No graph-free aggregation API is included in this delivery unless the Tahoe
pilot demonstrates that independent per-well graph partitioning is unsuitable.

#### Acceptance criteria

- [x] Existing topology and `within_cluster` calls remain backward compatible.
- [x] `partition_by` accepts one or multiple metadata factor names and no output
  crosses any exact joint key.
- [x] Every eligible input cell is assigned exactly once and retains its
  canonical initial AnnData observation ID in source provenance.
- [x] Inexact METIS target sizes preserve cells and emit one compact warning
  containing the factor key, requested target, and observed size range.
- [x] Metadata records source count/hash, factor names/values, mode, seed, ID
  source, and whether full source lists were retained.
- [x] Sparse counts and categorical, string, and integer factors are covered.

#### Validation

```bash
python -m pytest -q tests/test_pseudobulk.py
python -m pytest -q tests/test_axis_conventions.py
```

Record the implementation commit, warning example, provenance JSON, and tests.

### M4 Harden the existing count-split adapter (P1)

**Depends on:** M0

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

#### Acceptance criteria

- [x] Dense and sparse fixtures conserve every element exactly.
- [x] Negative, non-integral, non-finite, malformed, and invalid-proportion
  inputs fail before `multi_split` can coerce them.
- [x] Same seed plus the same ordered cells x genes matrix repeats exactly.
- [x] A failed numba seed bridge cannot pass silently when determinism is
  requested.
- [x] The adapter delegates to `multi_split`, states both orientations, and
  documents fixed-order reproducibility versus order invariance.

#### Validation

```bash
python -m pytest -q tests/test_count_split.py
python -m pytest -q tests/test_robust_pipeline.py
```

Record the implementation commit, conservation summary, and test output.

### M5 Expose seeded Leiden consistently (P1)

**Depends on:** M0

Add one canonical `random_state` parameter to
`perform_leiden_clustering`, accepting `seed` as a compatibility alias if
needed, and forward it to `leidenalg.find_partition`. Preserve the current
return tuple. Propagate the parameter through `single_graph_and_leiden` and
gene-module workflows, recording the resolved seed and package versions in
higher-level provenance.

The same graph, resolution, dependency versions, and seed must produce exactly
the same labels in a regression fixture. After release, LUAD's local helpers
and monkeypatches can be removed in a separate LUAD commit.

#### Acceptance criteria

- [x] Same graph, resolution, dependency versions, and seed produce identical
  labels and preserve the existing return contract.
- [x] `random_state` is canonical and any `seed` alias is deterministic and
  conflict-checked.
- [x] Graph and gene-module workflows propagate and record the resolved seed
  plus Leiden/igraph versions.
- [x] LUAD's local seeded helper call patterns have a source-compatible package
  replacement; no LUAD dataset execution is required.

#### Validation

```bash
python -m pytest -q tests/test_leiden_seed.py tests/test_graph.py
```

Record the implementation commit, version/seed provenance, and test output.

### M6 Integrate public APIs, documentation, and release validation (P0 gate)

**Depends on:** M0-M5

Integrate completed milestones without creating a second orchestration layer or
moving Tahoe-specific policy into the package.

Deliverables:

- public exports, compatibility aliases, and deprecation guidance;
- README tutorials for count splitting, partition factors, DE design, selected
  contrasts, diagnostics, provenance, and CPU controls;
- required core and pinned-DE CI jobs that cannot silently skip critical tests;
- a synthetic package integration flow adapted from LUAD's public call pattern,
  plus source-level review of LUAD call sites without re-analyzing its data; and
- immutable package/dependency pins and release notes.

#### Acceptance criteria

- [ ] Every M0-M5 acceptance checkbox is complete with linked evidence.
- [ ] Public examples use only exported APIs and reconstruct results from
  serialized provenance.
- [ ] Legacy aliases have tests and actionable deprecation messages.
- [ ] Core and DE CI jobs pass in clean environments with resolved versions.
- [ ] No production path imports a dirty sibling or applies a runtime monkeypatch.

#### Validation

```bash
python -m pytest -q
python -m build
```

Complete the LUAD source-compatibility checklist and record the release-candidate
commit, CI run, built wheel metadata, and documentation review.

### D1 Keep streaming QC deferred (P2)

**Depends on:** M6 and measured Tahoe pilot access patterns

Do not add a speculative streaming abstraction in this branch. Document the
existing separation among metric calculation, threshold derivation,
classification, and plotting, and capture Tahoe pilot measurements that would
drive a future API. Tahoe retains source scanning, mergeable summaries,
stable-ID sampling, and threshold policy until that follow-up is approved.

#### Start criteria

- [ ] M6 is complete and a frozen Tahoe pilot has measured actual access paths.
- [ ] The proposal identifies exact versus approximate metrics, merge behavior,
  stable-ID sampling, error bounds, and merge-order validation.
- [ ] A separate follow-up plan is approved. D1 does not add implementation
  code to the current hardening branch.

## Release Validation

Each milestone owns its focused tests and evidence; this section defines only
suite-level validation so commands are not duplicated in multiple inventories.

CI is required to provide:

- a fast core matrix across supported Python versions;
- a pinned PyDESeq2 job that fails on install, collection, or all-tests-skipped;
- the M6 synthetic public-workflow integration test without LUAD data;
- build and wheel metadata validation; and
- failure on dirty-sibling imports or runtime monkeypatch requirements.

M6 runs the complete release validation:

```bash
python -m pytest -q
python -m build
```

## Progress Log

The tracker table is the SSoT for current status. This append-only log records
events and evidence; it does not redefine milestone status.

| Date | Milestone | Event | Commit | Validation/evidence |
| --- | --- | --- | --- | --- |
| 2026-07-24 | PLAN | Initial audit, requester decisions, and baseline | `634bc8c`, `9c2165c`, `570b887`, `5f4e233` | baseline `38 passed`; staged diff checks passed |
| 2026-07-24 | PLAN | Dependency tracker and provenance/SSoT audit | `dedf910` | diff, heading-spacing, and code-fence checks passed |
| 2026-07-24 | M0 | Started provenance, SSoT, and compatibility foundation | pending | clean `feature/tahoe-hardening` worktree |
| 2026-07-24 | M0 | Added canonical JSON, hash semantics, stable IDs, and immutable provenance envelopes | `a8fcaea` | `23 passed` focused; `61 passed` full suite |
| 2026-07-24 | M0 | Added structured compatibility reports, strict policies, axis/hash checks, split conservation, and parameter validation | `2ce6162` | `43 passed` focused; `81 passed` full suite |
| 2026-07-24 | M0 | Integrated immutable provenance envelopes and deterministic exports into pseudobulk, DE, and pathway result APIs | `44f549b` | `3 passed` focused; existing pathway/provenance tests `35 passed` |
| 2026-07-24 | M0 | Canonicalized robust provenance, ordered AnnData axis hashes, lossless manifests, and immutable post-run diagnostics | `2b5f1c3` | `31 passed` focused; `84 passed` full suite |
| 2026-07-24 | M0 | Canonicalized gene-module report serialization and closed M0 acceptance criteria | `ba33452` | exact M0 validation `45 passed`; gene/robust tests `15 passed`; full suite `84 passed` |
| 2026-07-24 | M1 | Started PyDESeq2 fallback audit from clean base `5196ce1`; dirty sibling retained as reference | pending | clean sibling worktree creation next |
| 2026-07-24 | M1 | Added committed IRLS and Cox-Reid fallback diagnostics in clean PyDESeq2 worktree | `8d1c39f` (`5196ce1` base) | focused `3 passed`; singular retry smoke passed; stock/hardened ordinary utility outputs byte-identical; PyDESeq2 suite collection initially blocked by missing test dependencies |
| 2026-07-24 | M1 | Revalidated clean PyDESeq2 against isolated compatible dependencies | `8d1c39f` | `tests/test_pydeseq2.py`: `38 passed`; `tests/test_edge_cases.py`: `19 passed in 31.69s`; focused fallback tests: `3 passed`; no shared environment changes |
| 2026-07-24 | M1 | Reproduced byte-identical ordinary-fit numerical comparison | `0700d48` | stock `5196ce1c` and hardened `8d1c39f` produced identical beta, mu, H, alpha, and convergence JSON; pin gate remains |
| 2026-07-24 | M1 | Closed numerical, fallback, and machine-readable diagnostic acceptance criteria | `fa162e5` | first four M1 checkboxes checked; immutable production pin remains the sole open M1 gate |
| 2026-07-24 | M1 | Pinning gate audited | pending | package metadata still has unversioned `pydeseq2`; clean fallback commit `8d1c39f` is local and not yet a resolvable published/fork pin, so M1 remains in progress |
| 2026-07-24 | M1 | Added PyDESeq2 fallback decision record with exact commits, epsilon policy, and release-pin blocker | `84760ef` | `docs/decisions/001-pydeseq2-fallbacks.md`; immutable publication/approval remains required |
| 2026-07-24 | M2 | Started explicit DE design and diagnostics implementation after M0/M1 dependency audit | pending | preserve shorthand and add focused design/diagnostic tests |
| 2026-07-24 | M2 | Added canonical formula DesignSpec, annotation-only metadata, `pairs` alias, safe worker defaults, and result design records | `ac3ae75` | `tests/test_de_design.py`: `3 passed`; full suite: `87 passed, 15 warnings`; M2 remains open for complete fit/contrast diagnostics |
| 2026-07-24 | M2 | Added structured fit status, terminal errors, and design-matrix rank/non-finite diagnostics | `9221856` | full suite: `87 passed, 15 warnings`; per-contrast records and acceptance fixtures remain |
| 2026-07-24 | M2 | Added stable per-contrast IDs, vectors, direction, labels, and non-finite result counts | `191c731` | full suite: `87 passed, 15 warnings`; dedicated diagnostic fixtures remain |
| 2026-07-24 | M2 | Added public `DEFitError` with JSON-safe terminal diagnostics and failure fixture | `ec6896f` | focused design/diagnostic tests: `5 passed`; full suite: `89 passed, 15 warnings`; filtering/rank/fallback acceptance coverage remains |
| 2026-07-24 | M2 | Integrated per-fit PyDESeq2 fallback records and convergence counts | `4f8d6cf` | diagnostic tests: `3 passed`; full suite: `97 passed, 22 warnings`; numerical/rank/filter acceptance remains |
| 2026-07-24 | M2 | Added annotation, rank, filtering, requested-pair, fallback, and terminal-failure fixtures | `02a5be9` | design/diagnostic tests: `9 passed`; full suite: `102 passed, 24 warnings`; M1 dependency and ordinary numerical comparison remain |
| 2026-07-24 | M2 | Closed all local M2 acceptance criteria, including canonical contrast provenance reconstruction | `528f240` | focused design/diagnostic/provenance tests: `13 passed`; M2 remains in progress only because its declared M1 dependency is not release-closed |
| 2026-07-24 | M3 | Started metadata-driven pseudobulk boundary implementation after M0 dependency gate | `f9e6d74` | reuse METIS builder per exact joint boundary and preserve source-cell provenance |
| 2026-07-24 | M3 | Added joint `partition_by` boundaries, source IDs/hashes, retain-source control, and METIS advisory warnings | `1e999d9` | focused boundary/axis tests: `4 passed`; full suite before hash correction: `91 passed, 21 warnings`; acceptance review remains |
| 2026-07-24 | M3 | Added explicit source count, factor/value, mode, seed, ID-source, and retention metadata fields | `a24bc95` | focused boundary/axis tests: `4 passed`; full suite plus build rerun successfully |
| 2026-07-24 | M3 | Corrected advisory warning to include exact boundary keys and proved every source cell is assigned exactly once | `a82f099` | boundary tests: `2 passed`; exact source-ID union and uniqueness asserted |
| 2026-07-24 | M3 | Added sparse-count support and integer-factor fixture; normalized sparse metadata weights | `7adedab` | boundary/count-split focused tests: `7 passed`; all eligible source IDs remain unique |
| 2026-07-24 | M3 | Closed M3 acceptance criteria after exact boundary, warning, sparse, and integer-factor validation | `e715e0d` | all M3 checkboxes checked; M3 status done |
| 2026-07-24 | M4 | Started validated count-split adapter implementation | `4677292` | cells x genes public contract, conservation, deterministic seed policy, and shared robust path |
| 2026-07-24 | M4 | Added preflight validation, exact dense/sparse conservation, seed-failure policy, and shared `do_splits` integration | `a9804b7` | adapter/pipeline tests: `8 passed`; full suite: `94 passed, 22 warnings` |
| 2026-07-24 | M4 | Added explicit negative-count and seed-bridge failure fixtures | `35e8c61` | adapter tests: `4 passed`; deterministic, dense/sparse, malformed, and validation paths covered |
| 2026-07-24 | M4 | Closed M4 acceptance criteria after adapter/pipeline validation | `e715e0d` | all M4 checkboxes checked; full release gate remains open |
| 2026-07-24 | M5 | Started canonical seeded-Leiden implementation after M0 dependency gate | `b26c750` | random_state/seed alias, graph wrapper, and gene-module propagation |
| 2026-07-24 | M5 | Added deterministic Leiden seed alias, workflow propagation, and dependency-version provenance | `c6d09f3` | focused seed/graph/gene-module tests: `12 passed`; full suite: `96 passed, 22 warnings` |
| 2026-07-24 | M5 | Closed M5 acceptance criteria after seeded graph and LUAD source-pattern audit | `e715e0d` | all M5 checkboxes checked; no LUAD data execution performed |
| 2026-07-24 | M5 | LUAD source compatibility audit recorded | `6c3df14` | template helper `_seeded_leiden_labels` at `../bfx-luad-scrnaseq-GSE131907/src/build_selected_gene_robust_topology.py:169-190` maps to `sc_robust.utils.perform_leiden_clustering`; no LUAD data executed |
| 2026-07-24 | M6 | Started release integration audit after M0-M5 implementation increments | `8a40926` | public exports, docs, build/CI, dependency pins, and source compatibility remain to audit |
| 2026-07-24 | M6 | Added public root exports, handoff tutorials, and CI package-build validation | `1febfc2` | root import smoke passed; full suite: `96 passed, 22 warnings`; `python -m build` produced sdist and wheel; release pin/acceptance gates remain |
| 2026-07-24 | M6 | Corrected README imports and completed M3 provenance metadata fields | `a24bc95` | final full suite: `96 passed, 22 warnings`; final `python -m build` succeeded; release pin/acceptance gates remain |
| 2026-07-24 | M6 | Final release validation after M3-M5 closure | `c1b2506` | full suite: `99 passed, 24 warnings`; `python -m build` succeeded; M1/M2 acceptance and immutable pin gates remain |
| 2026-07-24 | M6 | Final validation after M2 acceptance fixtures | `82a67d1` | full suite: `102 passed, 24 warnings`; prior package build succeeded; M1 pin and M2/M6 release gates remain |
| 2026-07-24 | M6 | Corrected CI workflow test/build step structure | `d88a332` | workflow now has separate test and build steps; clean CI run and immutable DE dependency pin remain open |
| 2026-07-24 | M6 | Post-SSoT release validation | `5f77423` | full suite: `103 passed, 24 warnings`; `python -m build` succeeded; dependency pin and CI run remain open |
| 2026-07-24 | M6 | Added resolved upstream PyDESeq2 DE CI job | `5027298` | workflow YAML parsed with `test` and `de` jobs; DE constraint is `pydeseq2==0.5.0`; remote CI run and hardened fork pin remain open |
| 2026-07-24 | M6 | Added synthetic public-workflow smoke using LUAD call shape | `b48478a` | focused smoke: `1 passed`; synthetic only, no LUAD data access; public stage provenance reconstructed |
| 2026-07-24 | M6 | Post-smoke release validation | `b48478a` | full suite: `104 passed, 24 warnings`; `python -m build` succeeded; remote CI URL and hardened fork pin remain open |
| 2026-07-24 | M6 | Added actionable deprecation warnings for legacy DE aliases | `6b39890` | focused alias tests: `6 passed`; aliases now direct users to `design`, `annotation_columns`, and `pairs`; full validation follows |
| 2026-07-24 | M6 | Post-alias release validation | `6b39890` | full suite: `105 passed, 24 warnings`; `python -m build` succeeded; remote CI URL and hardened fork pin remain open |

Update protocol:

- Set a milestone `in progress` in the tracker before implementation begins.
- Append commits and validation evidence immediately after each scoped change.
- If blocked, record the concrete blocker, owner, and next decision needed.
- Never mark `done` based only on implementation; acceptance and validation gate it.

## Release Gates

A release candidate may be tagged only when:

- M0-M6 are `done` in the tracker and D1 remains explicitly `deferred`;
- every milestone checkbox has evidence in the append-only progress log;
- no error-severity compatibility finding or unexplained fit failure remains;
- M6 suite, build, CI, and LUAD source-compatibility validation pass;
- package/dependency versions resolve to reviewed immutable commits; and
- Tahoe integration imports released artifacts, never sibling working copies.

D1 requires its own later approval and is not a release blocker.

## Explicit Non-goals

- Tahoe source decoding, canonical shard ordering, distributed split storage,
  task scheduling, retry policy, DMSO eligibility, atlas-wide correction, and
  publication schemas.
- A second count-splitting algorithm or count-split result hierarchy.
- Arbitrary numerical DE contrasts in the initial generic one-factor package
  delivery; Tahoe's final design remains deferred.
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
