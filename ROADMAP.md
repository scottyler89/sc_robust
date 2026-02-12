sc_robust Roadmap
==================

This roadmap outlines a phased plan to:

- Add Euclidean (L2) metric support alongside cosine for FAISS neighbor search.
- Keep the current per-node linear weighting unchanged by default.
- Generalize graph construction to support a single-graph workflow (no train/val split).
- Provide a straightforward Leiden clustering path using the built graph.
- Improve numerical robustness, validation, and documentation.

The plan is designed to preserve backward compatibility by defaulting to existing behavior.

Context & Goals
---------------

- Current graph pipeline builds two KNN graphs (train/val), locally masks edges, merges edges with cross-split validation, and produces a weighted COO adjacency suitable for downstream clustering.
- We want to reuse the core machinery to: (1) optionally use Euclidean distance with Gaussian affinities, and (2) build a single graph from an embedding or feature space and cluster it with Leiden.

Phase 0 — Baseline Hygiene (No Feature Changes)
-----------------------------------------------

- Objective: Improve numerical safety and developer ergonomics before adding options.

- Tasks:
  - [x] Ensure FAISS I/O uses `np.float32` arrays (avoid passing torch tensors directly).
  - [x] Guard cosine normalization for zero-norm rows to prevent NaNs (e.g., skip or set to zeros and mask out later).
  - [x] Tidy debug prints and replace `print(poo)` with informative exceptions or gated logging.
  - [x] Add concise docstrings to key public helpers.

- Acceptance criteria:
  - Running `find_consensus_graph` on a small dataset does not emit NaNs or noisy debug prints.
  - Unit-style checks pass for: safe normalization, dtype handling, and no invalid indices (<0) leakage.

Phase 1 — Metric Options (Weighting Unchanged)
----------------------------------------------

- Objective: Introduce Euclidean (L2) metric support while keeping the existing linear weighting and masking unchanged.

- Tasks:
  - [x] Add `metric` support to FAISS index creation: `'cosine' | 'l2' | 'ip'` (raw inner product) with backward compatibility to the existing `cosine` flag.
  - [x] Add `metric` handling to neighbor search and distance conversion: `1 - sim` for cosine, squared L2 for `l2`, `-sim` for `ip` (pseudo-distance).
  - [x] Thread `metric` through `find_one_graph` while keeping the `cosine` arg for compatibility (fallback: `metric='cosine' if cosine else 'ip'`).
  - [x] Add concise docstrings and input validation for the new `metric` parameter.
  - [x] Sanity-check shapes/types and ensure k is bounded by n (already added in Phase 0; verify still honored).

- API updates (backward compatible defaults):
  - `get_faiss_idx(vectors, metric='cosine', use_gpu=False)`
    - `metric='cosine'`: L2-normalize rows; use `faiss.IndexFlatIP`.
    - `metric='l2'`: no normalization; use `faiss.IndexFlatL2`.
  - `find_k_nearest_neighbors(index, vectors, k, metric='cosine')`
    - `cosine`: FAISS returns inner products; convert to distances `d = 1 - sim`.
    - `l2`: FAISS returns squared L2 distances; keep squared (`d2`) for speed and monotonicity, and document.
  - `find_one_graph(pcs, k=None, metric='cosine', use_gpu=False)`
    - Threads `metric` into index/search; keeps current `distances_to_weights` behavior.

- Additional behavior:
  - Keep local masking unchanged; it relies on sorted distance differences and remains valid for squared L2.

- Acceptance criteria:
  - With no new arguments, output matches current behavior (within tolerance).
  - For `metric='l2'`, neighbor ordering and masking behave sensibly; distances are non-negative and finite.
  - Masked neighbor counts remain ≥ `min_k` in typical cases.

Phase 2 — Single-Graph Builder (No Splits)
------------------------------------------

- Objective: Provide a simple path to construct a graph and cluster with Leiden for datasets without splits.

- Tasks:
  - [x] Implement `build_single_graph(embedding_or_X, k=None, metric='cosine', min_k=None, symmetrize='none') -> coo_matrix` using existing KNN + masking + weighting pipeline.
  - [x] Add optional symmetrization `'none' | 'max' | 'avg'` using existing helpers.
  - [x] Implement `single_graph_and_leiden(embedding_or_X, k=None, metric='cosine', resolution=1.0, symmetrize='none') -> (coo_graph, labels)`.
  - [x] Add concise docstrings for new utilities.

- New utilities:
  - `build_single_graph(embedding_or_X, k=None, metric='cosine', min_k=None, symmetrize='none') -> coo_matrix`
    - If input is high-dimensional `X`, optionally expose a `use_pcs` flag and `pc_max` to run `tsvd` first; otherwise assume `embedding_or_X` is the embedding.
    - Calls `find_one_graph` to get indices/distances/mask; converts to weighted COO via existing helpers.
    - `symmetrize`: `'none' | 'max' | 'avg'`; applies symmetry using elementwise max or average if requested.
  - `single_graph_and_leiden(embedding_or_X, k=None, metric='cosine', resolution=1.0, symmetrize='none') -> (coo_graph, labels)`
    - Chains graph build and Leiden clustering (`utils.perform_leiden_clustering`).

- Acceptance criteria:
  - Basic example runs end-to-end on a toy dataset producing reasonable clusters.
  - Symmetrization options work and preserve sparsity.

Phase 3 — Documentation & Tests (Replaces prior Phase 3/4)
----------------------------------------------------------

- Objective: Provide high-quality inline documentation and basic smoke tests (skipping when optional dependencies are unavailable), instead of integration/benchmark phases.

- Tasks:
  - [x] Write comprehensive docstrings for core graph functions in `find_consensus.py` (KNN build, masking, merging, FAISS helpers).
  - [x] Write comprehensive docstrings for utilities in `utils.py` (single-graph builder and Leiden wrapper).
  - [x] Update README with a “Single-Graph + Leiden” usage example and brief note on metric options.
  - [x] Add `tests/` with pytest smoke tests for graph building (cosine and l2) and optional Leiden clustering; skip if deps missing.

- Acceptance criteria:
  - Docstrings provide clear parameter/return descriptions and edge-case notes.
  - README example runs given dependencies.
  - Tests pass or are cleanly skipped when dependencies are absent.

Phase 5 — Documentation & Examples
----------------------------------

- Objective: Document new options and provide runnable examples.

- Tasks:
  - Update README with a “Single-Graph + Leiden” usage snippet and the new `metric`/`affinity` options.
  - Add an example notebook or script demonstrating:
    - `metric='cosine', affinity='linear'` (parity with current behavior).
    - `metric='l2', affinity='gaussian', sigma='local'` on an embedding.
    - Optional symmetrization and Leiden resolution effects.

- Acceptance criteria:
  - Examples run without additional dependencies beyond `requirements.txt`.

Phase 6 — Future Enhancements (Nice-to-Haves)
---------------------------------------------

- Gaussian/vMF affinity options with local bandwidth (deferred; keep linear default).
- Mutual KNN pruning or Jaccard weighting as optional post-processing.
- Alternative local sigma heuristics (if Gaussian added), e.g., median of first `min_k` distances.
- Option to return CSR in addition to COO for downstream tools.
- Determinism controls (seeds) and reproducibility notes for neighbor search and Leiden.

API Summary (Proposed)
----------------------

```
def get_faiss_idx(vectors: np.ndarray, metric: str = 'cosine', use_gpu: bool = False) -> faiss.Index:
    ...

def find_k_nearest_neighbors(index, vectors: np.ndarray, k: int, metric: str = 'cosine') -> Tuple[np.ndarray, np.ndarray]:
    # returns (indices, distances)

def find_one_graph(pcs: np.ndarray, k: Optional[int] = None, metric: str = 'cosine', use_gpu: bool = False):
    # returns (indexes, distances, local_mask)

def build_single_graph(embedding_or_X: np.ndarray, k: Optional[int] = None, metric: str = 'cosine', min_k: Optional[int] = None, symmetrize: str = 'none') -> coo_matrix:
    ...

def single_graph_and_leiden(embedding_or_X: np.ndarray, k: Optional[int] = None, metric: str = 'cosine', resolution: float = 1.0, symmetrize: str = 'none') -> Tuple[coo_matrix, np.ndarray]:
    ...
```

Defaults maintain current behavior: `metric='cosine'`, linear weighting.

Risks & Mitigations
-------------------

- Cosine + Gaussian: Less well-defined kernel; recommend `metric='l2'` for Gaussian affinities and document behavior if supporting cosine.
- Zero-norm vectors: Guard normalization and mask out affected nodes to prevent NaNs.
- Squared vs. true L2: Use squared L2 for speed; document that masking/ordering is unaffected.
- Symmetry for Leiden: Provide opt-in symmetrization; default to current (asymmetric allowed) to preserve behavior.
- GPU parity: Keep CPU as baseline; test GPU paths opportunistically.

Acceptance Matrix (Abbreviated)
--------------------------------

- Phase 0: Safe dtypes, no NaNs, tidy logs.
- Phase 1: New params work; defaults unchanged; Gaussian weights correct.
- Phase 2: Single-graph utility builds and clusters; symmetrization works.
- Phase 4: Sanity tests for neighbors, masking, weights, and timing pass.
- Phase 5: Docs and examples runnable with existing requirements.


Future Expansion — Differential Expression & Pathways
=====================================================

Phase 1 — Shared Assets & Infrastructure
----------------------------------------
- [x] Package pathway GMT references under `sc_robust/data/pathways/` (or stub loader if licensing blocks distribution) and expose a `load_pathway_library(library: str)` helper with documented fallbacks.
- [x] Define common dataclasses (e.g., `PseudobulkResult`, `DEAnalysisResult`, `PathwayEnrichmentResult`) to standardize outputs and capture metadata (seeds, parameters, file locations).
- [x] Stand up a plotting utilities module scaffold (`sc_robust/de/plots.py`) that will host shared styling knobs for downstream figures.

Phase 2 — Pseudobulk Builders
-----------------------------
- [x] Implement `build_pseudobulk` in `sc_robust/de/pseudobulk.py` wrapping existing `prep_sample_pseudobulk`, with `mode="within_cluster"` (pre-filter edges via `filter_edges_within_clusters`, omit intercept columns) and `mode="topology"` (weight cluster contributions by per-pseudobulk count proportions).
- [x] Support both discrete (`cluster_key`) and continuous (`topology`/graph) inputs, propagating sample/cluster annotations, coordinates, and reproducibility seeds into the result dataclass.
- [x] Add lightweight plotting helpers for pseudobulk diagnostics (UMAP scatter colored by total counts or gene abundance).

Phase 3 — Differential Expression Pipeline
------------------------------------------
- [x] Create `sc_robust/de/differential_expression.py` utilities to prepare `DeseqDataSet` objects (guard imports with informative errors), including default cell-means design (no intercept in within-cluster mode) and topology-aware design matrices.
- [x] Implement cluster-vs-all and pairwise DE routines that return structured results, persist optional artifacts (MA plots, dill files), and surface helper functions for serialization.
- [x] Add plotting helpers (volcano plot, volcano with labels) based on the provided starter code, aligned with the new plotting module.

Phase 4 — Pathway Enrichment & Visualization
--------------------------------------------
- [x] Introduce `sc_robust/de/pathways.py` with loaders for packaged GMTs, enrichment functions (`run_pathway_enrichment`, `run_pathway_enrichment_for_clusters`), and per-cluster aggregation of pathway statistics.
- [x] Port and adapt supplied plotting examples: pathway volcano plots, S-curve plots, and density-difference enrichment plots driven by DEG summaries.
- [x] Wire optional high-level orchestration (`perform_de_workflow`) to run pseudobulk → DE → pathway analysis end-to-end, with hooks to persist results and reuse plotting helpers.

Phase 5 — Documentation, Examples, and Tests
--------------------------------------------
- [ ] Document the full DE/pathway workflow in README/docs (including packaged pathway usage, runtime dependencies, and example CLI/Notebook snippets).
- [ ] Add pytest smoke/integration tests using realistic synthetic data (small AnnData, simple pathway files) covering both pseudobulk modes, DE contrasts, and enrichment logic (skipping gracefully when optional deps absent).
- [ ] Provide reproducibility metadata (timestamps, parameter capture) in result dataclasses and ensure serialization helpers are exercised in tests/examples.


Upcoming — Pathway Enrichment Parallelism Enhancements
======================================================

- [x] Profile current pathway enrichment workflow to confirm bottlenecks and quantify task sizes.
- [x] Introduce ProcessPoolExecutor-backed execution path with picklable task payloads and safe resource loading.
- [x] Precompute gene membership masks/hashes to accelerate pathway subset selection without pandas indexing overhead.
- [x] Add unit and integration tests comparing sequential vs. parallel outputs for representative contrast/library combinations.
- [x] Benchmark new implementation on large contrast sets; document performance and resource considerations.


2026 User Feedback — Improvement Plan (Pending)
==============================================

This section tracks operational improvements based on downstream user feedback:
offline/HPC friendliness, reproducibility, provenance, and clearer conventions.

Phase A — Ergonomics + Safety (Low Risk)
----------------------------------------


2026 Meta-Analysis — Replicated Gene Programs from `spearman.hdf5`
==================================================================

Goal
----
Build a sample-aware, split-aware meta-analysis pipeline over the `spearman.hdf5` artifacts written by `anticor_features`, in order to:

- Identify replicated positive-correlation gene modules (gene programs) across samples.
- Preserve sample-unique structure, while quantifying reproducibility.
- Leverage per-sample empirical cutoffs `Cpos/Cneg` to keep “edge significance” calibrated across heterogeneous samples.
- Support optional local graph weighting / degree control to improve robustness to sample size and correlation-scale shifts.

Inputs (per (sample, split))
----------------------------

- Must read from HDF5:
  - `infile` (Spearman matrix; same as prior versions)
  - `ids/feature_ids_kept` (gene IDs in matrix order)
  - `meta/provenance_json` (schema + pathway removal + params; JSON string)
  - `infile.attrs["Cpos"]`, `infile.attrs["Cneg"]` (empirical cutoffs)

Deliverables (core artifacts)
-----------------------------

- `scratch_dir_index.tsv.gz`: manifest of discovered artifacts (sample, split, paths, schema_version, n_genes, Cpos/Cneg, pathway removal settings).
- `edges_pos.parquet` / `edges_neg.parquet` (or `.tsv.gz`): per-sample sparse edge lists with split support fields.
- `modules_per_sample.tsv.gz`: module membership per sample (+ per-module metrics).
- `meta_modules.tsv.gz`: replicated modules across samples with support statistics.
- `module_antagonism_per_sample.tsv.gz` and `meta_module_antagonism.tsv.gz`: negative-edge summaries between modules.

Phase 0 — Spec + Schema Lock
----------------------------

- [ ] Define output schemas (TSV/Parquet) including required provenance pointers:
  - HDF5 path, `schema_version`, `provenance_json` hash, `Cpos/Cneg`, and thresholds used for any downstream filtering.
- [ ] Decide file format defaults (TSV.GZ vs Parquet) and naming conventions under an output root directory.

Single-Dataset “Turn-Key” Module Discovery (Plugin)
---------------------------------------------------

In addition to cross-sample replication, provide a simple single-dataset path:

- Input: a single `spearman.hdf5` (or a train/val pair under a single scratch dir).
- Output: gene modules + antagonism summaries + provenance-rich manifests.
- Rationale: auto-identifying gene programs/modules is useful even without a cohort meta-analysis.

Tasks:
- [ ] Implement `sc_robust.gene_modules` helpers:
  - `read_spearman_h5(...)` (IDs + cutoffs + provenance)
  - `extract_thresholded_edges(...)` (Cpos/Cneg gating)
  - `reweight_edges_local(...)` (optional local scaling / SNN weights)
  - `run_leiden_gene_modules(...)` (positive graph → modules)
  - `summarize_module_antagonism(...)` (negative graph summaries)
- [ ] Provide “drop-in” convenience wrappers for existing pipelines:
  - accept a `scratch_dir` produced by `robust(..., scratch_dir=...)` and auto-find per-split `spearman.hdf5`.
  - produce `modules.tsv.gz` + `module_stats.json` in that same scratch dir.
- [ ] Ensure outputs include pointers to:
  - `meta/provenance_json`, `Cpos/Cneg`, `feature_ids_kept`, and the exact weighting/Leiden parameters used.

Phase 1 — Preflight + Indexing (No Biology)
-------------------------------------------

- [ ] Enumerate all scratch dirs; find `spearman.hdf5` under `train/` and `val/` (split-aware).
- [ ] Extract and record:
  - `feature_ids_kept`, `n_genes`, `schema_version`, pathway removal settings, and `Cpos/Cneg`.
- [ ] Validate:
  - gene IDs are unique within each file
  - `Cpos` and `Cneg` are finite and within [-1, 1]; flag degenerate cutoffs (e.g. ±1) with a clear reason.
- [ ] Emit `scratch_dir_index.tsv.gz` (+ optional `index.json` summary).

Phase 2 — Per-Sample Signed Sparse Graphs (Using `Cpos/Cneg`)
-------------------------------------------------------------

- [ ] Create thresholded sparse edges per (sample, split):
  - Positive edges: `rho >= Cpos`
  - Negative edges: `rho <= Cneg`
  - Store as COO/CSR (edge list + optional sparse matrix on disk).
- [ ] Within-sample train/val union:
  - Merge policy (inspired by existing train/val cell-graph merge, but sign-safe):
    - Include an edge if it is over-threshold in train OR val.
    - If present in both splits, require sign agreement; otherwise drop and record as sign-discordant.
    - Store `present_train`, `present_val`, `agree_sign`, and `rho_mean` (with non-sig treated as 0 for averaging).
    - Mark confidence: `both_splits` vs `single_split` to support downstream filtering without losing sample-specific structure.
- [ ] Optional local graph weighting / degree control (positive edges for modules):
  - KNN-style degree cap per gene (default ON for module discovery; mutual-KNN default OFF)
    - Default `k_gene = max(min_k, round(log(n_genes)))`, with `min_k=10` and `k_gene <= 200` and `k_gene <= n_genes-1`.
    - Apply to the positive-edge graph only (never mix pos/neg); negatives are summarized separately for antagonism.
  - local scaling / percentile weights (“beyond cutoff” normalization)
  - SNN/Jaccard reweighting based on neighbor overlap
  - Always apply after `Cpos/Cneg` gating (preserve empirical calibration).

Phase 3 — Within-Sample Gene Modules (Positive Graph)
-----------------------------------------------------

- [ ] Run Leiden on the positive-edge graph per sample (weighted).
- [ ] Emit `modules_per_sample.tsv.gz` with:
  - `sample`, `split_mode` (train/val/union), `module_id`, `gene_id`, and per-gene centrality metrics.
- [ ] Emit `module_stats.json` per sample with:
  - module size distribution, intra-module weight summaries, and provenance pointers.
- [ ] Optional refinement:
  - drop tiny modules below `m_min` genes (but record what was dropped).

Phase 4 — Within-Sample Module Antagonism (Negative Graph)
----------------------------------------------------------

- [ ] Summarize negative structure between positive modules:
  - negative edge density, mean negative weight, fraction strong pairs.
- [ ] Emit `module_antagonism_per_sample.tsv.gz`.

Phase 5 — Cross-Sample Replication (Key Meta Step)
---------------------------------------------------

Option A — Co-membership probability graph:
- [ ] Build `P(g1,g2)` = fraction of samples where `g1,g2` co-occur in a module (conditioned on both present).
- [ ] Threshold `P >= p0`; run Leiden to define replicated modules.

Option B — Module overlap matching:
- [ ] Build a module similarity graph across samples (Jaccard/overlap with minimum overlap count).
- [ ] Cluster this graph into “meta-modules”.
- [ ] Define:
  - core genes = appear in ≥k samples in the meta-module
  - peripheral genes = lower support but high within-sample centrality

Common replication constraints:
- [ ] Require gene presence support ≥2 samples (default).
- [ ] Record heterogeneity: support fraction, within-sample stability (train/val agreement), and strength distributions.

Phase 6 — Replicated Antagonistic Axes
--------------------------------------

- [ ] For each pair of replicated modules, meta-analyze antagonism across samples:
  - `n_samples`, `consistency`, and a robust antagonism strength summary.
- [ ] Emit `meta_module_antagonism.tsv.gz`.

Phase 7 — Link Replicated Programs to Replicated Cell Subtypes
--------------------------------------------------------------

- [ ] Score replicated gene modules on cells using test pf-log counts (no Scanpy preprocessing assumptions).
- [ ] Summarize module scores per within-sample cluster and meta-cluster.
- [ ] Meta-analyze module enrichment by meta-cluster (random effects).
- [ ] Emit `meta_cluster_module_enrichment.tsv.gz`.

Phase 8 — “Publication Gene Sets” Rubric
----------------------------------------

- [ ] Choose marker sets per meta-cluster using both:
  - replicated module membership evidence (core/peripheral)
  - meta-DE evidence
- [ ] Emit core (10–50) and extended (≤200) marker sets with support stats.

Phase 9 — Reproducibility + Provenance
--------------------------------------

- [ ] Ensure every output row/file includes:
  - pointer to source HDF5 provenance_json + `Cpos/Cneg`, thresholds (`p0`, `k`, `m_min`), and code version.
- [ ] Cache strategy:
  - never copy dense matrices; persist sparse edges + module assignments + summaries.
  - allow purging intermediates while keeping `spearman.hdf5` + manifests.

- [x] Replace library-level `print()` with `logging` (structured messages + progress hints).
- [x] Avoid persistent global RNG side effects (scope/restore state; thread `random_state` where supported).
- [ ] Eliminate global seeding entirely where possible (may require upstream support in `count_split`, `anticor_features`, `pymetis`).
- [ ] Clarify axis conventions with early validation and actionable errors (cells×genes expected) across `robust` + graph helpers (currently enforced in DE/pseudobulk only).
- [x] Surface anticor_features knobs end-to-end (scratch_dir, offline/id-bank/live-lookup) with clear failure messaging.

Phase B — Reproducibility + Provenance
--------------------------------------

- [x] Record first-class provenance on `robust` objects (inputs summary, hashes/IDs, params, dependency versions).
- [x] Persist feature-selection artifacts (kept feature order + manifest) when scratch_dir is used.
- [x] Make randomness fully deterministic from a single `random_state` across splitting + PC validation bootstraps.

Phase C — Offline/HPC Execution Modes
-------------------------------------

- [x] Make network calls explicit and optional (default local/offline; fail fast when live lookup required).
- [ ] Add execution backend controls where multiprocessing is used (thread/process/sequential) + safe defaults for restricted runtimes.
- [x] Improve progress reporting for long jobs (step + elapsed + sizes).
- [ ] Extend progress reporting to DE/pathways modules.

Phase D — API Conventions + Compatibility
-----------------------------------------

- [ ] Normalize matrix axis conventions across modules (cells×genes everywhere) or require explicit `cell_axis` and validate.
- [x] Add small end-to-end smoke tests on tiny real-ish matrices (beyond unit mocks) to cover major workflows.

Approval-Required Algorithm Changes (Do Not Implement Without Explicit OK)
--------------------------------------------------------------------------

- [ ] Decide a policy for small `k` in KNN masking (baseline can error when default `k=round(log(n))` yields `k<5`).
- [ ] Decide whether to force graph adjacency shape to `n×n` when building COO (prevents non-square shapes in tie / "missing incoming index" cases).
- [ ] Decide behavior for 2-split mode with AnnData + `count_split` (current baseline passes the AnnData object, which may not be supported by all `count_split` versions).
