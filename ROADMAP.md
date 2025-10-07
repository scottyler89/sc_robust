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
