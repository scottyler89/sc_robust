# sc_robust
A pipeline for robust reproducible single cell processing

Installation
------------

Option 1: pip (CPU-only FAISS)

```
python -m venv .venv && source .venv/bin/activate  # optional
pip install -r requirements.txt
```

Notes:
- The requirements pin `faiss-cpu`. On some platforms, pip wheels may be limited. If pip fails on FAISS, use Conda below.
- GPU FAISS is not required for this package; CPU FAISS works well for typical sizes.

Option 2: Conda (recommended for FAISS/igraph)

```
conda create -n sc_robust python=3.10 -y
conda activate sc_robust
conda install -c conda-forge anndata numpy scipy matplotlib seaborn statsmodels networkx igraph leidenalg pymetis -y
conda install -c pytorch faiss-cpu -y   # or: conda install -c conda-forge faiss
pip install torch count_split anticor_features
```

Optional/adjacent tools
- scanpy: plotting / clustering convenience (not required by sc_robust core)
- umap-learn: if you want to run UMAP on precomputed graphs

Python compatibility
- Tested on Python 3.10+. Other versions may work, but 3.10 is recommended.

QC Workflow Example
-------------------

The repository ships a reference quality-control scaffold in `sc_robust/qc.py`. It
computes mitochondrial / ribosomal / lncRNA metrics, derives heuristic
thresholds, and groups cells into interpretable QC buckets. A minimal example:

```
from pathlib import Path
import anndata as ad
from sc_robust.qc import quantify_qc_metrics, determine_qc_thresholds, classify_qc_categories

adata = ad.read_h5ad("data/anndata.h5ad")

# Phase 1 – quantify QC metrics (optionally emits plots)
quant_res = quantify_qc_metrics(
    adata,
    plotting_dir=Path("figures/qc"),
    make_plots=True,
    plot_annotation_keys=("sample",),  # columns from `adata.obs` to color plots
)

# Merge the QC metrics back into the working AnnData
qc_df = quant_res.to_dataframe(prefix="qc_")
adata.obs = adata.obs.join(qc_df, how="left")

# Phase 2 – derive thresholds and classify cells
thresholds = determine_qc_thresholds(quant_res.adata)
summary = classify_qc_categories(quant_res.adata, thresholds)

# Filter to high-quality cells
filtered = quant_res.adata[quant_res.adata.obs["qc_keep"].to_numpy()].copy()
print(summary)
print(filtered)
```

See `sc_robust/qc.py` for a ready-to-run `perform_qc_and_filtering` orchestration
function that combines these steps and materializes plots.

Single-Graph Usage (No Splits)
------------------------------

You can reuse the graph-building pipeline on any embedding or feature matrix and cluster with Leiden without using train/validation splits.

Example:

```
import numpy as np
from sc_robust.utils import build_single_graph, single_graph_and_leiden

# Suppose E is an (n_samples, n_dims) embedding
E = np.random.randn(500, 32).astype(np.float32)

# Build a graph using cosine metric (default)
G = build_single_graph(E, k=None, metric='cosine', symmetrize='none')

# Or build and cluster in one step
G, labels = single_graph_and_leiden(E, k=None, metric='cosine', resolution=1.0)

# To use Euclidean distance instead of cosine
G_l2 = build_single_graph(E, k=None, metric='l2', symmetrize='max')
```

Notes:
- The default `k` is round(log(n)) but floored to 20 when possible (and capped by 200 and by `n`).
- Weighting uses the package’s per-node linear rescale; masking is adaptive per-node by distance differences.
- Metrics:
  - `cosine` (default): inner product on L2-normalized rows.
  - `l2`: squared Euclidean distances via FAISS `IndexFlatL2`.
  - `ip`: raw inner product (pseudo-distance `-sim`).

Robust Pipeline Tutorial
------------------------

This is the split-based workflow that builds a consensus KNN graph from train/val splits, then clusters with Leiden.

Basics

```
import anndata as ad
import scanpy as sc
from sc_robust import robust
from sc_robust.utils import perform_leiden_clustering

# Load your AnnData
adata = ad.read_h5ad("path/to/data.h5ad")

# Optional: ensure gene identifiers are accessible
gene_ids = (
    adata.var.get("gene_ids", adata.var.get("gene_name", adata.var.index)).tolist()
)

# Build the robust object (splits -> normalize -> feature select -> PCs -> graph)
ro = robust(
    adata,
    gene_ids=gene_ids,
    norm_function="pf_log",
    # anticor_features integration knobs
    scratch_dir="scratch/anticor",
    offline_mode=True,  # hard-enforce no live GO/g:Profiler lookups
    use_live_pathway_lookup=False,
    do_plot=False,
)

# The consensus graph is available as a scipy.sparse COO matrix
G = ro.graph

# Cluster with Leiden (igraph/leidenalg backend) or Scanpy
clusters, partition, labels = perform_leiden_clustering(G, resolution_parameter=1.0)
adata.obs["leiden"] = labels.astype(int).astype("category")

# Or use Scanpy if preferred
# sc.tl.leiden(adata, adjacency=G.tocsr())
```

Pseudobulk Preparation (Optional)

```
from sc_robust.process_de_test_split import prep_sample_pseudobulk

# Build METIS-based pseudobulk partitions from the graph and counts
pb_exprs, pb_meta = prep_sample_pseudobulk(
    ro.graph,                 # COO weighted adjacency
    ro.test_counts,           # counts matrix (cells x genes)
    cells_per_pb=10,          # target group size
    sample_vect=adata.obs['sample'].tolist(),
    cluster_vect=adata.obs['leiden'].tolist(),
    gene_ids=adata.var_names.tolist(),
    coords=adata.obsm.get('X_umap'),
    cell_meta=adata.obs,      # optional cell-level covariates for aggregation
)
```

Tips
- The default neighbor count is adaptive: `k ≈ round(log(n))` but capped and masked locally.
- The robust object exposes: `train/val/test` (normalized), `train_pcs/val_pcs`, selected features, and the final `graph`.

Offline note
- With recent `anticor_features`, pathway-based pre-removal uses shipped ID banks by default (no network) unless `use_live_pathway_lookup=True` or the bank is missing for your `species`.
- If you need a guarantee that no live lookup can happen, pass `offline_mode=True` (recommended for HPC/sandboxed environments).

API Reference
-------------

- `sc_robust.robust(in_ad, gene_ids=None, splits=[0.325,0.325,0.35], pc_max=250, norm_function='pf_log', species='hsapiens', initial_k=None, do_plot=False, seed=123456)`
  - Builds a consensus KNN graph from train/val splits. Attributes: `graph` (COO adjacency), `indices/distances/weights` (per-node lists), `train/val/test`, `train_pcs/val_pcs`, `train_feature_df/val_feature_df`.

- `sc_robust.utils.perform_leiden_clustering(coo_mat, resolution_parameter=1.0)`
  - Converts COO to igraph and runs Leiden. Returns `(clusters_list, partition_obj, labels_array)`.

- `sc_robust.utils.build_single_graph(embedding_or_X, k=None, metric='cosine', min_k=None, symmetrize='none', use_gpu=False)`
  - Builds a weighted KNN graph directly from an embedding or features using existing masking/weighting. Returns a COO adjacency.

- `sc_robust.utils.single_graph_and_leiden(embedding_or_X, k=None, metric='cosine', resolution=1.0, symmetrize='none', use_gpu=False)`
  - Convenience: builds a graph and runs Leiden. Returns `(graph_coo, labels_array)`.

- `sc_robust.process_de_test_split.prep_sample_pseudobulk(in_graph, X, cells_per_pb=10, sample_vect=None, cluster_vect=None, gene_ids=None, coords=None, cell_meta=None)`
  - METIS partitions of cells into pseudobulk groups based on the graph, with expression and metadata aggregation. Returns `(pb_exprs, annotation_df)`.

- `sc_robust.find_consensus.tsvd(temp_mat, npcs=250)`
  - TruncatedSVD (samples x features) → embedding `(n_samples, npcs)`.

- `sc_robust.find_consensus.find_one_graph(pcs, k=None, metric='cosine', use_gpu=False)`
  - Row-wise KNN neighbors and local-difference mask. Returns `(indices, distances, mask)` (torch tensors).

- `sc_robust.find_consensus.process_idx_dist_mask_to_g(indexes, distances, local_mask)`
  - Converts per-node neighbors, distances, and mask into a weighted COO adjacency using the package’s linear weighting.

Differential Expression Updates
-------------------------------
- The differential-expression helpers automatically merge the packaged `ensg_annotations_abbreviated.txt` lookup so downstream tables always surface `gene_id` and `gene_name` columns, even when the caller does not supply annotations.
- Pathway enrichment now hashes gene memberships and, when `n_jobs != 1`, uses a process-backed executor by default to sidestep the Python GIL. Environments that block process creation will emit a warning and transparently fall back to threaded execution; you can also force threading with `backend="thread"`.
