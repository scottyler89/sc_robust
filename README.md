# sc_robust
A pipeline for robust reproducible single cell processing

Installation
------------

Option 1: pip (CPU-only FAISS)

```
python -m venv .venv && source .venv/bin/activate
python -m pip install ".[full]"

```

For pathway-only use, install `sc_robust` without extras; the base package contains
the pathway API and does not eagerly import the full graph/DE stack. For the full
graph/DE pipeline, install `sc_robust[full]`.

For reproducible CI or release reconstruction, install `uv`, run `uv lock --check`, and use `uv sync --frozen` or `uv export --frozen`.

Notes:
- The package installs its runtime dependencies, including `faiss-cpu`. On some platforms, pip wheels may be limited. If pip fails on FAISS, use Conda below.
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
- Supported and tested on Python 3.10-3.12. Other versions are not currently supported.

Data conventions
----------------

- Count matrices are **cells×genes** (rows = cells, columns = genes).
- Graph adjacencies are **cells×cells**.
- Embeddings/PCs are **n_cells×n_dims**.
- `anticor_features` expects cells in columns (genes×cells); sc_robust handles the transpose internally.

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
- The default `k` is `round(log(n))` floored to 10, capped by 200 and by `n` (and requires `n >= 3*k`).
- Weighting uses the package’s per-node linear rescale; masking is adaptive per-node by distance differences.
- Metrics:
  - `cosine` (default): inner product on L2-normalized rows.
  - `l2`: squared Euclidean distances via FAISS `IndexFlatL2`.
  - `ip`: raw inner product (pseudo-distance `-sim`).

Robust Pipeline Tutorial
------------------------

This is the split-based workflow that builds a consensus KNN graph from train/val splits, then clusters with Leiden.

Basics

Gene Modules (Single Dataset + Cohort Meta-Analysis)
---------------------------------------------------

`sc_robust` can reuse the `spearman.hdf5` artifacts written by `anticor_features` (via `robust(..., scratch_dir=...)`)
to build **gene–gene graphs**, discover **positive co-regulated gene modules**, and summarize **negative antagonism**
between those modules.

Key idea:
- **Positive correlations** answer: “same cell-state program” → used for module discovery (Leiden).
- **Negative correlations** answer: “mutually exclusive programs” → used only for module antagonism summaries.

Single dataset (one scratch dir)

```
from sc_robust.sc_robust import robust

ro = robust(
    adata,
    gene_ids=adata.var["gene_ids"].tolist(),
    scratch_dir="results/sample_001",
    offline_mode=True,
)

# Optional post-step (writes artifacts under scratch_dir and records paths in ro.provenance on save)
ro.run_gene_modules(split_mode="union", resolution=1.0)
ro.save("results/sample_001/robust_object.dill")
```

Outputs under `scratch_dir`:
- `gene_modules.tsv.gz`
- `gene_edges_pos.tsv.gz`, `gene_edges_neg.tsv.gz`
- `gene_module_antagonism.tsv.gz`
- `module_stats.json`
- `gene_modules.report.json`

Cohort meta-analysis (many scratch dirs)

```
from pathlib import Path
from sc_robust.gene_modules import run_gene_module_meta_analysis_for_cohort

scratch_dirs = [Path("results") / s for s in ["S1", "S2", "S3"]]
out = run_gene_module_meta_analysis_for_cohort(
    scratch_dirs,
    out_dir=Path("results") / "gene_module_meta",
)
print(out["replicated_modules"])  # replicated_modules.tsv.gz (annotated with support_n_samples)
```

Outputs under `out_dir`:
- `gene_modules_manifest.tsv.gz`
- `replicated_modules.tsv.gz` (includes `support_n_samples`)
- `replicated_module_instances.tsv.gz`
- `replicated_module_antagonism.tsv.gz`
- `*.report.json` sidecars (quick summary + artifact pointers)

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
- If no reproducible structure is found during PC validation, `ro.no_reproducible_pcs=True` and `ro.graph=None` (this is expected on null/no-structure data).

Offline note
- With recent `anticor_features`, pathway-based pre-removal uses shipped ID banks by default (no network) unless `use_live_pathway_lookup=True` or the bank is missing for your `species`.
- If no ID bank is available for your species (or you request custom pathway lists), `anticor_features` may require live lookup unless you provide `id_bank_dir=...`.
- If you need a guarantee that no live lookup can happen, pass `offline_mode=True` (recommended for HPC/sandboxed environments). When a live lookup would otherwise be required, sc_robust will now raise a single actionable error with fixes (disable live lookup, provide an ID bank, or skip pathway removal).

HPC/offline recommended defaults
- Use 3-way splits (train/val/test). If you pass 2-way splits, sc_robust will copy `val` into `test` and emit a warning: you must not use `test` for downstream DE in that case (double dipping).
- Consider setting:
  - `offline_mode=True` (hard guarantee: no network)
  - `use_live_pathway_lookup=False` (explicitly opt out of live GO/g:Profiler)
  - `scratch_dir=...` to persist `anticor_features` artifacts and kept-feature manifests per split (ordering + pathway-removal provenance when available)
  - `pre_remove_pathways=[]` if you want to skip pathway-based pre-removal entirely
  - `count_split_quiet=True` to suppress noisy stdout from `count_split` (set `False` to see its progress prints)
  - `count_split_bin_size=...` if you need to tune memory/performance during splitting
- Graph construction requires `n_cells >= 3*k_used` (with defaults, effectively `n_cells >= 30`), and the returned adjacency is always `n_cells×n_cells` in shape.

API Reference
-------------

 - `sc_robust.robust(...)`
   - Builds a consensus KNN graph from train/val splits. Key knobs: `scratch_dir`, `offline_mode`, `use_live_pathway_lookup`, `pre_remove_pathways`, `count_split_bin_size`, `count_split_quiet`. Attributes: `graph` (COO adjacency), `indices/distances/weights` (per-node lists), `train/val/test`, `train_pcs/val_pcs`, `train_feature_df/val_feature_df`.

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

Pathway Analysis Only (Tutorial)
--------------------------------

If you already have differential-expression results from another workflow and only
want to run pathway analysis in `sc_robust`, use the low-level pathway helpers
directly. You do not need to build a `robust(...)` object or run the full
`perform_de_workflow(...)` pipeline for this use case.

Imports

```python
from pathlib import Path

import pandas as pd

from sc_robust.de import (
    load_pathway_library,
    run_pathway_enrichment,
    run_pathway_enrichment_for_clusters,
)
```

What the pathway code expects

- A DE table is a `pandas.DataFrame` with one row per gene.
- You must provide a gene identifier column, a signed numeric score column, and a p-value column.
- By default, the pathway helpers expect:
  - `gene_name`: gene symbol used to match against the pathway library
  - `stat`: signed gene-level statistic used to rank and orient genes
  - `pvalue`: per-gene p-value
- If you use a different column name, pass it explicitly with `gene_col=...`, `stat_col=...`, and `p_col=...`.
- If you want the highlighted genes in `nom_sig_genes` to use a different significance threshold column such as `padj`, pass `significance_col="padj"`.
- The chosen `stat_col` should be zero-centered and directional. Internally, pathway enrichment runs a one-sample t-test of the pathway's gene-level scores against `0.0`.

Minimal single-contrast example

```python
de_df = pd.DataFrame(
    {
        "gene_name": ["IL7R", "LTB", "MALAT1", "NKG7", "CST3"],
        "stat": [4.8, 3.1, 0.2, -2.7, -1.9],
        "pvalue": [1e-5, 3e-4, 0.42, 8e-3, 1.6e-2],
        "padj": [5e-5, 9e-4, 0.60, 1.2e-2, 2.5e-2],
    }
)

hallmark = load_pathway_library("h.all")

pathway_df = run_pathway_enrichment(
    de_df,
    hallmark,
    gene_col="gene_name",
    stat_col="stat",
    p_col="pvalue",
    significance_col="padj",
    alpha=0.05,
)

print(pathway_df.head())
```

Expected output columns include:
- `size`: number of genes in the reference pathway
- `mean_t`: mean of the supplied gene-level statistic within the pathway
- `enrichment_t`: one-sample t statistic for that pathway
- `p`: pathway-level p-value from the one-sample t test
- `BH_adj_p`: Benjamini-Hochberg adjusted pathway p-value
- `signed_neglog10_BH`: signed summary score for ranking/plotting pathways
- `nom_sig_genes`: comma-separated genes passing the requested significance filter in the same direction as the pathway effect

Using a custom score column

If your DE output uses another gene-level score, pass that column name as
`stat_col`. For example, if your upstream workflow emits a moderated t statistic
in `wald_score`:

```python
de_df = pd.DataFrame(
    {
        "gene_name": ["IL7R", "LTB", "MALAT1", "NKG7", "CST3"],
        "wald_score": [4.8, 3.1, 0.2, -2.7, -1.9],
        "pvalue": [1e-5, 3e-4, 0.42, 8e-3, 1.6e-2],
        "padj": [5e-5, 9e-4, 0.60, 1.2e-2, 2.5e-2],
    }
)

pathway_df = run_pathway_enrichment(
    de_df,
    hallmark,
    gene_col="gene_name",
    stat_col="wald_score",
    p_col="pvalue",
    significance_col="padj",
)
```

Multiple contrasts with one shared score-column name

`run_pathway_enrichment_for_clusters(...)` accepts a mapping of contrast name to
DE table. This is the right entry point when every contrast table uses the same
schema.

```python
de_by_contrast = {
    "cluster_prop__1": pd.DataFrame(
        {
            "gene_name": ["IL7R", "LTB", "MALAT1", "NKG7", "CST3"],
            "pathway_score": [4.8, 3.1, 0.2, -2.7, -1.9],
            "pvalue": [1e-5, 3e-4, 0.42, 8e-3, 1.6e-2],
            "padj": [5e-5, 9e-4, 0.60, 1.2e-2, 2.5e-2],
        }
    ),
    "cluster_prop__2": pd.DataFrame(
        {
            "gene_name": ["IL7R", "LTB", "MALAT1", "NKG7", "CST3"],
            "pathway_score": [-3.9, -2.4, 0.1, 3.3, 2.2],
            "pvalue": [2e-4, 5e-3, 0.70, 7e-4, 1.1e-2],
            "padj": [6e-4, 9e-3, 0.82, 2e-3, 1.8e-2],
        }
    ),
}

pathway_res = run_pathway_enrichment_for_clusters(
    de_by_contrast,
    libraries=["h.all", "c2.all"],
    stat_col="pathway_score",
    gene_col="gene_name",
    p_col="pvalue",
    significance_col="padj",
    alpha=0.05,
    n_jobs=4,
)

cluster1_pathways = pathway_res.per_contrast["cluster_prop__1"]
all_pathways = pathway_res.tidy()
print(cluster1_pathways.head())
print(all_pathways.head())
```

Multiple contrasts with different score-column names

`run_pathway_enrichment_for_clusters(...)` accepts an optional
`stat_col_by_contrast` mapping. If contrast A uses `t_cell_score` and contrast B
uses `my_wald`, pass the mapping directly; the keys must exactly match the
contrast names.

Here is an explicit example of a per-contrast score-column map:

```python
raw_de_by_contrast = {
    "cluster_prop__1": pd.DataFrame(
        {
            "gene_name": ["IL7R", "LTB", "MALAT1", "NKG7", "CST3"],
            "t_cell_score": [4.8, 3.1, 0.2, -2.7, -1.9],
            "pvalue": [1e-5, 3e-4, 0.42, 8e-3, 1.6e-2],
            "padj": [5e-5, 9e-4, 0.60, 1.2e-2, 2.5e-2],
        }
    ),
    "cluster_prop__2": pd.DataFrame(
        {
            "gene_name": ["IL7R", "LTB", "MALAT1", "NKG7", "CST3"],
            "my_wald": [-3.9, -2.4, 0.1, 3.3, 2.2],
            "pvalue": [2e-4, 5e-3, 0.70, 7e-4, 1.1e-2],
            "padj": [6e-4, 9e-3, 0.82, 2e-3, 1.8e-2],
        }
    ),
}

score_column_by_contrast = {
    "cluster_prop__1": "t_cell_score",
    "cluster_prop__2": "my_wald",
}
```

Use the mapping directly:

```python
pathway_res = run_pathway_enrichment_for_clusters(
    raw_de_by_contrast,
    libraries=["h.all"],
    stat_col_by_contrast=score_column_by_contrast,
    gene_col="gene_name",
    p_col="pvalue",
    significance_col="padj",
)
```

The alternative normalization pattern remains useful when downstream code
requires one shared column name:

1. Rename each contrast-specific score column to a common name before calling
   `run_pathway_enrichment_for_clusters(...)`.

```python
prepared = {}
for contrast, df in raw_de_by_contrast.items():
    score_col = score_column_by_contrast[contrast]
    prepared[contrast] = df.rename(columns={score_col: "pathway_score"}).copy()

pathway_res = run_pathway_enrichment_for_clusters(
    prepared,
    libraries=["h.all"],
    stat_col="pathway_score",
    gene_col="gene_name",
    p_col="pvalue",
    significance_col="padj",
)
```

2. Run the single-contrast helper in a loop and concatenate results yourself.

```python
per_contrast = {}
for contrast, df in raw_de_by_contrast.items():
    per_contrast[contrast] = run_pathway_enrichment(
        df,
        hallmark,
        gene_col="gene_name",
        stat_col=score_column_by_contrast[contrast],
        p_col="pvalue",
        significance_col="padj",
    )
```

If you want a single long-form table after the per-contrast loop:

```python
combined_pathways = pd.concat(
    [
        df.assign(contrast=contrast)
        for contrast, df in per_contrast.items()
    ],
    ignore_index=True,
)
```

Pathway library formats

- Packaged libraries can be referenced by filename or prefix, for example:
  - `"h.all"`
  - `"c2.all"`
  - `"c5.all.v2025.1.Hs.symbols.gmt"`
- You can also pass a GMT file from disk:

```python
custom_gmt = Path("refs/custom_pathways.gmt")

pathway_res = run_pathway_enrichment_for_clusters(
    de_by_contrast,
    libraries=[str(custom_gmt)],
    stat_col="pathway_score",
    gene_col="gene_name",
    p_col="pvalue",
)
```

- GMT rows are expected to look like:

```text
PATHWAY_NAME<TAB>description<TAB>GENE1<TAB>GENE2<TAB>GENE3
```

The second GMT field is ignored by `sc_robust`; pathway membership comes from
the remaining tab-delimited gene symbols.

Notes for handoff

- Prefer one row per gene in each DE table.
- Make sure the gene identifier column matches the namespace used by the GMT file, typically HGNC-style symbols in `gene_name`.
- If you already have external DE results, prefer `run_pathway_enrichment(...)` or `run_pathway_enrichment_for_clusters(...)` over `perform_de_workflow(...)`.
- The current `perform_de_workflow(...)` helper hardcodes `stat_col="stat"`, so it is not the right interface for heterogeneous external score columns.


Tahoe Handoff APIs
------------------

For the production input, artifact, failure, and orchestration boundary
contract, see [`docs/production_runbook.md`](docs/production_runbook.md).
For the DE diagnostic field contract, see [`docs/diagnostics_schema.md`](docs/diagnostics_schema.md).

The public contracts below use cells x genes counts and preserve reconstructable
provenance. The examples are synthetic API templates; they do not require or
reanalyze the LUAD dataset.

Validated count splitting

```python
from sc_robust.count_split_adapter import split_counts

train, validation = split_counts(
    counts_cells_by_genes,
    proportions=[0.5, 0.5],
    seed=7,
)
```

The adapter rejects non-finite, negative, non-integral, malformed, or invalid
proportion inputs before calling `multi_split`, and checks exact element-wise
conservation. Reproducibility means the same seed and the same ordered matrix;
it does not claim order-invariance. Sparse inputs return sparse outputs.

Metadata-driven pseudobulk boundaries

```python
from sc_robust.de import build_pseudobulk

result = build_pseudobulk(
    graph,
    counts_cells_by_genes,
    mode="topology",
    cell_metadata=adata.obs,
    cell_ids=adata.obs_names,
    partition_by=["sample", "cluster"],
    retain_source_cells=True,
)
```

`partition_by` forms exact joint metadata keys and reuses the existing METIS
builder independently per key. Results include `source_cell_ids`, a sorted
membership hash, boundary factors, mode, seed, ID source, and source-list
retention. `cells_per_pb` is advisory; inexact METIS sizes emit a warning.

Explicit DE design and selected contrasts

```python
from sc_robust.de import prepare_deseq_dataset, fit_deseq_dataset, run_pairwise_de

dds = prepare_deseq_dataset(
    result,
    design="~ 0 + condition",
    annotation_columns=["sample", "cluster"],
    inference_kwargs={"n_cpus": 4},
)
fit_deseq_dataset(dds)
de = run_pairwise_de(dds, pairs=[("condition_treated", "condition_control")])
print(de.export_provenance())
```

Formula terms determine the model; annotation columns are retained for reporting
and are not silently added. `design_columns` remains a legacy no-intercept shorthand and emits a
`DeprecationWarning`; use an explicit `design` formula instead.
`metadata_columns` is a deprecated alias for `annotation_columns`, and
`cluster_pairs` is a deprecated alias for `pairs`; both aliases emit actionable
`DeprecationWarning`s. The two pair arguments cannot be supplied together. Fit and contrast diagnostics, stable IDs,
resolved worker counts, and terminal failures are machine-readable.

Leiden seed control

```python
from sc_robust.utils import single_graph_and_leiden
graph, labels = single_graph_and_leiden(embedding, random_state=11)
```

Use `random_state` canonically; `seed` is a conflict-checked compatibility alias.

Provenance and SSoT

Treat each result provenance envelope as the single source of truth for inputs,
configuration, identifiers, dependency versions, execution settings, and artifact
identity. Export it with `result.provenance_json()` or `result.export_provenance()`
rather than reconstructing configuration from mutable Python objects.
