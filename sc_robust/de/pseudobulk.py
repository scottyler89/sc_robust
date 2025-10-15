"""Pseudobulk utilities mirroring the legacy analysis scripts."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

from .base import PseudobulkResult
from sc_robust.process_de_test_split import prep_sample_pseudobulk

__all__ = [
    "filter_edges_within_clusters",
    "build_pseudobulk",
    "plot_pseudobulk_scatter",
]


def filter_edges_within_clusters(adj: coo_matrix, clusters: Sequence[Any]) -> coo_matrix:
    """
    Filter edges in a COO adjacency matrix to retain only within-cluster connections.
    """
    if not isinstance(adj, coo_matrix):
        adj = adj.tocoo()
    clusters_arr = np.asarray(clusters)
    if clusters_arr.shape[0] != adj.shape[0]:
        raise ValueError("Length of clusters does not match adjacency shape.")
    mask = clusters_arr[adj.row] == clusters_arr[adj.col]
    return coo_matrix((adj.data[mask], (adj.row[mask], adj.col[mask])), shape=adj.shape)


def _sum_counts_per_cell(counts: Any) -> np.ndarray:
    """Return per-cell total counts for dense or sparse matrices."""
    if sparse.issparse(counts):
        return np.asarray(counts.sum(axis=1)).ravel()
    return np.asarray(counts).sum(axis=1)


def _compute_weighted_proportions(
    cell_indices: Sequence[int],
    labels: Optional[np.ndarray],
    weights: np.ndarray,
) -> Mapping[Any, float]:
    """Compute weighted proportions for a collection of cells."""
    if labels is None:
        return {}
    cell_indices = np.asarray(cell_indices, dtype=int)
    current_labels = labels[cell_indices]
    current_weights = weights[cell_indices]
    total = current_weights.sum()
    if total <= 0:
        total = len(cell_indices)
        current_weights = np.ones_like(current_weights, dtype=float)
    values, inv = np.unique(current_labels, return_inverse=True)
    sums = np.bincount(inv, weights=current_weights)
    return {val: sums[idx] / total for idx, val in enumerate(values)}


def _expand_dict_column(df: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    """Expand a column of dicts into numeric columns."""
    if column not in df.columns:
        return df
    exploded = df[column].apply(lambda x: x if isinstance(x, Mapping) else {})
    if exploded.apply(len).sum() == 0:
        return df
    expanded = pd.DataFrame(exploded.tolist()).fillna(0.0)
    expanded = expanded.add_prefix(prefix)
    for col in expanded.columns:
        df[col] = expanded[col]
    return df


def build_pseudobulk(
    graph: coo_matrix,
    counts: Any,
    *,
    mode: str = "within_cluster",
    cells_per_pb: int = 10,
    cluster_labels: Optional[Sequence[Any]] = None,
    sample_labels: Optional[Sequence[Any]] = None,
    gene_ids: Optional[Sequence[str]] = None,
    coords: Optional[np.ndarray] = None,
    cell_metadata: Optional[pd.DataFrame] = None,
    random_state: Optional[int] = 123456,
    expand_proportions: bool = True,
) -> PseudobulkResult:
    """
    Build pseudobulk expression profiles with optional within-cluster graph filtering.
    """
    if mode not in {"within_cluster", "topology"}:
        raise ValueError("mode must be 'within_cluster' or 'topology'")

    graph = graph.tocoo() if not isinstance(graph, coo_matrix) else graph.copy()
    n_cells = graph.shape[0]

    counts_matrix = counts
    if isinstance(counts, pd.DataFrame):
        if gene_ids is None:
            gene_ids = list(counts.columns)
        counts_matrix = counts.to_numpy()
    if sparse.issparse(counts_matrix):
        n_counts = counts_matrix.shape[0]
    else:
        counts_matrix = np.asarray(counts_matrix)
        n_counts = counts_matrix.shape[0]
    if n_counts != n_cells:
        raise ValueError("counts and graph must have the same number of rows.")

    cluster_arr = np.asarray(cluster_labels) if cluster_labels is not None else None
    sample_arr = np.asarray(sample_labels) if sample_labels is not None else None
    if cluster_arr is not None and len(cluster_arr) != n_cells:
        raise ValueError("cluster_labels must match number of cells.")
    if sample_arr is not None and len(sample_arr) != n_cells:
        raise ValueError("sample_labels must match number of cells.")

    filtered_graph = graph
    if mode == "within_cluster":
        if cluster_arr is None:
            raise ValueError("cluster_labels required for within_cluster mode.")
        filtered_graph = filter_edges_within_clusters(graph, cluster_arr)

    rng_state = None
    if random_state is not None:
        rng_state = np.random.get_state()
        np.random.seed(random_state)

    try:
        pb_exprs, pb_meta = prep_sample_pseudobulk(
            filtered_graph,
            counts_matrix,
            cells_per_pb=cells_per_pb,
            sample_vect=sample_arr,
            cluster_vect=cluster_arr,
            gene_ids=gene_ids,
            coords=coords,
            cell_meta=cell_metadata,
        )
    finally:
        if rng_state is not None:
            np.random.set_state(rng_state)

    cell_totals = _sum_counts_per_cell(counts_matrix)
    pb_meta = pb_meta.copy()
    cluster_weight_dicts = []
    sample_weight_dicts = []
    for _, row in pb_meta.iterrows():
        cells = row["source_cells"]
        cluster_weight_dicts.append(
            _compute_weighted_proportions(cells, cluster_arr, cell_totals)
        )
        sample_weight_dicts.append(
            _compute_weighted_proportions(cells, sample_arr, cell_totals)
        )
    pb_meta["cluster_weight_counts"] = cluster_weight_dicts
    pb_meta["sample_weight_counts"] = sample_weight_dicts

    if expand_proportions:
        pb_meta = _expand_dict_column(pb_meta, "sample_proportions", "sample_prop__")
        pb_meta = _expand_dict_column(pb_meta, "cluster_proportions", "cluster_prop__")
        pb_meta = _expand_dict_column(pb_meta, "cluster_weight_counts", "cluster_weight__")
        pb_meta = _expand_dict_column(pb_meta, "sample_weight_counts", "sample_weight__")

    parameters = {
        "mode": mode,
        "cells_per_pb": cells_per_pb,
        "random_state": random_state,
        "n_cells": n_cells,
    }
    graph_summary = {
        "edges_initial": graph.nnz,
        "edges_filtered": filtered_graph.nnz,
    }
    if isinstance(pb_exprs, pd.DataFrame):
        pb_counts = pb_exprs.loc[pb_meta.index]
    else:
        pb_counts = pd.DataFrame(pb_exprs, index=pb_meta.index)

    return PseudobulkResult(
        counts=pb_counts,
        metadata=pb_meta,
        parameters=parameters,
        graph_summary=graph_summary,
    )


def plot_pseudobulk_scatter(
    result: PseudobulkResult,
    *,
    color: str = "total_count_sum",
    cmap: str = "inferno",
    ax: Optional[Any] = None,
    point_size: float = 10.0,
    linewidth: float = 0.0,
    title: str = "Pseudobulk UMAP",
):
    """
    Scatter plot helper mirroring the reference pseudobulk diagnostic figures.
    """
    meta = result.metadata
    if "pb_coord_x" not in meta.columns or "pb_coord_y" not in meta.columns:
        raise ValueError("Metadata must contain 'pb_coord_x' and 'pb_coord_y' columns.")

    data = meta.copy()
    if color not in data.columns:
        raise ValueError(f"Column '{color}' not found in metadata.")
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        created_fig = True
    else:
        fig = ax.figure

    scatter = ax.scatter(
        data["pb_coord_x"],
        data["pb_coord_y"],
        c=data[color],
        cmap=cmap,
        s=point_size,
        linewidth=linewidth,
    )
    ax.set_xlabel("pb_coord_x")
    ax.set_ylabel("pb_coord_y")
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, label=color)
    if created_fig:
        fig.tight_layout()
    return fig, ax
