"""Pseudobulk utilities mirroring the legacy analysis scripts."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

from .base import PseudobulkResult
from ..provenance import hash_membership_ids
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
    partition_by: Optional[str | Sequence[str]] = None,
    cell_ids: Optional[Sequence[str]] = None,
    retain_source_cells: bool = True,
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
        counts_shape = counts_matrix.shape
    else:
        counts_matrix = np.asarray(counts_matrix)
        counts_shape = counts_matrix.shape
    if len(counts_shape) != 2:
        raise ValueError(f"counts must be a 2D matrix with shape (n_cells, n_genes); got shape={counts_shape}.")
    if counts_shape[0] != n_cells:
        raise ValueError(
            "counts must be cells×genes with the same number of rows as the graph; "
            f"graph.shape={graph.shape} counts.shape={counts_shape}. "
            "If you provided genes×cells, transpose counts."
        )

    cluster_arr = np.asarray(cluster_labels) if cluster_labels is not None else None
    sample_arr = np.asarray(sample_labels) if sample_labels is not None else None
    if cluster_arr is not None and len(cluster_arr) != n_cells:
        raise ValueError("cluster_labels must match number of cells.")
    if cell_metadata is not None:
        if len(cell_metadata) != n_cells:
            raise ValueError("cell_metadata must have one row per graph cell.")
        if cell_metadata.index.has_duplicates:
            raise ValueError("cell_metadata index must contain unique cell IDs.")
    if cell_ids is None:
        resolved_cell_ids = [str(value) for value in (cell_metadata.index if cell_metadata is not None else range(n_cells))]
    else:
        if len(cell_ids) != n_cells:
            raise ValueError("cell_ids must match the number of graph cells.")
        resolved_cell_ids = [str(value) for value in cell_ids]
    if len(set(resolved_cell_ids)) != n_cells:
        raise ValueError("cell_ids must be unique.")
    if partition_by is not None:
        if cell_metadata is None:
            raise ValueError("partition_by requires cell_metadata with named factors.")
        factors = [partition_by] if isinstance(partition_by, str) else list(partition_by)
        if not factors or any(not isinstance(factor, str) or not factor for factor in factors):
            raise ValueError("partition_by must contain one or more non-empty metadata column names.")
        missing = [factor for factor in factors if factor not in cell_metadata.columns]
        if missing:
            raise KeyError(f"Partition factors not found in cell_metadata: {missing}")
        boundary_values = cell_metadata.loc[:, factors]
        if boundary_values.isna().any().any():
            raise ValueError("partition_by factors must not contain null values.")
        groups = {}
        for position, values in enumerate(boundary_values.itertuples(index=False, name=None)):
            groups.setdefault(tuple(str(value) for value in values), []).append(position)
        grouped_counts = []
        grouped_metadata = []
        for boundary_key in sorted(groups):
            positions = np.asarray(groups[boundary_key], dtype=int)
            subgroup = build_pseudobulk(
                graph.tocsr()[positions][:, positions].tocoo(),
                counts_matrix[positions],
                mode=mode, cells_per_pb=cells_per_pb,
                cluster_labels=None if cluster_arr is None else cluster_arr[positions],
                sample_labels=None if sample_arr is None else sample_arr[positions],
                gene_ids=gene_ids, coords=None if coords is None else coords[positions],
                cell_metadata=cell_metadata.iloc[positions].copy(),
                random_state=random_state, expand_proportions=expand_proportions,
                cell_ids=[resolved_cell_ids[position] for position in positions],
                retain_source_cells=True,
            )
            subgroup_meta = subgroup.metadata.copy()
            subgroup_counts = subgroup.counts.copy()
            local_sources = subgroup_meta["source_cells"].tolist()
            subgroup_meta["source_cells"] = [
                [resolved_cell_ids[positions[int(local)]] for local in source] for source in local_sources
            ]
            subgroup_meta["source_cell_ids"] = subgroup_meta["source_cells"]
            subgroup_meta["source_cell_hash"] = [
                hash_membership_ids(source, domain="pseudobulk.source-cells").to_dict()
                for source in subgroup_meta["source_cell_ids"]
            ]
            subgroup_meta["boundary_key"] = "|".join(boundary_key)
            subgroup_meta["partition_factors"] = "|".join(factors)
            subgroup_meta["partition_values"] = "|".join(boundary_key)
            subgroup_meta["mode"] = mode
            subgroup_meta["seed"] = random_state
            subgroup_meta["id_source"] = "explicit" if cell_ids is not None else "metadata_index" if cell_metadata is not None else "positional_legacy"
            subgroup_meta["retain_source_cells"] = retain_source_cells
            subgroup_meta["source_cell_count"] = subgroup_meta["source_cells"].map(len)
            for factor, value in zip(factors, boundary_key):
                subgroup_meta[factor] = value
            subgroup_meta.index = [
                f"{subgroup_meta.iloc[index]["boundary_key"]}__pb_{index}"
                for index in range(len(subgroup_meta))
            ]
            subgroup_counts.index = subgroup_meta.index
            grouped_counts.append(subgroup_counts)
            grouped_metadata.append(subgroup_meta)
        combined_meta = pd.concat(grouped_metadata, axis=0)
        combined_counts = pd.concat(grouped_counts, axis=0)
        observed_sizes = combined_meta["cell_n"].astype(int)
        if (observed_sizes != cells_per_pb).any():
            warnings.warn(
                f"METIS target cells_per_pb={cells_per_pb} was advisory for boundary keys {sorted(groups)!r} "
                f"with observed size range {observed_sizes.min()}-{observed_sizes.max()}.",
                UserWarning, stacklevel=2,
            )
        if not retain_source_cells:
            combined_meta = combined_meta.drop(columns=["source_cells", "source_cell_ids"], errors="ignore")
        return PseudobulkResult(
            counts=combined_counts,
            metadata=combined_meta,
            parameters={
                "mode": mode, "cells_per_pb": cells_per_pb, "random_state": random_state,
                "n_cells": n_cells, "partition_by": factors, "partition_levels": [list(key) for key in sorted(groups)],
        "partition_by": None,
        "cell_id_source": "explicit" if cell_ids is not None else "metadata_index" if cell_metadata is not None else "positional_legacy",
        "retain_source_cells": retain_source_cells,
                "cell_id_source": "explicit" if cell_ids is not None else "metadata_index" if cell_metadata is not None else "positional_legacy",
                "retain_source_cells": retain_source_cells,
            },
            graph_summary={"partitioned": True, "boundary_groups": len(groups)},
        )
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
    pb_meta["source_cell_ids"] = [
        [resolved_cell_ids[int(cell)] for cell in source] for source in pb_meta["source_cells"]
    ]
    pb_meta["source_cell_count"] = pb_meta["source_cells"].map(len)
    pb_meta["partition_factors"] = ""
    pb_meta["partition_values"] = ""
    pb_meta["boundary_key"] = ""
    pb_meta["mode"] = mode
    pb_meta["seed"] = random_state
    pb_meta["id_source"] = "explicit" if cell_ids is not None else "metadata_index" if cell_metadata is not None else "positional_legacy"
    pb_meta["retain_source_cells"] = retain_source_cells
    pb_meta["source_cell_hash"] = [
        hash_membership_ids(source, domain="pseudobulk.source-cells").to_dict() for source in pb_meta["source_cell_ids"]
    ]
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
        "partition_by": None,
        "cell_id_source": "explicit" if cell_ids is not None else "metadata_index" if cell_metadata is not None else "positional_legacy",
        "retain_source_cells": retain_source_cells,
    }
    if not retain_source_cells:
        pb_meta = pb_meta.drop(columns=["source_cells", "source_cell_ids"], errors="ignore")
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
