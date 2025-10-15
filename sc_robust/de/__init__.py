"""Differential expression and pathway analysis utilities."""

from .base import (
    DEAnalysisResult,
    PathwayEnrichmentResult,
    PseudobulkResult,
)
from .pathways import (
    list_available_pathway_libraries,
    load_pathway_library,
    resolve_pathway_filename,
)
from .pseudobulk import (
    build_pseudobulk,
    filter_edges_within_clusters,
    plot_pseudobulk_scatter,
)

__all__ = [
    "DEAnalysisResult",
    "PathwayEnrichmentResult",
    "PseudobulkResult",
    "list_available_pathway_libraries",
    "load_pathway_library",
    "resolve_pathway_filename",
    "build_pseudobulk",
    "filter_edges_within_clusters",
    "plot_pseudobulk_scatter",
]
