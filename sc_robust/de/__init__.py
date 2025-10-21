"""Differential expression and pathway analysis utilities."""

from .base import (
    DEAnalysisResult,
    PathwayEnrichmentResult,
    PseudobulkResult,
    load_default_gene_annotations,
)
from .pathways import (
    list_available_pathway_libraries,
    load_pathway_library,
    resolve_pathway_filename,
    run_pathway_enrichment,
    run_pathway_enrichment_for_clusters,
)
from .pseudobulk import (
    build_pseudobulk,
    filter_edges_within_clusters,
    plot_pseudobulk_scatter,
)
from .plots import (
    plot_de_volcano,
    plot_pathway_volcano,
    plot_pathway_density_difference,
    pathway_scurve_plot,
)
from .workflow import perform_de_workflow
from .differential_expression import (
    prepare_deseq_dataset,
    fit_deseq_dataset,
    run_cluster_vs_all,
    run_pairwise_de,
    run_all_pairwise_de,
)

__all__ = [
    "DEAnalysisResult",
    "PathwayEnrichmentResult",
    "PseudobulkResult",
    "load_default_gene_annotations",
    "list_available_pathway_libraries",
    "load_pathway_library",
    "resolve_pathway_filename",
    "run_pathway_enrichment",
    "run_pathway_enrichment_for_clusters",
    "build_pseudobulk",
    "filter_edges_within_clusters",
    "plot_pseudobulk_scatter",
    "plot_de_volcano",
    "plot_pathway_volcano",
    "plot_pathway_density_difference",
    "pathway_scurve_plot",
    "perform_de_workflow",
    "prepare_deseq_dataset",
    "fit_deseq_dataset",
    "run_cluster_vs_all",
    "run_pairwise_de",
    "run_all_pairwise_de",
]
