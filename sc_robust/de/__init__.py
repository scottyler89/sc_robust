"""Differential expression and pathway analysis utilities."""

from importlib import import_module

_EXPORTS = {
    "DEAnalysisResult": (".base", "DEAnalysisResult"),
    "PathwayEnrichmentResult": (".base", "PathwayEnrichmentResult"),
    "PseudobulkResult": (".base", "PseudobulkResult"),
    "load_default_gene_annotations": (".base", "load_default_gene_annotations"),
    "list_available_pathway_libraries": (".pathways", "list_available_pathway_libraries"),
    "load_pathway_library": (".pathways", "load_pathway_library"),
    "resolve_pathway_filename": (".pathways", "resolve_pathway_filename"),
    "run_pathway_enrichment": (".pathways", "run_pathway_enrichment"),
    "run_pathway_enrichment_for_clusters": (".pathways", "run_pathway_enrichment_for_clusters"),
    "build_pseudobulk": (".pseudobulk", "build_pseudobulk"),
    "filter_edges_within_clusters": (".pseudobulk", "filter_edges_within_clusters"),
    "plot_pseudobulk_scatter": (".pseudobulk", "plot_pseudobulk_scatter"),
    "plot_de_volcano": (".plots", "plot_de_volcano"),
    "plot_pathway_volcano": (".plots", "plot_pathway_volcano"),
    "plot_pathway_density_difference": (".plots", "plot_pathway_density_difference"),
    "pathway_scurve_plot": (".plots", "pathway_scurve_plot"),
    "perform_de_workflow": (".workflow", "perform_de_workflow"),
    "DEFitError": (".differential_expression", "DEFitError"),
    "DesignSpec": (".differential_expression", "DesignSpec"),
    "prepare_deseq_dataset": (".differential_expression", "prepare_deseq_dataset"),
    "fit_deseq_dataset": (".differential_expression", "fit_deseq_dataset"),
    "run_cluster_vs_all": (".differential_expression", "run_cluster_vs_all"),
    "run_pairwise_de": (".differential_expression", "run_pairwise_de"),
    "run_all_pairwise_de": (".differential_expression", "run_all_pairwise_de"),
}

__all__ = list(_EXPORTS)

def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
