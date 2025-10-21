"""High-level orchestration for pseudobulk → DE → pathway analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .base import DEAnalysisResult, PathwayEnrichmentResult, PseudobulkResult
from .differential_expression import (
    fit_deseq_dataset,
    prepare_deseq_dataset,
    run_cluster_vs_all,
    run_pairwise_de,
    run_all_pairwise_de,
)
from .pathways import run_pathway_enrichment_for_clusters, resolve_pathway_libraries
from .pseudobulk import build_pseudobulk


def _ensure_directory(path: Optional[Union[str, Path]]) -> Optional[Path]:
    if path is None:
        return None
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def perform_de_workflow(
    graph,
    counts,
    *,
    mode: str = "within_cluster",
    cells_per_pb: int = 10,
    cluster_labels: Sequence[Union[str, int]],
    sample_labels: Optional[Sequence[Union[str, int]]] = None,
    gene_ids: Optional[Sequence[str]] = None,
    coords: Optional[np.ndarray] = None,
    cell_metadata: Optional[pd.DataFrame] = None,
    random_state: int = 123456,
    # DE options
    design_columns: Optional[Sequence[str]] = None,
    metadata_columns: Optional[Sequence[str]] = None,
    min_counts: Optional[float] = 5.0,
    min_variance: Optional[float] = 1.0,
    gene_list: Optional[Sequence[str]] = None,
    refit_cooks: bool = True,
    inference_kwargs: Optional[Mapping[str, Union[int, float]]] = None,
    alpha: float = 0.05,
    gene_annotations: Optional[pd.DataFrame] = None,
    # Pairwise options
    pairwise_contrasts: Optional[Iterable[Tuple[str, str]]] = None,
    run_all_pairwise: bool = False,
    pairwise_n_jobs: Optional[int] = None,
    # Pathway options
    pathway_libraries: Optional[Iterable[str]] = None,
    pathway_alpha: float = 0.05,
    pathway_significance_col: Optional[str] = "pvalue",
    pathway_base_dir: Optional[Union[str, Path]] = None,
    pathway_include_pairwise: bool = False,
    pathway_n_jobs: int = 0,
    # Parallelism
    n_jobs: int = 0,
    # Output options
    output_dir: Optional[Union[str, Path]] = None,
    save_pseudobulk: bool = False,
    save_counts_filename: str = "pseudobulk_counts.parquet",
    save_metadata_filename: str = "pseudobulk_metadata.csv",
    cluster_ma_dir: Optional[Union[str, Path]] = None,
    pairwise_ma_dir: Optional[Union[str, Path]] = None,
    de_summary_dir: Optional[Union[str, Path]] = None,
) -> Mapping[str, Optional[Union[PseudobulkResult, DEAnalysisResult, PathwayEnrichmentResult]]]:
    """
    End-to-end pipeline mirroring the legacy notebook workflow.

    Parameters largely mirror the lower-level helpers and provide hooks for
    persistence of intermediate results and visualizations.
    """

    base_output = _ensure_directory(output_dir)
    cluster_ma_path = _ensure_directory(cluster_ma_dir or (base_output / "ma_cluster" if base_output else None))
    pairwise_ma_path = _ensure_directory(pairwise_ma_dir or (base_output / "ma_pairwise" if base_output else None))
    summary_path = _ensure_directory(de_summary_dir or (base_output / "summaries" if base_output else None))

    pb_result = build_pseudobulk(
        graph,
        counts,
        mode=mode,
        cells_per_pb=cells_per_pb,
        cluster_labels=cluster_labels,
        sample_labels=sample_labels,
        gene_ids=gene_ids,
        coords=coords,
        cell_metadata=cell_metadata,
        random_state=random_state,
        expand_proportions=True,
    )

    if save_pseudobulk and base_output is not None:
        counts_df = pb_result.counts
        meta_df = pb_result.metadata
        counts_df.to_parquet(base_output / save_counts_filename)
        meta_df.to_csv(base_output / save_metadata_filename)

    dds = prepare_deseq_dataset(
        pb_result,
        design_columns=design_columns,
        metadata_columns=metadata_columns,
        min_counts=min_counts,
        min_variance=min_variance,
        gene_list=gene_list,
        refit_cooks=refit_cooks,
        inference_kwargs=inference_kwargs,
    )
    dds = fit_deseq_dataset(dds)

    cluster_vs_all = run_cluster_vs_all(
        dds,
        alpha=alpha,
        gene_annotations=gene_annotations,
        plot_dir=cluster_ma_path,
        save_dir=summary_path,
        save_objects=False,
        cooks_filter=True,
        independent_filter=True,
        n_jobs=n_jobs,
    )

    pairwise_result: Optional[DEAnalysisResult] = None
    effective_pairwise_jobs = pairwise_n_jobs if pairwise_n_jobs is not None else n_jobs
    if pairwise_contrasts:
        pairwise_result = run_pairwise_de(
            dds,
            cluster_pairs=pairwise_contrasts,
            alpha=alpha,
            gene_annotations=gene_annotations,
            plot_dir=pairwise_ma_path,
            save_dir=summary_path,
            save_objects=False,
            cooks_filter=True,
            independent_filter=True,
            n_jobs=effective_pairwise_jobs,
        )
    elif run_all_pairwise:
        pairwise_result = run_all_pairwise_de(
            dds,
            alpha=alpha,
            gene_annotations=gene_annotations,
            cooks_filter=True,
            independent_filter=True,
            plot_dir=pairwise_ma_path,
            save_dir=summary_path,
            save_objects=False,
            n_jobs=effective_pairwise_jobs,
        )

    cluster_pathways: Optional[PathwayEnrichmentResult] = None
    pairwise_pathways: Optional[PathwayEnrichmentResult] = None
    resolved_libraries = resolve_pathway_libraries(pathway_libraries, base_dir=pathway_base_dir)
    if resolved_libraries:
        cluster_pathways = run_pathway_enrichment_for_clusters(
            cluster_vs_all.contrast_results,
            libraries=resolved_libraries,
            base_dir=pathway_base_dir,
            stat_col="stat",
            gene_col="gene_name",
            p_col="pvalue",
            significance_col=pathway_significance_col,
            alpha=pathway_alpha,
            n_jobs=pathway_n_jobs,
        )
        if pathway_include_pairwise and pairwise_result is not None:
            pairwise_pathways = run_pathway_enrichment_for_clusters(
                pairwise_result.contrast_results,
                libraries=resolved_libraries,
                base_dir=pathway_base_dir,
                stat_col="stat",
                gene_col="gene_name",
                p_col="pvalue",
                significance_col=pathway_significance_col,
                alpha=pathway_alpha,
                n_jobs=pathway_n_jobs,
            )

    return {
        "pseudobulk": pb_result,
        "dds": dds,
        "cluster_vs_all_de": cluster_vs_all,
        "pairwise_de": pairwise_result,
        "cluster_vs_all_pathways": cluster_pathways,
        "pairwise_pathways": pairwise_pathways,
    }
