"""Differential expression helpers backed by PyDESeq2."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .base import DEAnalysisResult, PseudobulkResult, load_default_gene_annotations

__all__ = [
    "prepare_deseq_dataset",
    "fit_deseq_dataset",
    "run_cluster_vs_all",
    "run_pairwise_de",
    "run_all_pairwise_de",
]


def _import_pydeseq2():
    try:
        from pydeseq2.dds import DeseqDataSet  # type: ignore
        from pydeseq2.ds import DeseqStats  # type: ignore
        from pydeseq2.default_inference import DefaultInference  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Optional dependency 'pydeseq2' is required for differential expression. "
            "Install it via pip or conda before using `sc_robust.de.differential_expression`."
        ) from exc
    return DeseqDataSet, DeseqStats, DefaultInference


def _sanitize_column(name: str) -> str:
    return (
        name.replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace(":", "_")
    )


def _effective_n_jobs(n_jobs: Optional[int]) -> int:
    """Normalize parallelism requests (0 -> all CPUs, negative offsets allowed)."""
    total = os.cpu_count() or 1
    if n_jobs is None:
        return 1
    if n_jobs == 0:
        return total
    if n_jobs < 0:
        return max(1, total + 1 + int(n_jobs))
    return max(1, int(n_jobs))


def _select_design_columns(
    metadata: pd.DataFrame,
    design_columns: Optional[Sequence[str]],
) -> List[str]:
    if design_columns:
        missing = [col for col in design_columns if col not in metadata.columns]
        if missing:
            raise KeyError(f"Design columns not found in metadata: {missing}")
        return list(design_columns)

    candidates = [col for col in metadata.columns if col.startswith("cluster_prop__")]
    if not candidates:
        candidates = [col for col in metadata.columns if col.startswith("cluster_weight__")]
    if not candidates:
        raise ValueError(
            "Could not infer design columns. Provide them explicitly via the 'design_columns' argument."
        )
    return candidates


def prepare_deseq_dataset(
    pseudobulk: PseudobulkResult,
    *,
    design_columns: Optional[Sequence[str]] = None,
    metadata_columns: Optional[Sequence[str]] = None,
    min_counts: Optional[float] = 10.0,
    min_variance: Optional[float] = 1.0,
    gene_list: Optional[Sequence[str]] = None,
    refit_cooks: bool = True,
    inference_kwargs: Optional[Mapping[str, Union[int, float]]] = None,
) -> "DeseqDataSet":
    """
    Prepare a :class:`pydeseq2.dds.DeseqDataSet` using pseudobulk outputs.

    Parameters mirror the reference notebook logic: genes can be filtered by
    minimum total counts / variance, and the design matrix is constructed from
    one-hot or weighted cluster columns with no intercept (cell-means model).
    """
    DeseqDataSet, _, DefaultInference = _import_pydeseq2()

    counts_df = pseudobulk.counts.copy()
    metadata = pseudobulk.metadata.copy()
    metadata = metadata.loc[counts_df.index]

    design_cols = _select_design_columns(metadata, design_columns)
    design_df = metadata.loc[:, design_cols].copy()

    # Optional additional metadata columns (e.g., sample/time/drug)
    if metadata_columns:
        extra_missing = [col for col in metadata_columns if col not in metadata.columns]
        if extra_missing:
            raise KeyError(f"Requested metadata columns not found: {extra_missing}")
        extra_df = metadata.loc[:, metadata_columns]
        design_df = pd.concat([design_df, extra_df], axis=1)

    # Sanitize column names for DESeq2 formula compatibility
    rename_map = {col: _sanitize_column(col) for col in design_df.columns}
    sanitized_design_cols = []
    for col in design_cols:
        new_name = rename_map.get(col, _sanitize_column(col))
        if new_name in sanitized_design_cols:
            raise ValueError(
                "Sanitized design column names are not unique. Please rename cluster columns to avoid collisions."
            )
        sanitized_design_cols.append(new_name)

    design_df = design_df.rename(columns=rename_map)
    design_colnames = sanitized_design_cols

    # Filter genes
    mask = pd.Series(True, index=counts_df.columns)
    if min_counts is not None:
        mask &= counts_df.sum(axis=0) > min_counts
    if min_variance is not None:
        mask &= counts_df.var(axis=0) > min_variance
    if gene_list is not None:
        gene_mask = counts_df.columns.isin(gene_list)
        mask &= gene_mask
    counts_df = counts_df.loc[:, mask]
    if counts_df.shape[1] == 0:
        raise ValueError("No genes remain after filtering; relax filtering thresholds or provide a gene list.")

    inference_kwargs = dict(inference_kwargs or {})
    eff_cpus = _effective_n_jobs(inference_kwargs.get("n_cpus", 0))
    inference_kwargs["n_cpus"] = eff_cpus
    inference = DefaultInference(**inference_kwargs)

    design_formula = "~ 0 + " + " + ".join(design_colnames)
    dds = DeseqDataSet(
        counts=counts_df,
        metadata=design_df,
        design=design_formula,
        refit_cooks=refit_cooks,
        inference=inference,
    )
    return dds


def fit_deseq_dataset(dds: "DeseqDataSet") -> "DeseqDataSet":
    """
    Run the standard DESeq2 fitting steps on an existing dataset.
    """
    _, _, _ = _import_pydeseq2()
    dds.fit_size_factors()
    dds.fit_genewise_dispersions()
    dds.fit_dispersion_prior()
    dds.fit_MAP_dispersions()
    dds.fit_LFC()
    dds.calculate_cooks()
    if getattr(dds, "refit_cooks", False):
        dds.refit()
    return dds


def _ensure_design_matrix(dds: "DeseqDataSet") -> pd.DataFrame:
    if "design_matrix" not in dds.obsm:
        raise AttributeError("DeseqDataSet is missing 'design_matrix' in `.obsm`. Fit the dataset first.")
    matrix = dds.obsm["design_matrix"]
    if not isinstance(matrix, pd.DataFrame):
        matrix = pd.DataFrame(matrix)
    return matrix


def _run_single_contrast(
    dds: "DeseqDataSet",
    contrast: np.ndarray,
    *,
    alpha: float,
    cooks_filter: bool,
    independent_filter: bool,
) -> "DeseqStats":
    _, DeseqStats, _ = _import_pydeseq2()
    ds = DeseqStats(
        dds,
        contrast=contrast,
        alpha=alpha,
        cooks_filter=cooks_filter,
        independent_filter=independent_filter,
    )
    ds.run_wald_test()
    if ds.cooks_filter:
        ds._cooks_filtering()
    if ds.independent_filter:
        ds._independent_filtering()
    else:
        ds._p_value_adjustment()
    ds.summary()
    return ds


def _standardize_gene_annotations(df: pd.DataFrame) -> pd.DataFrame:
    annot = df.copy()
    rename_map = {
        "Gene stable ID": "gene_id",
        "geneID": "gene_id",
        "GeneID": "gene_id",
        "ensg": "gene_id",
        "ensembl_id": "gene_id",
        "Chromosome/scaffold name": "chromosome",
        "chrom": "chromosome",
        "Gene name": "gene_name",
        "geneSymbol": "gene_name",
        "symbol": "gene_name",
        "Gene description": "gene_description",
        "description": "gene_description",
    }
    for orig, new in rename_map.items():
        if orig in annot.columns and new not in annot.columns:
            annot = annot.rename(columns={orig: new})

    if "gene_id" not in annot.columns:
        if annot.index.name == "gene_id":
            annot = annot.reset_index(drop=False)
        else:
            raise KeyError(
                "Gene annotations must include a 'gene_id' column or be indexed by gene_id."
            )

    annot["gene_id"] = annot["gene_id"].astype(str)
    annot = annot.drop_duplicates(subset="gene_id", keep="first")
    annot = annot.set_index("gene_id", drop=False)

    if "gene_name" not in annot.columns:
        raise KeyError("Gene annotations must include a 'gene_name' column.")

    annot["gene_name"] = annot["gene_name"].astype(str)
    return annot


def _merge_annotations(
    results_df: pd.DataFrame,
    gene_annotations: Optional[pd.DataFrame],
) -> pd.DataFrame:
    packaged = load_default_gene_annotations()

    if gene_annotations is None:
        annot = packaged
    else:
        user = _standardize_gene_annotations(gene_annotations)
        annot = packaged.copy()

        extra_cols = [col for col in user.columns if col not in annot.columns]
        if extra_cols:
            annot = annot.join(user[extra_cols], how="left")

        overlapping = [col for col in annot.columns if col in user.columns and col not in {"gene_id"}]
        for col in overlapping:
            annot[col] = annot[col].fillna(user[col])

        extra_genes = user.index.difference(annot.index)
        if len(extra_genes) > 0:
            annot = pd.concat([annot, user.loc[extra_genes]], axis=0, sort=False)

        annot = annot.sort_index()
        annot["gene_id"] = annot.index.astype(str)

    merged = annot.merge(results_df, left_index=True, right_index=True, how="right")
    index_series = pd.Series(merged.index.astype(str), index=merged.index)

    if "gene_id" in merged.columns:
        merged["gene_id"] = merged["gene_id"].fillna(index_series)
    else:
        merged["gene_id"] = index_series

    if "gene_name" in merged.columns:
        merged["gene_name"] = merged["gene_name"].fillna(index_series)
    else:
        merged["gene_name"] = index_series

    return merged


def run_cluster_vs_all(
    dds: "DeseqDataSet",
    *,
    alpha: float = 0.05,
    gene_annotations: Optional[pd.DataFrame] = None,
    cooks_filter: bool = True,
    independent_filter: bool = True,
    plot_dir: Optional[Union[str, Path]] = None,
    save_dir: Optional[Union[str, Path]] = None,
    save_objects: bool = False,
    n_jobs: int = 0,
) -> DEAnalysisResult:
    """Perform cluster-vs-all DE tests using a cell-means design matrix."""
    matrix = _ensure_design_matrix(dds)
    design_cols = list(matrix.columns)
    K = len(design_cols)
    if K < 2:
        raise ValueError("At least two design columns are required for cluster-vs-all contrasts.")

    plot_dir_path = Path(plot_dir) if plot_dir is not None else None
    save_dir_path = Path(save_dir) if save_dir is not None else None
    if plot_dir_path is not None:
        plot_dir_path.mkdir(parents=True, exist_ok=True)
    if save_dir_path is not None:
        save_dir_path.mkdir(parents=True, exist_ok=True)

    tasks: List[Tuple[str, np.ndarray]] = []
    for idx, column in enumerate(design_cols):
        contrast = np.full(K, -1.0 / (K - 1), dtype=float)
        contrast[idx] = 1.0
        tasks.append((column, contrast))

    results_map: Dict[str, Tuple[object, pd.DataFrame]] = {}

    def compute(column: str, contrast: np.ndarray) -> Tuple[str, object, pd.DataFrame]:
        stats_obj = _run_single_contrast(
            dds,
            contrast,
            alpha=alpha,
            cooks_filter=cooks_filter,
            independent_filter=independent_filter,
        )
        results_df = stats_obj.results_df.copy()
        results_df = _merge_annotations(results_df, gene_annotations)
        return column, stats_obj, results_df

    workers = _effective_n_jobs(n_jobs)
    if workers == 1 or len(tasks) <= 1:
        for column, contrast in tasks:
            col, stats_obj, results_df = compute(column, contrast)
            results_map[col] = (stats_obj, results_df)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(compute, column, contrast): column for column, contrast in tasks
            }
            for future in as_completed(future_map):
                col, stats_obj, results_df = future.result()
                results_map[col] = (stats_obj, results_df)

    contrast_results: Dict[str, pd.DataFrame] = {}
    artifacts: MutableMapping[str, object] = {}

    for column in design_cols:
        stats_obj, results_df = results_map[column]
        contrast_results[column] = results_df
        artifacts[column] = stats_obj

        if plot_dir_path is not None:
            ma_obj = stats_obj.plot_MA()
            if hasattr(ma_obj, "savefig"):
                fig = ma_obj
            elif hasattr(ma_obj, "figure"):
                fig = ma_obj.figure
            else:
                fig = None
            if fig is not None:
                fig.savefig(plot_dir_path / f"MA_plot_{column}.png", dpi=600)
                plt.close(fig)

        if save_objects and save_dir_path is not None:
            import dill  # local import to avoid hard dependency otherwise

            with open(save_dir_path / f"deseq_stats_{column}.dill", "wb") as handle:
                dill.dump(stats_obj, handle)
            results_df.to_csv(save_dir_path / f"cluster_vs_all_{column}.csv")

    return DEAnalysisResult(
        dds=dds,
        contrast_results=contrast_results,
        parameters={
            "alpha": alpha,
            "mode": "cluster_vs_all",
            "cooks_filter": cooks_filter,
            "independent_filter": independent_filter,
            "n_jobs": workers,
        },
        design_columns=design_cols,
        artifacts=artifacts,
    )


def run_pairwise_de(
    dds: "DeseqDataSet",
    cluster_pairs: Iterable[Tuple[str, str]],
    *,
    alpha: float = 0.05,
    gene_annotations: Optional[pd.DataFrame] = None,
    cooks_filter: bool = True,
    independent_filter: bool = True,
    plot_dir: Optional[Union[str, Path]] = None,
    save_dir: Optional[Union[str, Path]] = None,
    save_objects: bool = False,
    n_jobs: int = 0,
) -> DEAnalysisResult:
    """
    Perform pairwise differential expression analysis between specified clusters.
    """
    matrix = _ensure_design_matrix(dds)
    design_cols = list(matrix.columns)
    pairs = list(cluster_pairs)
    contrast_results: Dict[str, pd.DataFrame] = {}
    artifacts: MutableMapping[str, object] = {}

    plot_dir_path = Path(plot_dir) if plot_dir is not None else None
    save_dir_path = Path(save_dir) if save_dir is not None else None
    if plot_dir_path is not None:
        plot_dir_path.mkdir(parents=True, exist_ok=True)
    if save_dir_path is not None:
        save_dir_path.mkdir(parents=True, exist_ok=True)

    tasks: List[Tuple[str, str, np.ndarray]] = []
    for cluster1, cluster2 in pairs:
        if cluster1 not in design_cols or cluster2 not in design_cols:
            raise ValueError(f"Both clusters must exist in the design matrix: {cluster1}, {cluster2}")
        contrast = np.zeros(len(design_cols), dtype=float)
        contrast[design_cols.index(cluster1)] = 1.0
        contrast[design_cols.index(cluster2)] = -1.0
        tasks.append((cluster1, cluster2, contrast))

    results_map: Dict[str, Tuple[object, pd.DataFrame]] = {}

    def compute(cluster1: str, cluster2: str, contrast: np.ndarray) -> Tuple[str, object, pd.DataFrame]:
        stats_obj = _run_single_contrast(
            dds,
            contrast,
            alpha=alpha,
            cooks_filter=cooks_filter,
            independent_filter=independent_filter,
        )
        results_df = stats_obj.results_df.copy()
        results_df = _merge_annotations(results_df, gene_annotations)
        key = f"{cluster1}_vs_{cluster2}"
        return key, stats_obj, results_df

    workers = _effective_n_jobs(n_jobs)
    if workers == 1 or len(tasks) <= 1:
        for cluster1, cluster2, contrast in tasks:
            key, stats_obj, results_df = compute(cluster1, cluster2, contrast)
            results_map[key] = (stats_obj, results_df)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(compute, cluster1, cluster2, contrast): (cluster1, cluster2)
                for cluster1, cluster2, contrast in tasks
            }
            for future in as_completed(future_map):
                key, stats_obj, results_df = future.result()
                results_map[key] = (stats_obj, results_df)

    for cluster1, cluster2, _ in tasks:
        key = f"{cluster1}_vs_{cluster2}"
        stats_obj, results_df = results_map[key]
        contrast_results[key] = results_df
        artifacts[key] = stats_obj

        if plot_dir_path is not None:
            ma_obj = stats_obj.plot_MA()
            if hasattr(ma_obj, "savefig"):
                fig = ma_obj
            elif hasattr(ma_obj, "figure"):
                fig = ma_obj.figure
            else:
                fig = None
            if fig is not None:
                fig.savefig(plot_dir_path / f"MA_plot_{key}.png", dpi=600)
                plt.close(fig)
        if save_objects and save_dir_path is not None:
            import dill
            with open(save_dir_path / f"deseq_stats_{key}.dill", "wb") as handle:
                dill.dump(stats_obj, handle)
            results_df.to_csv(save_dir_path / f"pairwise_{key}.csv")

    return DEAnalysisResult(
        dds=dds,
        contrast_results=contrast_results,
        parameters={
            "alpha": alpha,
            "mode": "pairwise",
            "cooks_filter": cooks_filter,
            "independent_filter": independent_filter,
            "n_jobs": workers,
        },
        design_columns=design_cols,
        artifacts=artifacts,
    )


def run_all_pairwise_de(
    dds: "DeseqDataSet",
    *,
    alpha: float = 0.05,
    gene_annotations: Optional[pd.DataFrame] = None,
    cooks_filter: bool = True,
    independent_filter: bool = True,
    plot_dir: Optional[Union[str, Path]] = None,
    save_dir: Optional[Union[str, Path]] = None,
    save_objects: bool = False,
    n_jobs: int = 0,
) -> DEAnalysisResult:
    """Convenience wrapper that evaluates every pair of design columns."""
    matrix = _ensure_design_matrix(dds)
    design_cols = list(matrix.columns)
    if len(design_cols) < 2:
        raise ValueError("Need at least two design columns to run pairwise comparisons.")
    pairwise_iterable = list(combinations(design_cols, 2))
    return run_pairwise_de(
        dds,
        cluster_pairs=pairwise_iterable,
        alpha=alpha,
        gene_annotations=gene_annotations,
        cooks_filter=cooks_filter,
        independent_filter=independent_filter,
        plot_dir=plot_dir,
        save_dir=save_dir,
        save_objects=save_objects,
        n_jobs=n_jobs,
    )
