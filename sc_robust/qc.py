"""Quality control helpers for COV434 single-cell preprocessing.

This module scaffolds the QC workflow into distinct stages:
  1. Quantify QC metrics (with optional plotting hooks).
  2. Determine heuristic cutoffs from those metrics.
  3. Classify cells and materialize filtered AnnData objects.

The intent is to offer an importable API that mirrors prior project scripts
while keeping the phases composable for future experimentation.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


RAW_QC_METRICS = [
    "total_counts",
    "n_genes_by_counts",
]

PERCENT_QC_METRICS = [
    "pct_counts_mt",
    "pct_counts_ribo",
    "pct_counts_lncRNA",
]

LOG_QC_METRICS = [
    "log1p_total_counts",
    "log1p_total_counts_mt",
    "log1p_total_counts_ribo",
    "log1p_total_counts_lncRNA",
]

QC_METRIC_COLUMNS: Tuple[str, ...] = tuple(
    dict.fromkeys(RAW_QC_METRICS + PERCENT_QC_METRICS + LOG_QC_METRICS)
)


@dataclass
class QCQuantificationResult:
    """Container for the quantification stage."""

    adata: ad.AnnData
    plot_paths: Tuple[pathlib.Path, ...] = ()
    extra_columns: Tuple[str, ...] = ()

    def to_dataframe(
        self,
        *,
        prefix: str = "",
        columns: Optional[Sequence[str]] = None,
        include_extra: bool = True,
    ) -> pd.DataFrame:
        """Extract QC metrics as a DataFrame for easy merging."""
        all_columns = list(QC_METRIC_COLUMNS)
        if include_extra:
            all_columns.extend(self.extra_columns)
        selected = columns if columns is not None else all_columns
        existing = [col for col in selected if col in self.adata.obs]
        df = self.adata.obs.loc[:, existing].copy()
        if prefix:
            df = df.add_prefix(prefix)
        return df


@dataclass
class QCFilteringResult:
    """Container for the full QC workflow."""

    adata: ad.AnnData
    filtered_adata: ad.AnnData
    thresholds: Mapping[str, float]
    summary: Mapping[str, int]
    plot_paths: Tuple[pathlib.Path, ...] = ()
    extra_columns: Tuple[str, ...] = ()

    def to_dataframe(
        self,
        *,
        prefix: str = "",
        columns: Optional[Sequence[str]] = None,
        include_extra: bool = True,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of QC metrics suitable for merging into `adata.obs`.

        Parameters
        ----------
        prefix:
            Optional prefix applied to the returned column names.
        columns:
            Subset of columns (before prefixing) to extract. Defaults to all QC metric
            columns recorded in this result.
        include_extra:
            When ``True`` include columns registered in ``extra_columns`` (e.g.,
            classification labels like ``qc_category``).
        """
        all_columns = list(QC_METRIC_COLUMNS)
        if include_extra:
            all_columns.extend(self.extra_columns)
        selected = columns if columns is not None else all_columns
        existing = [col for col in selected if col in self.adata.obs]
        df = self.adata.obs.loc[:, existing].copy()
        if prefix:
            df = df.add_prefix(prefix)
        return df


def _get_gene_symbols(adata: ad.AnnData, gene_symbol_key: str) -> pd.Series:
    """Return a vector of gene symbols, falling back to var_names if needed."""
    if gene_symbol_key in adata.var.columns:
        series = adata.var[gene_symbol_key].astype(str)
    else:
        series = pd.Series(adata.var_names.astype(str), index=adata.var_names, name=gene_symbol_key)
    return series


def _annotate_qc_feature_sets(
    adata: ad.AnnData,
    *,
    gene_symbol_key: str,
    mt_prefix: str,
    ribo_prefixes: Sequence[str],
    lnc_reference_genes: Sequence[str],
) -> None:
    """Flag mitochondrial, ribosomal, and lncRNA marker genes in-place."""
    gene_symbols = _get_gene_symbols(adata, gene_symbol_key)

    adata.var["mt"] = gene_symbols.str.startswith(mt_prefix)
    ribo_mask = pd.Series(False, index=adata.var_names)
    for prefix in ribo_prefixes:
        ribo_mask |= gene_symbols.str.startswith(prefix)
    adata.var["ribo"] = ribo_mask
    adata.var["lncRNA"] = gene_symbols.isin(lnc_reference_genes)


def _calculate_qc_metrics(
    adata: ad.AnnData,
    *,
    gene_symbol_key: str = "gene_name",
    mt_prefix: str = "MT-",
    ribo_prefixes: Sequence[str] = ("RPL", "RPS"),
    lnc_reference_genes: Sequence[str] = ("MALAT1", "NEAT1", "XIST"),
) -> None:
    """
    Annotate QC feature sets and compute Scanpy QC metrics in-place.

    This includes log1p and raw summaries to align with downstream heuristics.
    """
    _annotate_qc_feature_sets(
        adata,
        gene_symbol_key=gene_symbol_key,
        mt_prefix=mt_prefix,
        ribo_prefixes=ribo_prefixes,
        lnc_reference_genes=lnc_reference_genes,
    )

    qc_vars = ["mt", "ribo", "lncRNA"]
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=qc_vars,
        percent_top=None,
        log1p=True,
        inplace=True,
    )
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=qc_vars,
        percent_top=None,
        log1p=False,
        inplace=True,
    )


def _resolve_figure(obj: object) -> Optional[Figure]:
    """Return a matplotlib Figure from a Scanpy plotting return object."""
    if isinstance(obj, Figure):
        return obj
    if isinstance(obj, Axes):
        return obj.figure
    return None


def _plot_violin_panel(
    adata: ad.AnnData,
    output_dir: pathlib.Path,
    *,
    metrics: Sequence[str],
    suffix: str,
) -> Optional[pathlib.Path]:
    """Replicate the Scanpy violin panels used in legacy QC notebooks."""
    if not metrics:
        return None
    missing = [m for m in metrics if m not in adata.obs]
    if missing:
        return None
    violin_obj = sc.pl.violin(
        adata,
        keys=list(metrics),
        jitter=0.4,
        multi_panel=True,
        show=False,
    )
    fig = _resolve_figure(violin_obj)
    if fig is None:
        fig = plt.gcf()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{suffix}_violins.png"
    fig.savefig(path, dpi=400)
    plt.close(fig)
    return path


def _plot_category_boxplots(
    adata: ad.AnnData,
    output_dir: pathlib.Path,
    *,
    metrics: Sequence[str],
    category_key: str,
    suffix: str,
) -> Tuple[pathlib.Path, ...]:
    """Generate seaborn catplots mirroring prior QC scripts."""
    if category_key not in adata.obs:
        return ()
    columns = [category_key] + list(metrics)
    try:
        plot_df = adata.obs[columns].copy()
    except KeyError:
        return ()
    if plot_df.dropna().empty:
        return ()
    melted = plot_df.melt(id_vars=category_key, var_name="metric", value_name="value")
    g = sns.catplot(
        data=melted,
        kind="box",
        x="value",
        y=category_key,
        col="metric",
        hue=category_key,
        sharex=False,
        sharey=True,
        col_wrap=3,
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("Value", category_key)
    g.fig.tight_layout()
    path = output_dir / f"{suffix}_box_{category_key}.png"
    g.fig.savefig(path, dpi=400)
    plt.close(g.fig)
    return (path,)


def _plot_pairplot(
    adata: ad.AnnData,
    output_dir: pathlib.Path,
    *,
    metrics: Sequence[str],
    hue_key: str,
    suffix: str,
    sample_fraction: float,
    random_state: int,
    max_cells: int,
) -> Optional[pathlib.Path]:
    """Generate seaborn pairplots sampled down to avoid overplotting."""
    if hue_key not in adata.obs:
        return None
    columns = [hue_key] + list(metrics)
    try:
        pair_df = adata.obs[columns].copy()
    except KeyError:
        return None
    pair_df = pair_df.dropna()
    if pair_df.empty:
        return None
    n_obs = len(pair_df)
    target_n = min(max_cells, int(max(1, n_obs * sample_fraction)))
    if n_obs > target_n:
        pair_df = pair_df.sample(n=target_n, random_state=random_state)
    g = sns.pairplot(pair_df, hue=hue_key, plot_kws={"s": 8, "edgecolor": "none"})
    if g._legend is not None:
        g._legend.set_bbox_to_anchor((1.05, 0.5))
        g._legend.set_loc("center left")
    g.fig.suptitle(f"Pairplot colored by {hue_key}", y=1.02)
    g.fig.tight_layout()
    path = output_dir / f"{suffix}_pairplot_{hue_key}.png"
    g.fig.savefig(path, dpi=400)
    plt.close(g.fig)
    return path


def _plot_count_plot(
    adata: ad.AnnData,
    output_dir: pathlib.Path,
    *,
    category_key: str,
    suffix: str,
) -> Optional[pathlib.Path]:
    """Bar chart of observation counts per category."""
    if category_key not in adata.obs:
        return None
    counts = adata.obs[category_key].dropna()
    if counts.empty:
        return None
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=adata.obs, y=category_key, order=counts.value_counts().index)
    ax.set_title(f"Counts by {category_key}")
    ax.set_xlabel("Count")
    ax.set_ylabel(category_key)
    plt.tight_layout()
    path = output_dir / f"{suffix}_count_{category_key}.png"
    plt.savefig(path, dpi=400)
    plt.close()
    return path


def _plot_scatter_lnc_vs_mt(
    adata: ad.AnnData,
    output_dir: pathlib.Path,
    *,
    color_key: Optional[str],
    suffix: str,
    sample_fraction: float,
    random_state: int,
) -> Optional[pathlib.Path]:
    """Scatter plot of lncRNA vs mitochondrial fractions, optionally colored."""
    for col in ("pct_counts_lncRNA", "pct_counts_mt"):
        if col not in adata.obs:
            return None
    scatter_df = adata.obs[["pct_counts_lncRNA", "pct_counts_mt"]].copy()
    if color_key is not None and color_key in adata.obs:
        scatter_df[color_key] = adata.obs[color_key].astype(str)
    else:
        color_key = None
    n_obs = len(scatter_df)
    target_n = max(500, int(n_obs * sample_fraction))
    if n_obs > target_n:
        scatter_df = scatter_df.sample(n=target_n, random_state=random_state)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(
        data=scatter_df,
        x="pct_counts_lncRNA",
        y="pct_counts_mt",
        hue=color_key,
        ax=ax,
        s=15,
        edgecolor="none",
    )
    ax.set_title("Fragment signatures")
    ax.set_xlabel("pct_counts_lncRNA")
    ax.set_ylabel("pct_counts_mt")
    if color_key is not None:
        ax.legend(title=color_key, bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        legend = ax.legend()
        if legend is not None:
            legend.remove()
    fig.tight_layout()
    path = output_dir / f"{suffix}_lnc_vs_mt{'_' + color_key if color_key else ''}.png"
    fig.savefig(path, dpi=400)
    plt.close(fig)
    return path


def _generate_qc_plots(
    adata: ad.AnnData,
    output_dir: pathlib.Path,
    *,
    prefix: str,
    metric_sets: Sequence[Tuple[Sequence[str], str]],
    annotation_keys: Sequence[str],
    scatter_color_keys: Sequence[Optional[str]],
    sample_fraction: float,
    random_state: int,
    pairplot_sample_fraction: float,
    pairplot_max_cells: int,
) -> Tuple[pathlib.Path, ...]:
    """Generate a suite of QC plots mirroring historic notebooks."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[pathlib.Path] = []

    for metrics, label in metric_sets:
        suffix = f"{prefix}_{label}"
        violin_path = _plot_violin_panel(adata, output_dir, metrics=metrics, suffix=f"{suffix}_qc")
        if violin_path:
            paths.append(violin_path)
        for ann_key in annotation_keys:
            box_paths = _plot_category_boxplots(
                adata,
                output_dir,
                metrics=metrics,
                category_key=ann_key,
                suffix=f"{suffix}_box",
            )
            paths.extend(box_paths)
            pair_path = _plot_pairplot(
                adata,
                output_dir,
                metrics=metrics,
                hue_key=ann_key,
                suffix=f"{suffix}",
                sample_fraction=pairplot_sample_fraction,
                random_state=random_state,
                max_cells=pairplot_max_cells,
            )
            if pair_path:
                paths.append(pair_path)

    for ann_key in annotation_keys:
        count_path = _plot_count_plot(
            adata,
            output_dir,
            category_key=ann_key,
            suffix=f"{prefix}",
        )
        if count_path:
            paths.append(count_path)

    for color_key in scatter_color_keys:
        scatter_path = _plot_scatter_lnc_vs_mt(
            adata,
            output_dir,
            color_key=color_key,
            suffix=f"{prefix}",
            sample_fraction=sample_fraction,
            random_state=random_state,
        )
        if scatter_path:
            paths.append(scatter_path)

    return tuple(paths)


def quantify_qc_metrics(
    adata: ad.AnnData,
    *,
    gene_symbol_key: str = "gene_name",
    mt_prefix: str = "MT-",
    ribo_prefixes: Sequence[str] = ("RPL", "RPS"),
    lnc_reference_genes: Sequence[str] = ("MALAT1", "NEAT1", "XIST"),
    plotting_dir: Optional[pathlib.Path] = None,
    make_plots: bool = False,
    violin_metrics: Sequence[str] = RAW_QC_METRICS,
    violin_percent_metrics: Sequence[str] = PERCENT_QC_METRICS,
    violin_log_metrics: Sequence[str] = LOG_QC_METRICS,
    plot_annotation_keys: Sequence[str] = (),
    scatter_annotation_keys: Optional[Sequence[Optional[str]]] = None,
    scatter_sample_fraction: float = 0.2,
    random_state: int = 42,
    pairplot_sample_fraction: float = 0.1,
    pairplot_max_cells: int = 2000,
) -> QCQuantificationResult:
    """
    Quantify QC metrics on a copy of the input AnnData and optionally plot them.

    Returns a dataclass capturing the annotated AnnData along with any generated
    figure paths so that downstream stages can reuse the artifacts. By default
    it produces three plot sets: raw count metrics (n_genes, total_counts),
    percentage metrics (mitochondrial / ribosomal / lncRNA fractions), and
    log-transformed totals.
    """
    annotated = adata.copy()
    _calculate_qc_metrics(
        annotated,
        gene_symbol_key=gene_symbol_key,
        mt_prefix=mt_prefix,
        ribo_prefixes=ribo_prefixes,
        lnc_reference_genes=lnc_reference_genes,
    )

    plot_paths: Tuple[pathlib.Path, ...] = ()
    if make_plots and plotting_dir is not None:
        metric_sets: list[Tuple[Sequence[str], str]] = []
        if violin_metrics:
            metric_sets.append((violin_metrics, "counts"))
        if violin_percent_metrics:
            metric_sets.append((violin_percent_metrics, "pct"))
        if violin_log_metrics:
            metric_sets.append((violin_log_metrics, "log"))
        annotation_keys = list(dict.fromkeys(plot_annotation_keys))
        scatter_keys = (
            list(dict.fromkeys(scatter_annotation_keys))
            if scatter_annotation_keys is not None
            else (annotation_keys if annotation_keys else (None,))
        )
        plot_paths = _generate_qc_plots(
            annotated,
            pathlib.Path(plotting_dir),
            prefix="quantification",
            metric_sets=metric_sets,
            annotation_keys=annotation_keys,
            scatter_color_keys=scatter_keys,
            sample_fraction=scatter_sample_fraction,
            random_state=random_state,
            pairplot_sample_fraction=pairplot_sample_fraction,
            pairplot_max_cells=pairplot_max_cells,
        )
    return QCQuantificationResult(adata=annotated, plot_paths=plot_paths)


def determine_qc_thresholds(
    adata: ad.AnnData,
    *,
    doublet_total_counts_quantile: float = 0.98,
    doublet_n_genes_quantile: float = 0.98,
    doublet_max_pct_mito: float = 20.0,
    fragment_lncRNA_high_pct: float = 6.0,
    fragment_lncRNA_low_pct: float = 1.0,
    fragment_mito_high_pct: float = 25.0,
    fragment_max_total_counts: Optional[float] = None,
    fragment_max_n_genes: Optional[float] = None,
    min_total_counts: float = 2000.0,
    min_n_genes: int = 500,
    max_pct_mito: float = 30.0,
) -> Dict[str, float]:
    """
    Derive heuristic thresholds used for downstream classification.

    Quantile-based cutoffs align with prior analyses and keep behaviour robust
    across batches. Absolute thresholds provide guard-rails for obviously poor cells.
    """
    obs = adata.obs
    thresholds: Dict[str, float] = {}

    thresholds["doublet_total_counts"] = float(np.quantile(obs["total_counts"], doublet_total_counts_quantile))
    thresholds["doublet_n_genes"] = float(np.quantile(obs["n_genes_by_counts"], doublet_n_genes_quantile))
    thresholds["doublet_max_pct_mito"] = doublet_max_pct_mito

    thresholds["fragment_lncRNA_high_pct"] = fragment_lncRNA_high_pct
    thresholds["fragment_lncRNA_low_pct"] = fragment_lncRNA_low_pct
    thresholds["fragment_mito_high_pct"] = fragment_mito_high_pct
    thresholds["fragment_max_total_counts"] = (
        float(np.quantile(obs["total_counts"], 0.75)) if fragment_max_total_counts is None else fragment_max_total_counts
    )
    thresholds["fragment_max_n_genes"] = (
        float(np.quantile(obs["n_genes_by_counts"], 0.75))
        if fragment_max_n_genes is None
        else fragment_max_n_genes
    )

    thresholds["min_total_counts"] = min_total_counts
    thresholds["min_n_genes"] = float(min_n_genes)
    thresholds["max_pct_mito"] = max_pct_mito

    return thresholds


def classify_qc_categories(
    adata: ad.AnnData,
    thresholds: Mapping[str, float],
    *,
    category_key: str = "qc_category",
    keep_key: str = "qc_keep",
) -> Dict[str, int]:
    """
    Add QC-driven annotations to `adata.obs` using precomputed thresholds.

    Returns a mapping of QC category labels to observation counts.
    """
    obs = adata.obs
    qc_category = np.full(adata.n_obs, "pass_qc", dtype=object)

    doublet_mask = (
        (obs["total_counts"].to_numpy() >= thresholds["doublet_total_counts"])
        & (obs["n_genes_by_counts"].to_numpy() >= thresholds["doublet_n_genes"])
        & (obs["pct_counts_mt"].to_numpy() <= thresholds["doublet_max_pct_mito"])
    )

    nuclear_fragment_mask = (
        (obs["pct_counts_lncRNA"].to_numpy() >= thresholds["fragment_lncRNA_high_pct"])
        & (obs["pct_counts_mt"].to_numpy() <= thresholds["fragment_mito_high_pct"])
        & (obs["total_counts"].to_numpy() <= thresholds["fragment_max_total_counts"])
        & (obs["n_genes_by_counts"].to_numpy() <= thresholds["fragment_max_n_genes"])
    )

    anucleated_fragment_mask = (
        (obs["pct_counts_lncRNA"].to_numpy() <= thresholds["fragment_lncRNA_low_pct"])
        & (obs["pct_counts_mt"].to_numpy() >= thresholds["fragment_mito_high_pct"])
        & (obs["total_counts"].to_numpy() <= thresholds["fragment_max_total_counts"])
    )

    low_quality_mask = (
        (obs["total_counts"].to_numpy() < thresholds["min_total_counts"])
        | (obs["n_genes_by_counts"].to_numpy() < thresholds["min_n_genes"])
        | (obs["pct_counts_mt"].to_numpy() > thresholds["max_pct_mito"])
    )

    def assign(label: str, mask: np.ndarray, *, allow_override: bool = False) -> None:
        if allow_override:
            qc_category[mask] = label
        else:
            qc_category[(mask) & (qc_category == "pass_qc")] = label

    assign("doublet_candidate", doublet_mask, allow_override=True)
    assign("fragment_nuclear_lncRNA_enriched", nuclear_fragment_mask)
    assign("fragment_anucleated", anucleated_fragment_mask)
    assign("low_quality", low_quality_mask)

    adata.obs[category_key] = pd.Categorical(qc_category, categories=np.unique(qc_category))
    adata.obs[keep_key] = adata.obs[category_key] == "pass_qc"

    summary = adata.obs[category_key].value_counts().to_dict()
    return {str(k): int(v) for k, v in summary.items()}


def perform_qc_and_filtering(
    adata: ad.AnnData,
    *,
    gene_symbol_key: str = "gene_name",
    mt_prefix: str = "MT-",
    ribo_prefixes: Sequence[str] = ("RPL", "RPS"),
    lnc_reference_genes: Sequence[str] = ("MALAT1", "NEAT1", "XIST"),
    plotting_dir: Optional[pathlib.Path] = pathlib.Path("figures/qc"),
    make_plots: bool = True,
    violin_metrics: Sequence[str] = RAW_QC_METRICS,
    violin_percent_metrics: Sequence[str] = PERCENT_QC_METRICS,
    violin_log_metrics: Sequence[str] = LOG_QC_METRICS,
    plot_annotation_keys: Sequence[str] = (),
    scatter_annotation_keys: Optional[Sequence[Optional[str]]] = None,
    scatter_sample_fraction: float = 0.2,
    pairplot_sample_fraction: float = 0.1,
    pairplot_max_cells: int = 2000,
    random_state: int = 42,
    doublet_total_counts_quantile: float = 0.98,
    doublet_n_genes_quantile: float = 0.98,
    doublet_max_pct_mito: float = 20.0,
    fragment_lncRNA_high_pct: float = 6.0,
    fragment_lncRNA_low_pct: float = 1.0,
    fragment_mito_high_pct: float = 25.0,
    fragment_max_total_counts: Optional[float] = None,
    fragment_max_n_genes: Optional[float] = None,
    min_total_counts: float = 2000.0,
    min_n_genes: int = 500,
    max_pct_mito: float = 30.0,
) -> QCFilteringResult:
    """
    Main QC entry-point for the COV434 analysis pipeline.

    The function runs the quantification stage, derives thresholds, annotates QC
    categories, returns filtered AnnData views, and coordinates optional plotting.
    Plotting mirrors the legacy ST941C workflow with count-, percent-, and
    log-metric panels plus pairplots and count summaries keyed to annotations.
    """
    plotting_dir = pathlib.Path(plotting_dir) if plotting_dir is not None else None
    quant = quantify_qc_metrics(
        adata,
        gene_symbol_key=gene_symbol_key,
        mt_prefix=mt_prefix,
        ribo_prefixes=ribo_prefixes,
        lnc_reference_genes=lnc_reference_genes,
        plotting_dir=plotting_dir,
        make_plots=make_plots,
        violin_metrics=violin_metrics,
        violin_percent_metrics=violin_percent_metrics,
        violin_log_metrics=violin_log_metrics,
        plot_annotation_keys=plot_annotation_keys,
        scatter_annotation_keys=scatter_annotation_keys,
        scatter_sample_fraction=scatter_sample_fraction,
        random_state=random_state,
        pairplot_sample_fraction=pairplot_sample_fraction,
        pairplot_max_cells=pairplot_max_cells,
    )
    annotated = quant.adata

    thresholds = determine_qc_thresholds(
        annotated,
        doublet_total_counts_quantile=doublet_total_counts_quantile,
        doublet_n_genes_quantile=doublet_n_genes_quantile,
        doublet_max_pct_mito=doublet_max_pct_mito,
        fragment_lncRNA_high_pct=fragment_lncRNA_high_pct,
        fragment_lncRNA_low_pct=fragment_lncRNA_low_pct,
        fragment_mito_high_pct=fragment_mito_high_pct,
        fragment_max_total_counts=fragment_max_total_counts,
        fragment_max_n_genes=fragment_max_n_genes,
        min_total_counts=min_total_counts,
        min_n_genes=min_n_genes,
        max_pct_mito=max_pct_mito,
    )

    summary = classify_qc_categories(annotated, thresholds)
    filtered = annotated[annotated.obs["qc_keep"].to_numpy()].copy()

    plot_paths = list(quant.plot_paths)
    if make_plots and plotting_dir is not None:
        metric_sets: list[Tuple[Sequence[str], str]] = []
        if violin_metrics:
            metric_sets.append((violin_metrics, "counts"))
        if violin_percent_metrics:
            metric_sets.append((violin_percent_metrics, "pct"))
        if violin_log_metrics:
            metric_sets.append((violin_log_metrics, "log"))
        annotation_keys = list(dict.fromkeys(list(plot_annotation_keys) + ["qc_category"]))
        scatter_keys = (
            list(dict.fromkeys(scatter_annotation_keys))
            if scatter_annotation_keys is not None
            else (annotation_keys if annotation_keys else (None,))
        )
        plot_paths.extend(
            _generate_qc_plots(
                annotated,
                plotting_dir,
                prefix="classified_all_cells",
                metric_sets=metric_sets,
                annotation_keys=annotation_keys,
                scatter_color_keys=scatter_keys,
                sample_fraction=scatter_sample_fraction,
                random_state=random_state,
                pairplot_sample_fraction=pairplot_sample_fraction,
                pairplot_max_cells=pairplot_max_cells,
            )
        )
        plot_paths.extend(
            _generate_qc_plots(
                filtered,
                plotting_dir,
                prefix="classified_filtered_cells",
                metric_sets=metric_sets,
                annotation_keys=annotation_keys,
                scatter_color_keys=scatter_keys,
                sample_fraction=scatter_sample_fraction,
                random_state=random_state,
                pairplot_sample_fraction=pairplot_sample_fraction,
                pairplot_max_cells=pairplot_max_cells,
            )
        )

    return QCFilteringResult(
        adata=annotated,
        filtered_adata=filtered,
        thresholds=thresholds,
        summary=summary,
        plot_paths=tuple(plot_paths),
        extra_columns=("qc_category", "qc_keep"),
    )


def run_default_workflow(
    input_h5ad: pathlib.Path = pathlib.Path("data/anndata.h5ad"),
    output_filtered_h5ad: pathlib.Path = pathlib.Path("data/anndata_cleaned.h5ad"),
    *,
    plotting_dir: pathlib.Path = pathlib.Path("figures/qc"),
) -> QCFilteringResult:
    """
    Convenience wrapper mirroring the behaviour of the ST941C script.

    Returns the result dataclass produced by :func:`perform_qc_and_filtering`
    and writes the filtered AnnData back to disk.
    """
    adata = ad.read_h5ad(str(input_h5ad))

    result = perform_qc_and_filtering(adata, plotting_dir=plotting_dir)
    result.filtered_adata.write_h5ad(str(output_filtered_h5ad))

    return result


if __name__ == "__main__":
    run_default_workflow()
