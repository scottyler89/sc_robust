"""Plotting utilities for differential expression and pathway analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Tuple, List, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.metrics import pairwise_distances

__all__ = [
    "volcano_plot",
    "volcano_plot_with_labels",
    "pathway_scurve_plot",
    "plot_pathway_enrichment",
    "plot_de_volcano",
    "plot_pathway_volcano",
    "plot_pathway_density_difference",
    "plot_pathway_enrichment_heatmap",
]


def _prepare_dataframe(df: pd.DataFrame, required: Sequence[str]) -> pd.DataFrame:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Dataframe missing required columns: {missing}")
    return df.copy()


def volcano_plot(
    df: pd.DataFrame,
    *,
    alpha: float = 0.05,
    x_col: str = "log2FoldChange",
    p_col: str = "padj",
    x_lab: str = "log2 Fold Change",
    y_lab: str = "-log10(padj)",
    title: str = "Volcano Plot",
    figsize: Tuple[float, float] = (8, 6),
    point_size: float = 10.0,
    epsilon: float = 1e-300,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a volcano plot from a differential expression dataframe.
    """
    plot_df = _prepare_dataframe(df, [x_col, p_col])
    plot_df = plot_df.copy()
    plot_df["minusLog10Padj"] = -np.log10(plot_df[p_col] + epsilon)
    plot_df["significant"] = plot_df[p_col] < alpha

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    ax.scatter(
        plot_df.loc[~plot_df["significant"], x_col],
        plot_df.loc[~plot_df["significant"], "minusLog10Padj"],
        c="black",
        s=point_size,
        label="Not significant",
    )
    ax.scatter(
        plot_df.loc[plot_df["significant"], x_col],
        plot_df.loc[plot_df["significant"], "minusLog10Padj"],
        c="red",
        s=point_size,
        label="Significant",
    )
    threshold = -np.log10(alpha)
    ax.axhline(
        threshold,
        color="grey",
        linestyle="dashed",
        linewidth=1,
        label=f"p-adj = {alpha} (-log10: {threshold:.2f})",
    )
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    ax.set_title(title)
    ax.legend()

    if created_fig:
        fig.tight_layout()
    return fig, ax


def volcano_plot_with_labels(
    df: pd.DataFrame,
    genes_of_interest: Iterable[str],
    *,
    var_name_col: str = "gene_name",
    alpha: float = 0.05,
    x_col: str = "log2FoldChange",
    p_col: str = "padj",
    x_lab: str = "log2 Fold Change",
    y_lab: str = "-log10(padj)",
    title: str = "Volcano Plot",
    highlight_only: bool = False,
    figsize: Tuple[float, float] = (8, 6),
    point_size: float = 10.0,
    offset: Tuple[float, float] = (5, 5),
    epsilon: float = 1e-300,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a volcano plot and annotate selected genes of interest.
    """
    plot_df = _prepare_dataframe(df, [var_name_col, x_col, p_col])
    fig, ax = volcano_plot(
        plot_df,
        alpha=alpha,
        x_col=x_col,
        p_col=p_col,
        x_lab=x_lab,
        y_lab=y_lab,
        title=title,
        figsize=figsize,
        point_size=point_size,
        epsilon=epsilon,
        ax=ax,
    )

    for gene in genes_of_interest:
        gene_rows = plot_df[plot_df[var_name_col] == gene]
        if gene_rows.empty:
            continue
        for _, row in gene_rows.iterrows():
            x_val = row[x_col]
            y_val = -np.log10(row[p_col] + epsilon)
            if highlight_only:
                ax.scatter(x_val, y_val, c="blue", s=point_size * 1.5, zorder=3)
            else:
                ax.annotate(
                    gene,
                    xy=(x_val, y_val),
                    xytext=offset,
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    arrowprops=dict(arrowstyle="-", color="blue", lw=1.5, alpha=1.0),
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.5),
                )
    fig.tight_layout()
    return fig, ax


def plot_de_volcano(
    df: pd.DataFrame,
    genes_of_interest: Optional[Iterable[str]] = None,
    *,
    alpha: float = 0.05,
    x_col: str = "log2FoldChange",
    p_col: str = "padj",
    title: str = "Differential Expression Volcano",
    figsize: Tuple[float, float] = (8, 6),
    point_size: float = 10.0,
    offset: Tuple[float, float] = (5, 5),
    highlight_only: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Convenience wrapper around :func:`volcano_plot` for DE results.
    """
    go_list: Optional[List[str]] = None
    if genes_of_interest is not None:
        go_list = list(genes_of_interest)

    if go_list:
        return volcano_plot_with_labels(
            df,
            go_list,
            var_name_col="gene_name",
            alpha=alpha,
            x_col=x_col,
            p_col=p_col,
            title=title,
            figsize=figsize,
            point_size=point_size,
            offset=offset,
            highlight_only=highlight_only,
        )
    return volcano_plot(
        df,
        alpha=alpha,
        x_col=x_col,
        p_col=p_col,
        title=title,
        figsize=figsize,
        point_size=point_size,
    )


def pathway_scurve_plot(
    df: pd.DataFrame,
    *,
    codes: Optional[Iterable[str]] = None,
    alpha: float = 0.05,
    title: str = "Pathway S-curve Plot",
    figsize: Tuple[float, float] = (10, 6),
    point_size: float = 20.0,
    highlight_pathways: Optional[Iterable[str]] = None,
    offset: Tuple[float, float] = (5, 5),
    highlight_only: bool = False,
    enrichment_col: str = "enrichment_t",
    p_col: str = "BH_adj_p",
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot ranked enrichment statistics, highlighting significant pathways.
    """
    required = {"pathway", enrichment_col, p_col}
    plot_df = _prepare_dataframe(df, required)

    if codes is not None and "code" in plot_df.columns:
        plot_df = plot_df[plot_df["code"].isin(list(codes))]

    plot_df = plot_df.copy().sort_values(by=enrichment_col, ascending=True).reset_index(drop=True)
    plot_df["rank"] = np.arange(1, len(plot_df) + 1)
    plot_df["significant"] = plot_df[p_col] < alpha

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    ax.plot(plot_df["rank"], plot_df[enrichment_col], color="grey", alpha=0.7, zorder=1)
    nonsig = plot_df[~plot_df["significant"]]
    ax.scatter(nonsig["rank"], nonsig[enrichment_col], c="black", s=point_size, label="Not significant", zorder=2)
    sig = plot_df[plot_df["significant"]]
    ax.scatter(sig["rank"], sig[enrichment_col], c="red", s=point_size, label="Significant", zorder=3)

    if highlight_pathways is not None:
        for pathway in highlight_pathways:
            subset = plot_df[plot_df["pathway"] == pathway]
            if subset.empty:
                continue
            for _, row in subset.iterrows():
                x_val = row["rank"]
                y_val = row[enrichment_col]
                if highlight_only:
                    ax.scatter(x_val, y_val, color="blue", s=point_size * 1.5, zorder=4)
                else:
                    horiz_offset = (-abs(offset[0]), offset[1]) if y_val >= 0 else (abs(offset[0]), offset[1])
                    ha = "right" if y_val >= 0 else "left"
                    ax.annotate(
                        pathway,
                        xy=(x_val, y_val),
                        xytext=horiz_offset,
                        textcoords="offset points",
                        ha=ha,
                        va="bottom",
                        arrowprops=dict(arrowstyle="-", color="blue", lw=1.5),
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.5),
                    )

    ax.set_xlabel("Rank Order")
    ax.set_ylabel(enrichment_col)
    ax.set_title(title)
    ax.legend()
    if created_fig:
        fig.tight_layout()
    return fig, ax


def plot_pathway_enrichment(
    deg_df: pd.DataFrame,
    path_df: pd.DataFrame,
    pathway_name: str,
    *,
    enrichment_col: str = "enrichment_t",
    stat_col: str = "stat",
    gene_name_col: str = "gene_name",
    path_gene_col: str = "nom_sig_genes",
    out_path: Optional[Path] = None,
    fig_size: Tuple[float, float] = (8, 5),
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot density differences between pathway-member gene statistics and the background.
    """
    if pathway_name not in path_df["pathway"].values:
        raise ValueError(f"Pathway '{pathway_name}' not present in dataframe.")
    if path_gene_col not in path_df.columns:
        raise KeyError(f"Pathway dataframe missing '{path_gene_col}' column.")
    required_deg = {gene_name_col, stat_col}
    deg_df = _prepare_dataframe(deg_df, required_deg)

    pathway_row = path_df[path_df["pathway"] == pathway_name].iloc[0]
    raw_gene_list = [
        gene.strip()
        for gene in str(pathway_row[path_gene_col]).split(",")
        if gene and isinstance(gene, str)
    ]
    background_stats = deg_df[[gene_name_col, stat_col]].dropna()
    pathway_stats = background_stats[background_stats[gene_name_col].isin(raw_gene_list)]
    if pathway_stats.empty:
        raise ValueError(f"No overlapping genes between DEG table and pathway '{pathway_name}'.")

    if x_min is None:
        x_min = min(background_stats[stat_col].min(), pathway_stats[stat_col].min())
    if x_max is None:
        x_max = max(background_stats[stat_col].max(), pathway_stats[stat_col].max())

    x_grid = np.linspace(x_min, x_max, 512)
    kde_bg = gaussian_kde(background_stats[stat_col])
    kde_pw = gaussian_kde(pathway_stats[stat_col])
    y_bg = kde_bg(x_grid)
    y_pw = kde_pw(x_grid)

    bg_area = np.trapz(y_bg, x_grid)
    pw_area = np.trapz(y_pw, x_grid)
    if bg_area > 0:
        y_bg /= bg_area
    if pw_area > 0:
        y_pw /= pw_area
    delta = y_pw - y_bg

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
        created_fig = True
    else:
        fig = ax.figure

    ax.plot(x_grid, y_bg, label="Background", color="#1f77b4", linewidth=2)
    ax.plot(x_grid, y_pw, label="Pathway", color="#d62728", linewidth=2)
    ax.fill_between(x_grid, y_bg, y_pw, where=delta > 0, color="#d62728", alpha=0.2)
    ax.fill_between(x_grid, y_bg, y_pw, where=delta < 0, color="#1f77b4", alpha=0.2)
    ax.axvline(0, color="black", linewidth=0.8)

    mean_t_val = pathway_row.get("mean_t", np.nan)
    ax.axvline(mean_t_val, color="green", linestyle="--", linewidth=1.2, label=f"mean_t: {mean_t_val:.2f}")

    enrichment_t_val = pathway_row.get(enrichment_col, np.nan)
    padj_val = pathway_row.get("BH_adj_p", np.nan)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel(f"Gene statistic ({stat_col})")
    ax.set_ylabel("Density (AUC=1)")
    ax.set_title(f"{pathway_name}\n{enrichment_col}={enrichment_t_val:.2f}, padj={padj_val:.2e}")
    ax.legend(frameon=False)

    if created_fig:
        fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=600)
    return fig, ax


def plot_pathway_enrichment_heatmap(
    matrix: pd.DataFrame,
    *,
    original_values: Optional[pd.DataFrame] = None,
    transpose: bool = False,
    zscore_axis: str = "columns",
    zscore: bool = True,
    clustering_metric: str = "cosine",
    figsize: Tuple[float, float] = (10, 6),
    cmap: str = "coolwarm",
    annot: bool = False,
    annot_fmt: str = ".2f",
    logger: Optional[Callable[[str], None]] = None,
    out_path: Optional[Union[str, Path]] = None,
    cbar_label: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a clustered heatmap of pathway enrichment statistics.

    Parameters
    ----------
    matrix:
        DataFrame indexed by contrasts (rows) and pathways (columns) containing statistics.
    original_values:
        Optional DataFrame with the same shape as ``matrix`` used for annotations (raw values).
    transpose:
        When True, swap axes prior to plotting.
    zscore_axis:
        Axis to normalise: ``"columns"`` (default) z-scores per pathway; ``"rows"`` z-scores per contrast.
    zscore:
        Whether to z-score values before clustering (default True).
    clustering_metric:
        Distance metric used for hierarchical clustering (default cosine).
    figsize, cmap, annot, annot_fmt:
        Matplotlib/Seaborn styling arguments.
    logger:
        Optional callable for diagnostic messages.
    out_path:
        Optional path to save the figure.
    cbar_label:
        Custom colorbar label.
    """
    if matrix.empty:
        raise ValueError("Heatmap matrix is empty.")
    if original_values is not None and original_values.shape != matrix.shape:
        raise ValueError("original_values must have the same shape as matrix.")

    matrix = matrix.copy()
    orig = original_values.copy() if original_values is not None else matrix.copy()

    axis = 0 if zscore_axis.lower().startswith("col") else 1

    def _zscore(series: pd.Series) -> pd.Series:
        mean = series.mean(skipna=True)
        std = series.std(ddof=0, skipna=True)
        if pd.isna(std) or np.isclose(std, 0.0):
            return pd.Series(np.nan, index=series.index)
        return (series - mean) / std

    if zscore:
        zscored = matrix.apply(_zscore, axis=axis)
    else:
        zscored = matrix.copy()

    def _cluster_order(df: pd.DataFrame, cluster_axis: int) -> List[int]:
        axis_df = df if cluster_axis == 0 else df.T
        if axis_df.shape[0] <= 1:
            return list(range(axis_df.shape[0]))
        arr = axis_df.to_numpy()
        arr = np.nan_to_num(arr, nan=0.0)
        dist = pairwise_distances(arr, metric=clustering_metric)
        dist = np.nan_to_num(dist, nan=0.0)
        condensed = squareform(dist, checks=False)
        linkage_matrix = linkage(condensed, method="average")
        order = leaves_list(linkage_matrix)
        return order.tolist()

    row_order = _cluster_order(zscored, cluster_axis=0)
    col_order = _cluster_order(zscored, cluster_axis=1)
    ordered = zscored.iloc[row_order, col_order]
    ordered_orig = orig.iloc[row_order, col_order]

    if transpose:
        ordered = ordered.T
        ordered_orig = ordered_orig.T

    annot_data = ordered_orig.to_numpy() if annot else False

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        ordered,
        ax=ax,
        cmap=cmap,
        center=0.0,
        annot=annot_data,
        fmt=annot_fmt,
        cbar_kws={"label": cbar_label or ("z-scored statistic" if zscore else "statistic")},
        mask=ordered.isnull(),
        annot_kws={"fontsize": 8} if annot else None,
    )
    ax.set_xlabel("Pathway" if not transpose else "Contrast")
    ax.set_ylabel("Contrast" if not transpose else "Pathway")
    ax.set_title("Pathway enrichment heatmap")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        if logger:
            logger(f"Saved heatmap to {out_path}")

    return fig, ax


def plot_pathway_volcano(
    df: pd.DataFrame,
    pathways_of_interest: Optional[Iterable[str]] = None,
    *,
    alpha: float = 0.05,
    x_col: str = "enrichment_t",
    p_col: str = "BH_adj_p",
    title: str = "Pathway Volcano Plot",
    figsize: Tuple[float, float] = (8, 6),
    point_size: float = 10.0,
    offset: Tuple[float, float] = (5, 5),
    highlight_only: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Volcano plot tailored to pathway enrichment tables.
    """
    poi = list(pathways_of_interest) if pathways_of_interest is not None else []
    if poi:
        return volcano_plot_with_labels(
            df,
            poi,
            var_name_col="pathway",
            alpha=alpha,
            x_col=x_col,
            p_col=p_col,
            title=title,
            figsize=figsize,
            point_size=point_size,
            offset=offset,
            highlight_only=highlight_only,
        )
    return volcano_plot(
        df,
        alpha=alpha,
        x_col=x_col,
        p_col=p_col,
        x_lab=x_col,
        y_lab=f"-log10({p_col})",
        title=title,
        figsize=figsize,
        point_size=point_size,
    )


def plot_pathway_density_difference(
    de_df: pd.DataFrame,
    path_df: pd.DataFrame,
    pathway_name: str,
    *,
    enrichment_col: str = "enrichment_t",
    stat_col: str = "stat",
    gene_name_col: str = "gene_name",
    path_gene_col: str = "nom_sig_genes",
    fig_size: Tuple[float, float] = (8, 5),
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    alpha: float = 0.05,
    out_path: Optional[Path] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Wrapper around :func:`plot_pathway_enrichment` with more explicit naming.
    """
    fig, ax = plot_pathway_enrichment(
        deg_df=de_df,
        path_df=path_df,
        pathway_name=pathway_name,
        enrichment_col=enrichment_col,
        stat_col=stat_col,
        gene_name_col=gene_name_col,
        path_gene_col=path_gene_col,
        fig_size=fig_size,
        x_min=x_min,
        x_max=x_max,
        out_path=out_path,
    )
    return fig, ax
