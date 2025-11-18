"""Shared data structures for differential expression workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

from importlib import resources
import re

import pandas as pd
from matplotlib import pyplot as plt

from .plots import (
    plot_pathway_density_difference,
    pathway_scurve_plot,
    plot_de_volcano,
    plot_pathway_volcano,
    plot_pathway_enrichment_heatmap,
)

GENE_ANNOTATIONS_FILENAME = "ensg_annotations_abbreviated.txt"


@dataclass
class PseudobulkResult:
    """Container for pseudobulk expression matrices and metadata."""

    counts: pd.DataFrame
    metadata: pd.DataFrame
    parameters: Mapping[str, Any] = field(default_factory=dict)
    graph_summary: Optional[Mapping[str, Any]] = None

    def to_dataframe(
        self,
        prefix: str = "",
        columns: Optional[Sequence[str]] = None,
        include_metadata: bool = True,
    ) -> pd.DataFrame:
        """
        Return a tidy DataFrame representation suitable for downstream joins.

        Parameters
        ----------
        prefix:
            Optional prefix applied to column names.
        columns:
            Subset of count columns to retain (defaults to all).
        include_metadata:
            When True, concatenate metadata columns alongside counts.
        """
        counts = self.counts if columns is None else self.counts.loc[:, list(columns)]
        if prefix:
            counts = counts.add_prefix(prefix)
        if include_metadata:
            meta = self.metadata.copy()
            if prefix:
                meta = meta.add_prefix(f"{prefix}meta_" if not prefix.endswith("_") else f"{prefix}meta")
            merged = pd.concat([meta, counts], axis=1)
            return merged
        return counts


@dataclass
class DEAnalysisResult:
    """Structured output for differential expression runs."""

    dds: Any
    contrast_results: Mapping[str, pd.DataFrame]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    design_columns: Optional[Sequence[str]] = None
    artifacts: Optional[MutableMapping[str, Any]] = None

    def get_summary(self, key: str) -> pd.DataFrame:
        """Return the results DataFrame for a specific contrast."""
        try:
            return self.contrast_results[key]
        except KeyError as exc:
            raise KeyError(f"Contrast '{key}' not found.") from exc
    @property
    def available_contrasts(self) -> List[str]:
        """List of valid contrast identifiers."""
        return list(self.contrast_results.keys())

    def get_contrast_df(self, key: str, *, copy: bool = False) -> pd.DataFrame:
        """
        Retrieve the differential expression dataframe for a contrast.

        Parameters
        ----------
        key:
            Contrast identifier.
        copy:
            When True, return a defensive copy to avoid mutating cached results.
        """
        try:
            df = self.contrast_results[key]
        except KeyError as exc:
            available = ", ".join(sorted(self.contrast_results))
            raise KeyError(
                f"Contrast '{key}' not found. Available contrasts: {available or '∅'}."
            ) from exc
        return df.copy() if copy else df

    def save_volcano_plots(
        self,
        *,
        contrasts: Optional[Sequence[str]] = None,
        output_dir: Union[str, Path],
        file_prefix: Optional[str] = None,
        genes_of_interest: Optional[Sequence[str]] = None,
        alpha: float = 0.05,
        x_col: str = "log2FoldChange",
        p_col: str = "padj",
        dpi: int = 300,
        fig_size: Tuple[float, float] = (8, 6),
        logger: Optional[Callable[[str], None]] = None,
    ) -> List[Path]:
        """
        Save volcano plots for one or more contrasts using :func:`plot_de_volcano`.
        """
        selected = list(contrasts) if contrasts is not None else self.available_contrasts
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = file_prefix or "de_volcano"

        saved: List[Path] = []
        for contrast in selected:
            df = self.get_contrast_df(contrast, copy=True)
            fig, _ = plot_de_volcano(
                df,
                genes_of_interest=genes_of_interest,
                alpha=alpha,
                x_col=x_col,
                p_col=p_col,
                figsize=fig_size,
            )
            filename = f"{prefix}_{PathwayEnrichmentResult._sanitize_fragment(contrast)}.png"
            destination = out_dir / filename
            PathwayEnrichmentResult._save_figure(fig, destination, dpi=dpi, logger=logger)
            saved.append(destination)
        return saved


@dataclass
class PathwayEnrichmentResult:
    """Container for pathway enrichment outputs."""

    per_contrast: Mapping[str, pd.DataFrame]
    libraries: Sequence[str]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    concatenated: Optional[pd.DataFrame] = None

    def tidy(self) -> pd.DataFrame:
        """Return a concatenated long-form DataFrame (compute if needed)."""
        if self.concatenated is not None:
            return self.concatenated
        frames = []
        for contrast, df in self.per_contrast.items():
            temp = df.copy()
            temp.insert(0, "contrast", contrast)
            frames.append(temp)
        if frames:
            self.concatenated = pd.concat(frames, ignore_index=True)
        else:
            self.concatenated = pd.DataFrame()
        return self.concatenated

    @property
    def available_contrasts(self) -> List[str]:
        """List of contrasts with enrichment tables."""
        return list(self.per_contrast.keys())

    def get_contrast_df(self, key: str, *, copy: bool = False) -> pd.DataFrame:
        """
        Retrieve the pathway enrichment dataframe for a contrast.

        Parameters
        ----------
        key:
            Contrast identifier.
        copy:
            When True, return a defensive copy to avoid mutating cached results.
        """
        try:
            df = self.per_contrast[key]
        except KeyError as exc:
            available = ", ".join(sorted(self.per_contrast))
            raise KeyError(
                f"Contrast '{key}' not found. Available contrasts: {available or '∅'}."
            ) from exc
        return df.copy() if copy else df

    def save_density_difference_panels(
        self,
        de_result: "DEAnalysisResult",
        contrast: str,
        pathways: Sequence[str],
        *,
        output_dir: Union[str, Path],
        file_prefix: Optional[str] = None,
        stat_col: str = "stat",
        enrichment_col: str = "enrichment_t",
        gene_name_col: str = "gene_name",
        path_gene_col: str = "nom_sig_genes",
        dpi: int = 300,
        fig_size: Tuple[float, float] = (8, 5),
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> List[Path]:
        """
        Render and save density-difference panels for a cluster-vs-all contrast.
        """
        out_dir = self._ensure_output_dir(output_dir)
        de_df = de_result.get_contrast_df(contrast, copy=True)
        path_df = self.get_contrast_df(contrast, copy=True)

        prefix = file_prefix or self._sanitize_fragment(contrast)
        return self._render_density_panels(
            de_df=de_df,
            path_df=path_df,
            pathways=pathways,
            filename_prefix=prefix,
            out_dir=out_dir,
            stat_col=stat_col,
            enrichment_col=enrichment_col,
            gene_name_col=gene_name_col,
            path_gene_col=path_gene_col,
            dpi=dpi,
            fig_size=fig_size,
            x_min=x_min,
            x_max=x_max,
            logger=logger,
        )

    def save_pairwise_density_panels(
        self,
        de_result: "DEAnalysisResult",
        primary_cluster: str,
        comparator_cluster: str,
        *,
        pathways: Sequence[str],
        output_dir: Union[str, Path],
        file_prefix: Optional[str] = None,
        stat_col: str = "stat",
        enrichment_col: str = "enrichment_t",
        gene_name_col: str = "gene_name",
        path_gene_col: str = "nom_sig_genes",
        columns_to_flip: Sequence[str] = ("log2FoldChange", "stat"),
        pathway_columns_to_flip: Sequence[str] = ("mean_t", "enrichment_t", "signed_neglog10_BH"),
        dpi: int = 300,
        fig_size: Tuple[float, float] = (8, 5),
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> List[Path]:
        """
        Render pairwise pathway density panels, reorienting reverse contrasts when needed.
        """
        out_dir = self._ensure_output_dir(output_dir)
        forward_key = f"{primary_cluster}_vs_{comparator_cluster}"
        reverse_key = f"{comparator_cluster}_vs_{primary_cluster}"
        orientation = "forward"

        if forward_key in self.per_contrast and forward_key in de_result.contrast_results:
            de_df = de_result.get_contrast_df(forward_key, copy=True)
            path_df = self.get_contrast_df(forward_key, copy=True)
        elif reverse_key in self.per_contrast and reverse_key in de_result.contrast_results:
            de_df = self._invert_directional_columns(
                de_result.get_contrast_df(reverse_key, copy=True),
                columns_to_flip,
            )
            path_df = self._invert_directional_columns(
                self.get_contrast_df(reverse_key, copy=True),
                pathway_columns_to_flip,
            )
            orientation = "reverse"
        else:
            if logger:
                logger(
                    f"No pairwise contrast found for {primary_cluster} vs {comparator_cluster}."
                )
            return []

        prefix_root = file_prefix or (
            f"{self._sanitize_fragment(primary_cluster)}_vs_{self._sanitize_fragment(comparator_cluster)}"
        )
        if logger and orientation == "reverse":
            logger(
                f"Using reverse contrast '{reverse_key}' for orientation "
                f"{primary_cluster} vs {comparator_cluster}."
            )

        return self._render_density_panels(
            de_df=de_df,
            path_df=path_df,
            pathways=pathways,
            filename_prefix=prefix_root,
            out_dir=out_dir,
            stat_col=stat_col,
            enrichment_col=enrichment_col,
            gene_name_col=gene_name_col,
            path_gene_col=path_gene_col,
            dpi=dpi,
            fig_size=fig_size,
            x_min=x_min,
            x_max=x_max,
            logger=logger,
        )

    @staticmethod
    def _sanitize_fragment(fragment: str) -> str:
        clean = re.sub(r"[^A-Za-z0-9._-]+", "_", fragment.strip())
        clean = re.sub(r"_+", "_", clean).strip("_")
        return clean or "pathway"

    @staticmethod
    def _ensure_output_dir(output_dir: Union[str, Path]) -> Path:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _invert_directional_columns(
        df: pd.DataFrame,
        columns: Iterable[str],
    ) -> pd.DataFrame:
        adjusted = df.copy()
        for col in columns:
            if col in adjusted.columns:
                adjusted[col] = -adjusted[col]
        return adjusted

    def _render_density_panels(
        self,
        *,
        de_df: pd.DataFrame,
        path_df: pd.DataFrame,
        pathways: Sequence[str],
        filename_prefix: str,
        out_dir: Path,
        stat_col: str,
        enrichment_col: str,
        gene_name_col: str,
        path_gene_col: str,
        dpi: int,
        fig_size: Tuple[float, float],
        x_min: Optional[float],
        x_max: Optional[float],
        logger: Optional[Callable[[str], None]],
    ) -> List[Path]:
        if "pathway" not in path_df.columns:
            raise KeyError("Pathway dataframe is missing required 'pathway' column.")

        available_pathways = set(path_df["pathway"].astype(str))
        saved_paths: List[Path] = []
        for pathway in pathways:
            if pathway not in available_pathways:
                if logger:
                    logger(f"Pathway '{pathway}' not found in enrichment results; skipping.")
                continue
            fig, _ = plot_pathway_density_difference(
                de_df=de_df,
                path_df=path_df,
                pathway_name=pathway,
                enrichment_col=enrichment_col,
                stat_col=stat_col,
                gene_name_col=gene_name_col,
                path_gene_col=path_gene_col,
                fig_size=fig_size,
                x_min=x_min,
                x_max=x_max,
            )
            filename = f"{filename_prefix}_{self._sanitize_fragment(pathway)}.png"
            destination = out_dir / filename
            self._save_figure(
                fig,
                destination,
                dpi=dpi,
                logger=logger,
                message=f"Saved pathway density plot to {destination}",
            )
            saved_paths.append(destination)
        return saved_paths

    def save_pathway_scurves(
        self,
        contrast: str,
        pathways: Sequence[str],
        *,
        output_dir: Union[str, Path],
        file_prefix: Optional[str] = None,
        dpi: int = 300,
        fig_size: Tuple[float, float] = (10, 6),
        alpha: float = 0.05,
        highlight_only: bool = False,
        logger: Optional[Callable[[str], None]] = None,
    ) -> List[Path]:
        """
        Save pathway S-curve plots for the requested pathways.
        """
        out_dir = self._ensure_output_dir(output_dir)
        df = self.get_contrast_df(contrast, copy=True)
        if "pathway" not in df.columns:
            raise KeyError("Pathway dataframe is missing required 'pathway' column.")
        available = set(df["pathway"].astype(str))

        prefix = file_prefix or f"{self._sanitize_fragment(contrast)}_scurve"
        saved: List[Path] = []
        for pathway in pathways:
            if pathway not in available:
                if logger:
                    logger(f"Pathway '{pathway}' not found in enrichment results; skipping.")
                continue
            fig, _ = pathway_scurve_plot(
                df,
                codes=[pathway],
                alpha=alpha,
                title=f"{contrast} – {pathway}",
                figsize=fig_size,
                highlight_pathways=[pathway],
                highlight_only=highlight_only,
            )
            filename = f"{prefix}_{self._sanitize_fragment(pathway)}.png"
            destination = out_dir / filename
            self._save_figure(
                fig,
                destination,
                dpi=dpi,
                logger=logger,
                message=f"Saved pathway s-curve plot to {destination}",
            )
            saved.append(destination)
        return saved

    def save_pathway_volcano_plots(
        self,
        *,
        contrasts: Optional[Sequence[str]] = None,
        output_dir: Union[str, Path],
        file_prefix: Optional[str] = None,
        pathways_of_interest: Optional[Sequence[str]] = None,
        alpha: float = 0.05,
        x_col: str = "enrichment_t",
        p_col: str = "BH_adj_p",
        dpi: int = 300,
        fig_size: Tuple[float, float] = (8, 6),
        logger: Optional[Callable[[str], None]] = None,
    ) -> List[Path]:
        """
        Save pathway volcano plots for one or more contrasts.
        """
        selected = list(contrasts) if contrasts is not None else self.available_contrasts
        out_dir = self._ensure_output_dir(output_dir)
        prefix = file_prefix or "pathway_volcano"

        saved: List[Path] = []
        for contrast in selected:
            df = self.get_contrast_df(contrast, copy=True)
            fig, _ = plot_pathway_volcano(
                df,
                pathways_of_interest=pathways_of_interest,
                alpha=alpha,
                x_col=x_col,
                p_col=p_col,
                title=f"{contrast} pathway volcano",
                figsize=fig_size,
            )
            filename = f"{prefix}_{self._sanitize_fragment(contrast)}.png"
            destination = out_dir / filename
            self._save_figure(
                fig,
                destination,
                dpi=dpi,
                logger=logger,
                message=f"Saved pathway volcano plot to {destination}",
            )
            saved.append(destination)
        return saved

    @staticmethod
    def _save_figure(
        fig: plt.Figure,
        destination: Path,
        *,
        dpi: int,
        logger: Optional[Callable[[str], None]] = None,
        message: Optional[str] = None,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        if logger:
            logger(message or f"Saved plot to {destination}")

    def plot_pathway_enrichment_heatmap(
        self,
        pathways: Sequence[str],
        *,
        stat_column: str = "signed_neglog10_BH",
        contrast_prefix: Optional[str] = "cluster_prop__",
        contrast_filter: Optional[Callable[[str], bool]] = None,
        transpose: bool = False,
        zscore_axis: str = "columns",
        zscore: bool = True,
        clustering_metric: str = "cosine",
        figsize: Tuple[float, float] = (10, 6),
        cmap: str = "coolwarm",
        annot: bool = False,
        annot_fmt: str = ".2f",
        logger: Optional[Callable[[str], None]] = None,
        return_fig: bool = True,
        out_path: Optional[Union[str, Path]] = None,
        cbar_label: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot a heatmap of pathway enrichment across cluster-vs-all contrasts.

        Parameters
        ----------
        pathways:
            Iterable of pathway identifiers to include.
        stat_column:
            Column from enrichment tables to visualise (default ``signed_neglog10_BH``).
        contrast_prefix:
            Optional prefix filter for contrasts (default ``cluster_prop__``). Ignored if
            ``contrast_filter`` provided.
        contrast_filter:
            Callable taking a contrast name and returning bool; supersedes prefix.
        transpose:
            When True, swap axes (pathways on rows, clusters on columns).
        zscore_axis:
            Axis along which to z-score. ``"columns"`` (default) normalises pathways.
        zscore:
            Whether to z-score values prior to clustering (default True).
        clustering_metric:
            Distance metric for clustering (default cosine).
        figsize, cmap, annot, annot_fmt:
            Matplotlib/Seaborn stylistic options.
        logger:
            Optional callable for status messages.
        return_fig:
            When True (default) return ``(fig, ax)`` from the heatmap.
        out_path:
            If provided, save the heatmap to this path.
        cbar_label:
            Optional colorbar label. Defaults to ``stat_column``.
        """
        if not pathways:
            raise ValueError("No pathways provided.")
        contrasts = []
        for contrast in self.available_contrasts:
            if contrast_filter:
                if contrast_filter(contrast):
                    contrasts.append(contrast)
            elif contrast_prefix is None or contrast.startswith(contrast_prefix):
                contrasts.append(contrast)
        if not contrasts:
            raise ValueError("No contrasts matched the provided filters.")

        pathways = list(pathways)
        data_frames = []
        for contrast in contrasts:
            df = self.get_contrast_df(contrast, copy=True)
            if stat_column not in df.columns:
                raise KeyError(f"Column '{stat_column}' missing in contrast '{contrast}'.")
            subset = df[df["pathway"].isin(pathways)][["pathway", stat_column]].copy()
            subset["contrast"] = contrast
            data_frames.append(subset)

        heatmap_df = (
            pd.concat(data_frames, ignore_index=True)
            .pivot(index="contrast", columns="pathway", values=stat_column)
        )

        if heatmap_df.isnull().all(axis=0).any():
            missing_cols = heatmap_df.columns[heatmap_df.isnull().all(axis=0)].tolist()
            if logger:
                logger(f"Dropping pathways with no data: {missing_cols}")
            heatmap_df = heatmap_df.drop(columns=missing_cols)
        if heatmap_df.isnull().all(axis=1).any():
            missing_rows = heatmap_df.index[heatmap_df.isnull().all(axis=1)].tolist()
            if logger:
                logger(f"Dropping contrasts with no data: {missing_rows}")
            heatmap_df = heatmap_df.drop(index=missing_rows)

        if heatmap_df.empty:
            raise ValueError("No data available after filtering pathways and contrasts.")

        # We always want to normalise along the pathway axis. If the heatmap is
        # transposed (pathways on rows), z-score across rows; otherwise across columns.
        effective_axis = "rows" if transpose else zscore_axis

        fig, ax = plot_pathway_enrichment_heatmap(
            heatmap_df,
            original_values=heatmap_df,
            transpose=transpose,
            zscore_axis=effective_axis,
            zscore=zscore,
            clustering_metric=clustering_metric,
            figsize=figsize,
            cmap=cmap,
            annot=annot,
            annot_fmt=annot_fmt,
            logger=logger,
            out_path=out_path,
            cbar_label=cbar_label or stat_column,
        )
        if not return_fig:
            plt.close(fig)
        return fig, ax


def load_default_gene_annotations() -> pd.DataFrame:
    """
    Load the packaged ENSG annotations and return a standardized DataFrame.

    The returned frame is indexed by Ensembl gene IDs (`gene_id`) and always
    exposes at least `gene_id`, `gene_name`, `chromosome`, and
    `gene_description` columns so downstream DE helpers can rely on a stable
    schema.
    """
    try:
        data_path = resources.files("sc_robust.data").joinpath(GENE_ANNOTATIONS_FILENAME)
    except AttributeError as exc:
        raise FileNotFoundError(
            f"Could not locate packaged gene annotations '{GENE_ANNOTATIONS_FILENAME}'. "
            "Verify that the resource is included in the installed distribution."
        ) from exc

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Packaged gene annotation file '{GENE_ANNOTATIONS_FILENAME}' is missing. "
            "Reinstall the package or ensure setup.py includes the data file."
        )

    with data_path.open("rb") as handle:
        df = pd.read_csv(handle, sep="\t")

    rename_map = {
        "Gene stable ID": "gene_id",
        "Chromosome/scaffold name": "chromosome",
        "Gene name": "gene_name",
        "Gene description": "gene_description",
    }
    missing = [orig for orig in rename_map if orig not in df.columns]
    if missing:
        raise KeyError(
            f"Packaged gene annotation file is missing expected columns: {missing}. "
            "Ensure the resource matches the documented schema."
        )

    df = df.rename(columns=rename_map)
    df = df.drop_duplicates(subset="gene_id", keep="first")
    df = df.set_index("gene_id", drop=False)
    df.index = df.index.astype(str)
    df["gene_name"] = df["gene_name"].astype(str)
    return df
