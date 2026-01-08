"""Helpers for loading packaged pathway gene sets."""

from __future__ import annotations

import csv
import os
import multiprocessing as mp
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .base import PathwayEnrichmentResult  # imported for typing/export convenience
from sc_robust.data import pathways as pathways_pkg

__all__ = [
    "list_available_pathway_libraries",
    "resolve_pathway_filename",
    "load_pathway_library",
    "load_multiple_pathway_libraries",
    "run_pathway_enrichment",
    "run_pathway_enrichment_for_clusters",
    "resolve_pathway_libraries",
]


PATHWAY_SOURCE_MAP = {
    "c1": "Positional gene sets",
    "c2": "Curated gene sets",
    "c3": "Regulatory motif gene sets",
    "c4": "Computational gene sets",
    "c5": "Gene ontology gene sets",
    "c6": "Oncogenic signatures",
    "c7": "Immunologic signatures",
    "c8": "Cell type signatures",
    "ccc": "ST custom Consensus cell cycle",
    "ccw": "ST custom Whitfield cell cycle",
}


def list_available_pathway_libraries() -> List[str]:
    """
    Return the list of packaged pathway files.

    The returned list contains file names (e.g., ``c1.all.v2025.1.Hs.symbols.gmt``)
    sorted alphabetically.
    """
    return sorted(pathways_pkg.iter_pathway_files())


def _effective_n_jobs(n_jobs: Optional[int]) -> int:
    total = os.cpu_count() or 1
    if n_jobs is None:
        return 1
    if n_jobs == 0:
        return total
    if n_jobs < 0:
        return max(1, total + 1 + int(n_jobs))
    return max(1, int(n_jobs))


def _infer_pathway_source(code: str) -> str:
    key = code.split(".")[0]
    return PATHWAY_SOURCE_MAP.get(key, "NA")


def _create_process_pool(max_workers: int) -> ProcessPoolExecutor:
    ctx = mp.get_context("spawn")
    return ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)


def _prepare_de_arrays(
    de_table: pd.DataFrame,
    gene_col: str,
    stat_col: str,
    p_col: str,
    significance_col: Optional[str],
) -> Dict[str, Union[np.ndarray, Dict[str, Sequence[int]], bool]]:
    required_cols = {gene_col, stat_col, p_col}
    if significance_col is not None:
        required_cols.add(significance_col)

    missing = [col for col in required_cols if col not in de_table.columns]
    if missing:
        raise KeyError(f"DE table missing required columns: {missing}")

    subset = de_table.loc[:, list(required_cols)].copy()
    subset = subset.dropna(subset=[gene_col, stat_col, p_col])
    subset[gene_col] = subset[gene_col].astype(str)

    genes = subset[gene_col].to_numpy()
    stats = subset[stat_col].to_numpy(dtype=float)
    pvals = subset[p_col].to_numpy(dtype=float)
    if significance_col is None or significance_col == p_col:
        sigvals = pvals
    else:
        sigvals = subset[significance_col].to_numpy(dtype=float)

    gene_to_indices: Dict[str, List[int]] = {}
    for idx, gene in enumerate(genes):
        gene_to_indices.setdefault(gene, []).append(idx)

    return {
        "genes": genes,
        "stats": stats,
        "pvals": pvals,
        "sigvals": sigvals,
        "gene_to_indices": gene_to_indices,
        "use_separate_sig": significance_col is not None and significance_col != p_col,
    }


def _pathway_indices(
    pathways: Dict[str, List[str]],
    gene_to_indices: Dict[str, List[int]],
) -> Dict[str, np.ndarray]:
    index_map: Dict[str, np.ndarray] = {}
    for pathway, genes in pathways.items():
        positions: List[int] = []
        for gene in genes:
            positions.extend(gene_to_indices.get(gene, ()))
        if positions:
            index_map[pathway] = np.array(sorted(positions), dtype=np.int64)
        else:
            index_map[pathway] = np.array([], dtype=np.int64)
    return index_map


def _run_pathway_enrichment_arrays(
    pathways: Dict[str, List[str]],
    prepared: Dict[str, Union[np.ndarray, Dict[str, Sequence[int]], bool]],
    alpha: float,
) -> pd.DataFrame:
    genes = prepared["genes"]  # type: ignore[assignment]
    stats = prepared["stats"]  # type: ignore[assignment]
    pvals = prepared["pvals"]  # type: ignore[assignment]
    sigvals = prepared["sigvals"]  # type: ignore[assignment]
    gene_to_indices = prepared["gene_to_indices"]  # type: ignore[assignment]
    use_separate_sig = prepared["use_separate_sig"]  # type: ignore[assignment]

    index_map = _pathway_indices(pathways, gene_to_indices)

    results: Dict[str, Dict[str, object]] = {}
    for pathway, idxs in index_map.items():
        ref_size = len(pathways.get(pathway, ()))
        if idxs.size == 0:
            results[pathway] = {
                "size": ref_size,
                "mean_t": np.nan,
                "enrichment_t": np.nan,
                "p": np.nan,
                "BH_adj_p": np.nan,
                "signed_neglog10_BH": 0.0,
                "nom_sig_genes": "",
            }
            continue

        member_stats = stats[idxs]
        member_p = pvals[idxs]
        member_sig = sigvals[idxs]
        # mean t statistic
        mean_t = float(np.nanmean(member_stats)) if member_stats.size else np.nan

        if np.isfinite(member_stats).sum() >= 3:
            from scipy.stats import ttest_1samp

            t_stat, p_val = ttest_1samp(member_stats, popmean=0.0, nan_policy="omit")
            p_val = float(p_val) if np.isfinite(p_val) else np.nan
            signed = np.sign(t_stat)

            sig_mask = np.isfinite(member_stats)
            if use_separate_sig:
                sig_mask &= np.isfinite(member_sig)
                sig_mask &= member_sig < alpha
            else:
                sig_mask &= member_p < alpha

            if signed > 0:
                sig_mask &= member_stats > 0
                order = np.argsort(member_stats[sig_mask])[::-1]
            elif signed < 0:
                sig_mask &= member_stats < 0
                order = np.argsort(member_stats[sig_mask])
            else:
                order = np.argsort(member_stats[sig_mask])[::-1]

            if sig_mask.any():
                selected_genes = genes[idxs][sig_mask]
                ordered = selected_genes[order]
                sig_gene_string = ",".join(ordered.tolist())
            else:
                sig_gene_string = ""
        else:
            t_stat = np.nan
            p_val = np.nan
            sig_gene_string = ""

        results[pathway] = {
            "size": ref_size,
            "mean_t": mean_t,
            "enrichment_t": t_stat,
            "p": p_val,
            "nom_sig_genes": sig_gene_string,
        }

    out_df = pd.DataFrame(results).T
    from scipy.stats import false_discovery_control

    p_vals = out_df["p"].to_numpy(dtype=float)
    mask = np.isfinite(p_vals)
    adj = np.ones_like(p_vals)
    if mask.any():
        adj_vals = false_discovery_control(p_vals[mask])
        adj[mask] = adj_vals
    out_df["BH_adj_p"] = adj

    signed = -np.log10(np.clip(out_df["BH_adj_p"].to_numpy(dtype=float), 1e-300, None))
    neg_mask = out_df["enrichment_t"].to_numpy(dtype=float) < 0
    signed[neg_mask] *= -1
    out_df["signed_neglog10_BH"] = signed
    out_df = out_df.sort_values("signed_neglog10_BH", ascending=False)
    if "nom_sig_genes" in out_df.columns:
        ordered_cols = [col for col in out_df.columns if col != "nom_sig_genes"] + ["nom_sig_genes"]
        out_df = out_df.loc[:, ordered_cols]
    return out_df


def _enrichment_worker(
    args: Tuple[
        str,
        pd.DataFrame,
        Dict[str, Dict[str, List[str]]],
        str,
        str,
        str,
        Optional[str],
        float,
    ],
) -> Tuple[str, pd.DataFrame]:
    (
        contrast,
        de_df,
        library_gene_sets,
        stat_col,
        gene_col,
        p_col,
        significance_col,
        alpha,
    ) = args

    prepared = _prepare_de_arrays(
        de_df,
        gene_col=gene_col,
        stat_col=stat_col,
        p_col=p_col,
        significance_col=significance_col,
    )

    frames: List[pd.DataFrame] = []
    for lib_name, pathways in library_gene_sets.items():
        enriched = _run_pathway_enrichment_arrays(pathways, prepared, alpha=alpha)
        enriched = enriched.copy()
        code = Path(lib_name).stem
        enriched.insert(0, "code", code)
        enriched.insert(1, "source", _infer_pathway_source(code))
        enriched.insert(2, "pathway", enriched.index)
        frames.append(enriched)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty and "signed_neglog10_BH" in combined.columns:
        combined = combined.sort_values("signed_neglog10_BH", ascending=False, ignore_index=True)
    return contrast, combined


def resolve_pathway_filename(library: str, base_dir: Optional[Path] = None) -> Path:
    """
    Resolve a pathway library identifier to an on-disk Path.

    This helper searches the filesystem (optionally within ``base_dir``) and does
    not consider packaged data. Use :func:`load_pathway_library` to access bundled
    pathway resources.
    """
    candidate = Path(library)
    if candidate.exists():
        return candidate

    if base_dir is not None:
        base_dir = Path(base_dir)
        direct = base_dir / library
        if direct.exists():
            return direct
        if not library.endswith(pathways_pkg.PATHWAY_FILE_SUFFIX):
            matches = list(base_dir.glob(f"{library}*{pathways_pkg.PATHWAY_FILE_SUFFIX}"))
            if matches:
                return matches[0]

    raise FileNotFoundError(
        f"Could not resolve pathway library '{library}' on disk. "
        "Either provide a full path or ensure the file exists under the supplied base_dir."
    )


def _read_gmt(path: Path) -> Dict[str, List[str]]:
    """Parse a GMT pathway file into a mapping of pathway -> gene list."""
    pathways: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            name = row[0].strip()
            genes = [gene.strip() for gene in row[2:] if gene.strip()]
            pathways[name] = genes
    return pathways


@lru_cache(maxsize=None)
def load_pathway_library(
    library: str,
    *,
    base_dir: Optional[Path] = None,
) -> Dict[str, List[str]]:
    """
    Load a pathway library into memory.

    Parameters
    ----------
    library:
        Library identifier or path (see :func:`resolve_pathway_filename`).
    base_dir:
        Optional directory to search before falling back to packaged assets.

    Returns
    -------
    dict
        Mapping from pathway name to list of member genes.
    """
    try:
        resolved = resolve_pathway_filename(library, base_dir=base_dir)
    except FileNotFoundError:
        resolved = None

    if resolved is not None and resolved.exists():
        return _read_gmt(resolved)

    # Fall back to packaged data (matching by filename or prefix)
    packaged_files = list_available_pathway_libraries()
    target_name: Optional[str] = None
    if library in packaged_files:
        target_name = library
    else:
        stem = Path(library).stem
        matches = [name for name in packaged_files if Path(name).stem.startswith(stem)]
        if matches:
            target_name = matches[0]

    if target_name is None:
        raise FileNotFoundError(
            f"Could not locate pathway library '{library}'. Checked base_dir and packaged assets."
        )

    resource = resources.files(pathways_pkg.__name__) / target_name
    with resources.as_file(resource) as tmp_path:
        return _read_gmt(Path(tmp_path))


def load_multiple_pathway_libraries(
    libraries: Iterable[str],
    *,
    base_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Load multiple pathway libraries at once.

    Returns a nested mapping ``{library_id: {pathway_name: genes}}``.
    """
    out: Dict[str, Dict[str, List[str]]] = {}
    for library in libraries:
        out[library] = load_pathway_library(library, base_dir=base_dir)
    return out


def resolve_pathway_libraries(
    libraries: Optional[Iterable[str]],
    *,
    base_dir: Optional[Union[str, Path]] = None,
) -> List[str]:
    """Resolve user-provided library specs into concrete GMT identifiers."""
    packaged = list_available_pathway_libraries()
    resolved: List[str] = []
    base_dir_path = Path(base_dir) if base_dir is not None else None

    def add(value: Union[str, Path]) -> None:
        value_str = str(value)
        if value_str not in resolved:
            resolved.append(value_str)

    if libraries is None:
        for name in packaged:
            add(name)
        return resolved

    for spec in libraries:
        if spec in {"all", "*"}:
            for name in packaged:
                add(name)
            continue

        candidate = Path(spec)
        if candidate.exists():
            add(candidate)
            continue

        if base_dir_path is not None:
            candidate = base_dir_path / spec
            if candidate.exists():
                add(candidate)
                continue

        if spec in packaged:
            add(spec)
            continue

        matches = [name for name in packaged if Path(name).stem.startswith(spec)]
        if matches:
            for name in matches:
                add(name)
            continue

        raise FileNotFoundError(
            f"Could not resolve pathway library specification '{spec}'. "
            "Provide a valid filename, prefix, or set pathway_libraries=None to use packaged defaults."
        )

    return resolved


def run_pathway_enrichment(
    de_table: pd.DataFrame,
    pathways: Dict[str, List[str]],
    *,
    stat_col: str = "stat",
    gene_col: str = "gene_name",
    p_col: str = "pvalue",
    significance_col: Optional[str] = "pvalue",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute pathway enrichment statistics given a DE results table."""
    prepared = _prepare_de_arrays(
        de_table,
        gene_col=gene_col,
        stat_col=stat_col,
        p_col=p_col,
        significance_col=significance_col,
    )
    return _run_pathway_enrichment_arrays(pathways, prepared, alpha=alpha)


def run_pathway_enrichment_for_clusters(
    de_by_cluster: Mapping[str, pd.DataFrame],
    libraries: Optional[Iterable[str]] = None,
    *,
    base_dir: Optional[Union[str, Path]] = None,
    stat_col: str = "stat",
    gene_col: str = "gene_name",
    p_col: str = "pvalue",
    significance_col: Optional[str] = "pvalue",
    alpha: float = 0.05,
    n_jobs: int = 0,
    backend: str = "auto",
) -> PathwayEnrichmentResult:
    """Compute pathway enrichment across multiple contrasts and libraries."""
    resolved_libraries = resolve_pathway_libraries(libraries, base_dir=base_dir)
    if not resolved_libraries:
        return PathwayEnrichmentResult(
            per_contrast={contrast: pd.DataFrame() for contrast in de_by_cluster},
            libraries=[],
            parameters={
                "stat_col": stat_col,
                "gene_col": gene_col,
                "p_col": p_col,
                "significance_col": significance_col,
                "alpha": alpha,
                "n_jobs": 1,
                "backend": "sequential",
            },
            concatenated=pd.DataFrame(),
        )

    base_dir_path = Path(base_dir) if base_dir is not None else None
    library_gene_sets = load_multiple_pathway_libraries(resolved_libraries, base_dir=base_dir_path)
    tasks: List[
        Tuple[
            str,
            pd.DataFrame,
            Dict[str, Dict[str, List[str]]],
            str,
            str,
            str,
            Optional[str],
            float,
        ]
    ] = []
    for contrast, de_df in de_by_cluster.items():
        tasks.append(
            (
                contrast,
                de_df,
                library_gene_sets,
                stat_col,
                gene_col,
                p_col,
                significance_col,
                alpha,
            )
        )

    per_contrast: Dict[str, pd.DataFrame] = {contrast: pd.DataFrame() for contrast in de_by_cluster}

    workers = _effective_n_jobs(n_jobs)
    resolved_backend = backend
    if backend == "auto":
        resolved_backend = "process" if workers > 1 and len(tasks) > 1 else "thread"

    runner = _enrichment_worker
    if workers == 1 or len(tasks) <= 1:
        resolved_backend = "sequential"
        for args in tasks:
            contrast, enriched = runner(args)
            per_contrast[contrast] = enriched
    else:
        if resolved_backend not in {"process", "thread"}:
            raise ValueError("backend must be one of {'auto', 'process', 'thread', 'sequential'}.")
        if resolved_backend == "process":
            try:
                executor = _create_process_pool(workers)
            except (OSError, PermissionError, RuntimeError) as exc:
                warnings.warn(
                    f"Process-based pathway enrichment unavailable ({exc}); "
                    "falling back to thread-based parallelism.",
                    RuntimeWarning,
                )
                resolved_backend = "thread"
                executor = ThreadPoolExecutor(max_workers=workers)
        else:
            executor = ThreadPoolExecutor(max_workers=workers)
        with executor as pool:
            future_map = {
                pool.submit(runner, args): args[0] for args in tasks
            }
            for future in as_completed(future_map):
                contrast, enriched = future.result()
                per_contrast[contrast] = enriched

    concat_frames: List[pd.DataFrame] = []
    for contrast, merged in per_contrast.items():
        if not merged.empty:
            temp = merged.copy()
            temp.insert(0, "contrast", contrast)
            concat_frames.append(temp)

    concatenated = pd.concat(concat_frames, ignore_index=True) if concat_frames else pd.DataFrame()
    return PathwayEnrichmentResult(
        per_contrast=per_contrast,
        libraries=resolved_libraries,
        parameters={
            "stat_col": stat_col,
            "gene_col": gene_col,
            "p_col": p_col,
            "significance_col": significance_col,
            "alpha": alpha,
            "n_jobs": workers,
            "backend": resolved_backend,
        },
        concatenated=concatenated,
    )
