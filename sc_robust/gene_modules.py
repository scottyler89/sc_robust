from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpearmanArtifact:
    path: Path
    feature_ids_kept: List[str]
    cpos: float
    cneg: float
    provenance_json: Optional[str]
    schema_version: Optional[str]

    @property
    def n_features(self) -> int:
        return int(len(self.feature_ids_kept))

    @property
    def provenance_sha256(self) -> Optional[str]:
        if self.provenance_json is None:
            return None
        return hashlib.sha256(self.provenance_json.encode("utf-8", errors="ignore")).hexdigest()


def _safe_json_loads(value: Optional[str]) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def read_spearman_h5(path: Union[str, Path]) -> SpearmanArtifact:
    """Read metadata required for module discovery from a `spearman.hdf5` file.

    Expected keys:
      - dataset: `infile` (matrix)
      - dataset: `ids/feature_ids_kept` (gene IDs in matrix order)
      - dataset: `meta/provenance_json` (JSON string, optional but recommended)
      - attrs: `infile`.attrs['Cpos'], `infile`.attrs['Cneg']
    """
    import h5py  # local import: keep optionality for users who don't use this feature

    path = Path(path)
    with h5py.File(path, "r") as h5:
        if "infile" not in h5:
            raise KeyError(f"{path} missing required dataset 'infile'")
        if "ids/feature_ids_kept" not in h5:
            raise KeyError(f"{path} missing required dataset 'ids/feature_ids_kept'")
        ids_raw = h5["ids/feature_ids_kept"][()]
        feature_ids_kept = [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in ids_raw]

        infile = h5["infile"]
        cpos = float(infile.attrs.get("Cpos", np.nan))
        cneg = float(infile.attrs.get("Cneg", np.nan))

        provenance_json = None
        if "meta/provenance_json" in h5:
            raw = h5["meta/provenance_json"][()]
            if isinstance(raw, (bytes, np.bytes_)):
                provenance_json = raw.decode("utf-8", errors="ignore")
            elif isinstance(raw, str):
                provenance_json = raw
            else:
                provenance_json = str(raw)

    schema_version = None
    prov = _safe_json_loads(provenance_json)
    if isinstance(prov, dict):
        schema_version = prov.get("schema_version") or prov.get("schema", {}).get("version")
        if schema_version is not None:
            schema_version = str(schema_version)

    return SpearmanArtifact(
        path=path,
        feature_ids_kept=feature_ids_kept,
        cpos=cpos,
        cneg=cneg,
        provenance_json=provenance_json,
        schema_version=schema_version,
    )


def index_spearman_scratch_dirs(
    scratch_dirs: Sequence[Union[str, Path]],
    *,
    require_splits: Sequence[str] = ("train", "val"),
) -> pd.DataFrame:
    """Preflight/index scratch dirs containing `spearman.hdf5` per split."""
    rows: List[Dict[str, Any]] = []
    for scratch_dir in scratch_dirs:
        scratch_dir = Path(scratch_dir)
        sample = scratch_dir.name
        for split in require_splits:
            h5_path = scratch_dir / split / "spearman.hdf5"
            if not h5_path.exists():
                continue
            art = read_spearman_h5(h5_path)
            feature_ids_unique = len(set(art.feature_ids_kept)) == len(art.feature_ids_kept)
            rows.append(
                {
                    "sample": sample,
                    "split": split,
                    "path": str(art.path),
                    "n_genes": art.n_features,
                    "schema_version": art.schema_version,
                    "provenance_sha256": art.provenance_sha256,
                    "cpos": art.cpos,
                    "cneg": art.cneg,
                    "feature_ids_unique": bool(feature_ids_unique),
                }
            )
    return pd.DataFrame(rows)


def _iter_thresholded_edges_from_dense_block(
    block: np.ndarray,
    *,
    row_offset: int,
    cpos: float,
    cneg: float,
    upper_triangle_only: bool,
) -> Iterator[Tuple[int, int, float]]:
    if block.ndim != 2:
        raise ValueError("Expected a 2D block")
    n_rows, n_cols = block.shape
    for i_local in range(n_rows):
        i = row_offset + i_local
        row = block[i_local]
        if upper_triangle_only:
            start = i + 1
        else:
            start = 0
        if start >= n_cols:
            continue
        # Fast path: use vectorized index extraction on the slice, avoiding per-element python work.
        row_slice = row[start:]
        pos_idx = np.flatnonzero(row_slice >= cpos)
        if pos_idx.size:
            js = pos_idx + start
            for j in js.tolist():
                yield i, int(j), float(row[int(j)])
        neg_idx = np.flatnonzero(row_slice <= cneg)
        if neg_idx.size:
            js = neg_idx + start
            for j in js.tolist():
                yield i, int(j), float(row[int(j)])


def extract_thresholded_edges(
    spearman_h5: Union[str, Path],
    *,
    chunk_rows: int = 512,
    upper_triangle_only: bool = True,
) -> pd.DataFrame:
    """Extract threshold-passing signed edges from `spearman.hdf5` as an edge list.

    Returns columns:
      - i, j: integer indices into `feature_ids_kept`
      - rho: Spearman rho
      - sign: 'pos' or 'neg'
    """
    art = read_spearman_h5(spearman_h5)
    if not np.isfinite(art.cpos) or not np.isfinite(art.cneg):
        raise ValueError(f"{art.path} missing finite Cpos/Cneg attrs on infile")
    if art.cpos < -1 or art.cpos > 1 or art.cneg < -1 or art.cneg > 1:
        raise ValueError(f"{art.path} has invalid Cpos/Cneg outside [-1,1]: cpos={art.cpos} cneg={art.cneg}")

    import h5py

    rows_i: List[int] = []
    rows_j: List[int] = []
    rows_rho: List[float] = []
    rows_sign: List[str] = []

    with h5py.File(art.path, "r") as h5:
        ds = h5["infile"]
        n = int(ds.shape[0])
        if int(ds.shape[1]) != n:
            raise ValueError(f"{art.path} infile must be square; got {ds.shape}")
        if n != art.n_features:
            # Strict consistency check with ids list
            raise ValueError(f"{art.path} ids/feature_ids_kept length {art.n_features} != infile n {n}")

        for start in range(0, n, int(chunk_rows)):
            end = min(n, start + int(chunk_rows))
            block = ds[start:end, :]
            block = np.asarray(block)
            for i, j, rho in _iter_thresholded_edges_from_dense_block(
                block,
                row_offset=start,
                cpos=art.cpos,
                cneg=art.cneg,
                upper_triangle_only=upper_triangle_only,
            ):
                rows_i.append(i)
                rows_j.append(j)
                rows_rho.append(rho)
                rows_sign.append("pos" if rho >= art.cpos else "neg")

    df = pd.DataFrame({"i": rows_i, "j": rows_j, "rho": rows_rho, "sign": rows_sign})
    return df


def edges_to_sparse_adjacency(
    edges: pd.DataFrame,
    *,
    n_nodes: int,
    weight_col: str = "rho",
    symmetric: bool = True,
) -> sp.coo_matrix:
    if edges.empty:
        return sp.coo_matrix((n_nodes, n_nodes), dtype=np.float32)
    i = edges["i"].to_numpy(dtype=np.int64, copy=False)
    j = edges["j"].to_numpy(dtype=np.int64, copy=False)
    w = edges[weight_col].to_numpy(dtype=np.float32, copy=False)
    mat = sp.coo_matrix((w, (i, j)), shape=(n_nodes, n_nodes), dtype=np.float32)
    if symmetric:
        mat = mat + mat.T
    return mat


def run_leiden_gene_modules(
    positive_edges: pd.DataFrame,
    *,
    n_genes: int,
    resolution: float = 1.0,
) -> np.ndarray:
    """Run Leiden community detection on a positive-edge gene graph."""
    from .utils import perform_leiden_clustering

    graph = edges_to_sparse_adjacency(positive_edges, n_nodes=n_genes, symmetric=True)
    _, _, labels = perform_leiden_clustering(graph, resolution_parameter=float(resolution))
    return labels


def _split_edges_by_sign(edges: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if edges.empty:
        return edges.copy(), edges.copy()
    pos = edges[edges["sign"] == "pos"].copy()
    neg = edges[edges["sign"] == "neg"].copy()
    return pos, neg


def _union_split_edges(
    edges_train: pd.DataFrame,
    edges_val: pd.DataFrame,
) -> pd.DataFrame:
    """Union edge lists, recording split presence and mean rho (missing treated as 0)."""
    cols = ["i", "j", "sign", "rho"]
    train = edges_train[cols].rename(columns={"rho": "rho_train"}).copy()
    val = edges_val[cols].rename(columns={"rho": "rho_val"}).copy()

    merged = train.merge(val, on=["i", "j", "sign"], how="outer")
    merged["present_train"] = merged["rho_train"].notna()
    merged["present_val"] = merged["rho_val"].notna()
    merged["rho_train"] = merged["rho_train"].fillna(0.0)
    merged["rho_val"] = merged["rho_val"].fillna(0.0)
    merged["rho_mean"] = (merged["rho_train"] + merged["rho_val"]) / 2.0

    merged["agree_sign"] = True
    both = merged["present_train"] & merged["present_val"]
    merged.loc[both, "agree_sign"] = np.sign(merged.loc[both, "rho_train"]) == np.sign(merged.loc[both, "rho_val"])

    # Sign-safe merge policy: if an edge appears in both but disagrees in sign, drop it.
    merged = merged[~(both & (~merged["agree_sign"]))].copy()

    merged["confidence"] = np.where(both, "both_splits", "single_split")
    return merged


def _default_k_gene(n_genes: int, *, min_k: int = 10, max_k: int = 200) -> int:
    if n_genes <= 1:
        return 0
    k = int(round(np.log(n_genes), 0))
    k = max(int(min_k), k)
    k = min(int(max_k), int(n_genes - 1), k)
    return int(k)


def sparsify_positive_edges_knn(
    positive_edges: pd.DataFrame,
    *,
    n_genes: int,
    k_gene: Optional[int] = None,
    min_k: int = 10,
    max_k: int = 200,
    weight_col: str = "rho_mean",
) -> pd.DataFrame:
    """Degree-control the positive graph via per-gene top-k neighbor selection.

    - Does NOT require mutual-kNN (keeps union of directed selections).
    - Operates on positive edges only; expects columns i, j, and `weight_col`.
    - Returns an undirected edge list with i<j and weight column preserved.
    """
    if positive_edges.empty:
        return positive_edges.copy()
    if k_gene is None:
        k_gene = _default_k_gene(int(n_genes), min_k=int(min_k), max_k=int(max_k))
    k_gene = int(k_gene)
    if k_gene <= 0:
        return positive_edges.iloc[0:0].copy()

    if weight_col not in positive_edges.columns:
        # Fall back to raw rho if union weights not present.
        if "rho" in positive_edges.columns:
            weight_col = "rho"
        else:
            raise KeyError(f"positive_edges missing weight column {weight_col!r}")

    # Expand to directed edges so each node can pick its own top-k neighbors.
    und = positive_edges.copy()
    und = und[und["sign"] == "pos"].copy() if "sign" in und.columns else und
    und = und[["i", "j", weight_col]].copy()

    a = und.rename(columns={"i": "src", "j": "dst"})
    b = und.rename(columns={"j": "src", "i": "dst"})
    directed = pd.concat([a, b], ignore_index=True)

    # Pick top-k by weight per src.
    directed = directed.sort_values([ "src", weight_col], ascending=[True, False], kind="mergesort")
    directed = directed.groupby("src", sort=False, as_index=False).head(k_gene)

    # Convert back to undirected (i<j) and deduplicate.
    src = directed["src"].to_numpy(dtype=np.int64, copy=False)
    dst = directed["dst"].to_numpy(dtype=np.int64, copy=False)
    i = np.minimum(src, dst)
    j = np.maximum(src, dst)
    out = pd.DataFrame({"i": i, "j": j, weight_col: directed[weight_col].to_numpy(dtype=np.float32, copy=False)})
    out = out[out["i"] < out["j"]].copy()
    out = out.groupby(["i", "j"], as_index=False)[weight_col].max()
    out["sign"] = "pos"
    return out


def _write_module_stats_json(
    out_path: Path,
    *,
    artifact_paths: Dict[str, str],
    artifact_meta: Dict[str, Any],
    params: Dict[str, Any],
    labels: np.ndarray,
    positive_edges: pd.DataFrame,
    negative_edges: pd.DataFrame,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels)
    sizes = pd.Series(labels).value_counts().sort_index()

    if not positive_edges.empty:
        li = labels[positive_edges["i"].to_numpy(dtype=np.int64, copy=False)]
        lj = labels[positive_edges["j"].to_numpy(dtype=np.int64, copy=False)]
        intra = positive_edges[li == lj].copy()
        intra["module_id"] = li[li == lj]
        weight_col = "rho_mean" if "rho_mean" in intra.columns else "rho"
        intra_summary = (
            intra.groupby("module_id")[weight_col]
            .agg(["count", "mean", "median"])
            .reset_index()
            .to_dict(orient="records")
        )
    else:
        intra_summary = []

    payload = {
        "artifacts": artifact_paths,
        "source": artifact_meta,
        "params": params,
        "modules": {
            "n_modules": int(sizes.shape[0]),
            "sizes": sizes.to_dict(),
            "intra_positive_edge_summary": intra_summary,
            "n_positive_edges": int(len(positive_edges)),
            "n_negative_edges": int(len(negative_edges)),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def summarize_gene_module_antagonism(
    negative_edges: pd.DataFrame,
    *,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Summarize negative-edge structure between positive gene modules."""
    if negative_edges.empty:
        return pd.DataFrame(
            columns=["module_a", "module_b", "n_edges", "mean_rho", "edge_density", "size_a", "size_b"]
        )
    labels = np.asarray(labels)
    i = negative_edges["i"].to_numpy(dtype=np.int64, copy=False)
    j = negative_edges["j"].to_numpy(dtype=np.int64, copy=False)
    mi = labels[i]
    mj = labels[j]
    mask = mi != mj
    if not np.any(mask):
        return pd.DataFrame(
            columns=["module_a", "module_b", "n_edges", "mean_rho", "edge_density", "size_a", "size_b"]
        )

    mi = mi[mask]
    mj = mj[mask]
    module_a = np.minimum(mi, mj).astype(int)
    module_b = np.maximum(mi, mj).astype(int)
    rho_col = "rho_mean" if "rho_mean" in negative_edges.columns else "rho"
    rho = negative_edges.loc[mask, rho_col].to_numpy(dtype=np.float32, copy=False)

    df = pd.DataFrame({"module_a": module_a, "module_b": module_b, "rho": rho})
    agg = df.groupby(["module_a", "module_b"], as_index=False)["rho"].agg(n_edges="count", mean_rho="mean")

    sizes = pd.Series(labels).value_counts().to_dict()
    agg["size_a"] = agg["module_a"].map(lambda m: int(sizes.get(m, 0)))
    agg["size_b"] = agg["module_b"].map(lambda m: int(sizes.get(m, 0)))
    denom = (agg["size_a"] * agg["size_b"]).replace(0, np.nan)
    agg["edge_density"] = (agg["n_edges"] / denom).astype(np.float32)
    agg["edge_density"] = agg["edge_density"].fillna(0.0)
    return agg


def run_gene_modules_from_scratch_dir(
    scratch_dir: Union[str, Path],
    *,
    split_mode: str = "union",
    chunk_rows: int = 512,
    resolution: float = 1.0,
    k_gene: Optional[int] = None,
    min_k_gene: int = 10,
    max_k_gene: int = 200,
    persist_edges: bool = True,
    out_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    """Turn-key gene module discovery from a `robust(..., scratch_dir=...)` directory."""
    scratch_dir = Path(scratch_dir)
    train_h5 = scratch_dir / "train" / "spearman.hdf5"
    val_h5 = scratch_dir / "val" / "spearman.hdf5"
    if not train_h5.exists() or not val_h5.exists():
        raise FileNotFoundError(f"Expected train/val spearman.hdf5 under {scratch_dir}")

    out_dir = Path(out_dir) if out_dir is not None else scratch_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    art_train = read_spearman_h5(train_h5)
    art_val = read_spearman_h5(val_h5)
    if art_train.feature_ids_kept != art_val.feature_ids_kept:
        raise ValueError("train/val feature_ids_kept differ; align feature selection before module discovery.")

    edges_train = extract_thresholded_edges(train_h5, chunk_rows=chunk_rows, upper_triangle_only=True)
    edges_val = extract_thresholded_edges(val_h5, chunk_rows=chunk_rows, upper_triangle_only=True)

    if split_mode not in {"train", "val", "union"}:
        raise ValueError("split_mode must be one of: 'train', 'val', 'union'")

    if split_mode == "train":
        edges = edges_train.copy()
        edge_meta = {"mode": "train"}
    elif split_mode == "val":
        edges = edges_val.copy()
        edge_meta = {"mode": "val"}
    else:
        edges = _union_split_edges(edges_train, edges_val)
        edge_meta = {"mode": "union"}

    pos, neg = _split_edges_by_sign(edges)

    weight_col = "rho_mean" if "rho_mean" in pos.columns else "rho"
    # Degree control on positive graph for module discovery
    pos_knn = sparsify_positive_edges_knn(
        pos,
        n_genes=art_train.n_features,
        k_gene=k_gene,
        min_k=min_k_gene,
        max_k=max_k_gene,
        weight_col=weight_col,
    )
    pos_for_leiden = pos_knn[["i", "j", weight_col]].rename(columns={weight_col: "rho"}).copy()
    pos_for_leiden["sign"] = "pos"

    labels = run_leiden_gene_modules(pos_for_leiden, n_genes=art_train.n_features, resolution=resolution)

    pos_adj = edges_to_sparse_adjacency(pos_for_leiden, n_nodes=art_train.n_features, symmetric=True).tocsr()
    strength = np.asarray(pos_adj.sum(axis=1)).reshape(-1)
    degree = np.asarray((pos_adj != 0).sum(axis=1)).reshape(-1)

    modules_df = pd.DataFrame(
        {
            "gene_id": art_train.feature_ids_kept,
            "module_id": labels.astype(int),
            "degree_pos": degree.astype(int),
            "strength_pos": strength.astype(np.float32),
        }
    )

    modules_path = out_dir / "gene_modules.tsv.gz"
    modules_df.to_csv(modules_path, sep="\t", index=False, compression="gzip")

    edge_pos_path = out_dir / "gene_edges_pos.tsv.gz"
    edge_neg_path = out_dir / "gene_edges_neg.tsv.gz"
    antagonism_path = out_dir / "gene_module_antagonism.tsv.gz"
    if persist_edges:
        pos.to_csv(edge_pos_path, sep="\t", index=False, compression="gzip")
        neg.to_csv(edge_neg_path, sep="\t", index=False, compression="gzip")

    antagonism_df = summarize_gene_module_antagonism(neg, labels=labels)
    antagonism_df.to_csv(antagonism_path, sep="\t", index=False, compression="gzip")

    stats_path = out_dir / "module_stats.json"
    source_meta = {
        "scratch_dir": str(scratch_dir),
        "feature_ids_kept_sha256": hashlib.sha256(
            ("\n".join(art_train.feature_ids_kept)).encode("utf-8", errors="ignore")
        ).hexdigest(),
        "train": {
            "path": str(art_train.path),
            "schema_version": art_train.schema_version,
            "provenance_sha256": art_train.provenance_sha256,
            "cpos": art_train.cpos,
            "cneg": art_train.cneg,
        },
        "val": {
            "path": str(art_val.path),
            "schema_version": art_val.schema_version,
            "provenance_sha256": art_val.provenance_sha256,
            "cpos": art_val.cpos,
            "cneg": art_val.cneg,
        },
        "edges": edge_meta,
    }
    params = {
        "split_mode": split_mode,
        "chunk_rows": int(chunk_rows),
        "resolution": float(resolution),
        "persist_edges": bool(persist_edges),
    }
    params.update({"k_gene": int(k_gene) if k_gene is not None else None, "min_k_gene": int(min_k_gene), "max_k_gene": int(max_k_gene)})
    artifact_paths = {
        "gene_modules_tsv_gz": str(modules_path),
        "gene_edges_pos_tsv_gz": str(edge_pos_path) if persist_edges else None,
        "gene_edges_neg_tsv_gz": str(edge_neg_path) if persist_edges else None,
        "gene_module_antagonism_tsv_gz": str(antagonism_path),
        "module_stats_json": str(stats_path),
    }

    _write_module_stats_json(
        stats_path,
        artifact_paths=artifact_paths,
        artifact_meta=source_meta,
        params=params,
        labels=labels,
        positive_edges=pos,
        negative_edges=neg,
    )

    out_paths: Dict[str, Path] = {
        "gene_modules": modules_path,
        "module_stats": stats_path,
        "gene_module_antagonism": antagonism_path,
    }
    if persist_edges:
        out_paths.update({"gene_edges_pos": edge_pos_path, "gene_edges_neg": edge_neg_path})
    return out_paths
