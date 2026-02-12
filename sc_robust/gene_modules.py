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
        # Positive edges
        pos = np.where(row[start:] >= cpos)[0]
        for j_local in pos.tolist():
            j = start + j_local
            yield i, j, float(row[j])
        # Negative edges
        neg = np.where(row[start:] <= cneg)[0]
        for j_local in neg.tolist():
            j = start + j_local
            yield i, j, float(row[j])


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

