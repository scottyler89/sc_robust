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

_DEFAULT_EDGE_WEIGHT_EPS = 1e-6
_DEFAULT_EDGE_WEIGHT_MIN = 1e-3


def _write_report_json(out_path: Path, payload: Dict[str, Any]) -> None:
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _edge_weight_from_excess(
    *,
    rho: np.ndarray,
    sign: np.ndarray,
    cpos: float,
    cneg: float,
    eps: float = _DEFAULT_EDGE_WEIGHT_EPS,
    min_weight: float = _DEFAULT_EDGE_WEIGHT_MIN,
) -> np.ndarray:
    """Compute edge weights as a (rounded) log1p of how far beyond the cutoff we are.

    This is meant to be analogous to the cell-cell KNN graph weighting: we preserve the
    cutoff semantics (include/exclude edges based on Cpos/Cneg) and then map the
    "excess over cutoff" to a smooth, positive scale for kNN degree control and Leiden.

    Notes:
    - Positive edges (sign == 'pos'): excess = max(rho - Cpos, 0), normalized by (1 - Cpos).
    - Negative edges (sign == 'neg'): excess = max(Cneg - rho, 0), normalized by (1 + Cneg).
    - When Cpos==1 or Cneg==-1, the denominator is ~0; we fall back to eps to avoid divide-by-zero.
    """
    rho = np.asarray(rho, dtype=np.float64)
    sign = np.asarray(sign)
    w = np.zeros_like(rho, dtype=np.float64)

    pos_mask = sign == "pos"
    if np.any(pos_mask):
        denom = max(eps, 1.0 - float(cpos))
        excess = rho[pos_mask] - float(cpos)
        excess = np.maximum(excess, 0.0) / denom
        w[pos_mask] = np.log1p(excess + eps)

    neg_mask = sign == "neg"
    if np.any(neg_mask):
        denom = max(eps, 1.0 + float(cneg))  # cneg is negative; (1 + cneg) in (0,1]
        excess = float(cneg) - rho[neg_mask]
        excess = np.maximum(excess, 0.0) / denom
        w[neg_mask] = np.log1p(excess + eps)

    w = np.maximum(w, float(min_weight))
    return np.round(w, 6).astype(np.float32)


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


def _extract_provenance_fields(prov: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Best-effort extraction of stable provenance fields from anticor_features JSON."""
    if not isinstance(prov, dict):
        return {}
    out: Dict[str, Any] = {}
    out["schema_version"] = prov.get("schema_version") or prov.get("schema", {}).get("version")
    # Anticor features teams are iterating; handle a couple likely shapes.
    pr = prov.get("pathway_removal") or prov.get("pathways") or {}
    if isinstance(pr, dict):
        for key in (
            "pathway_source",
            "species",
            "bank_manifest_path",
            "bank_sha256",
            "id_bank_dir",
            "use_live_pathway_lookup",
            "offline_mode",
            "requested_pathways",
            "resolved_pathways",
            "missing_pathways",
        ):
            if key in pr:
                out[f"pathway_removal.{key}"] = pr.get(key)
    return {k: v for k, v in out.items() if v is not None}


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


def run_gene_modules_for_cohort(
    scratch_dirs: Sequence[Union[str, Path]],
    *,
    out_dir: Union[str, Path],
    split_mode: str = "union",
    chunk_rows: int = 512,
    resolution: float = 1.0,
    k_gene: Optional[int] = None,
    min_k_gene: int = 10,
    max_k_gene: int = 200,
    persist_edges: bool = True,
    reuse_existing_edges: bool = True,
) -> pd.DataFrame:
    """Batch runner: process many scratch dirs and write per-sample gene artifacts.

    Returns a manifest dataframe with per-sample status and artifact paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, Any]] = []
    for scratch in scratch_dirs:
        scratch = Path(scratch)
        sample = scratch.name
        sample_out = out_dir / sample
        try:
            # Fast-path: if gene edges already exist and reuse is enabled, avoid re-reading dense HDF5.
            edge_pos_path = sample_out / "gene_edges_pos.tsv.gz"
            edge_neg_path = sample_out / "gene_edges_neg.tsv.gz"
            if reuse_existing_edges and edge_pos_path.exists() and edge_neg_path.exists():
                art_train = read_spearman_h5(scratch / "train" / "spearman.hdf5")
                pos = pd.read_csv(edge_pos_path, sep="\t")
                neg = pd.read_csv(edge_neg_path, sep="\t")
                # Build modules from existing positive edges (respect current k_gene defaults).
                weight_col = "w_mean" if "w_mean" in pos.columns else ("rho_mean" if "rho_mean" in pos.columns else "rho")
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
                sample_out.mkdir(parents=True, exist_ok=True)
                modules_path = sample_out / "gene_modules.tsv.gz"
                modules_df.to_csv(modules_path, sep="\t", index=False, compression="gzip")
                antagonism_path = sample_out / "gene_module_antagonism.tsv.gz"
                antagonism_df = summarize_gene_module_antagonism(neg, labels=labels)
                antagonism_df.to_csv(antagonism_path, sep="\t", index=False, compression="gzip")
                stats_path = sample_out / "module_stats.json"
                artifact_paths = {
                    "gene_modules_tsv_gz": str(modules_path),
                    "gene_edges_pos_tsv_gz": str(edge_pos_path),
                    "gene_edges_neg_tsv_gz": str(edge_neg_path),
                    "gene_module_antagonism_tsv_gz": str(antagonism_path),
                    "module_stats_json": str(stats_path),
                }
                params = {
                    "split_mode": split_mode,
                    "chunk_rows": int(chunk_rows),
                    "resolution": float(resolution),
                    "persist_edges": True,
                    "reuse_existing_edges": True,
                    "k_gene": int(k_gene) if k_gene is not None else None,
                    "min_k_gene": int(min_k_gene),
                    "max_k_gene": int(max_k_gene),
                }
                source_meta = {"scratch_dir": str(scratch), "edges": {"mode": "reused"}}
                _write_module_stats_json(
                    stats_path,
                    artifact_paths=artifact_paths,
                    artifact_meta=source_meta,
                    params=params,
                    labels=labels,
                    positive_edges=pos,
                    negative_edges=neg,
                )
                paths = {"gene_modules": modules_path, "module_stats": stats_path, "gene_edges_pos": edge_pos_path, "gene_edges_neg": edge_neg_path, "gene_module_antagonism": antagonism_path}
            else:
                paths = run_gene_modules_from_scratch_dir(
                    scratch,
                    split_mode=split_mode,
                    chunk_rows=chunk_rows,
                    resolution=resolution,
                    k_gene=k_gene,
                    min_k_gene=min_k_gene,
                    max_k_gene=max_k_gene,
                    persist_edges=persist_edges,
                    out_dir=sample_out,
                )
            # Summaries for the cohort manifest.
            edge_pos = paths.get("gene_edges_pos")
            edge_neg = paths.get("gene_edges_neg")
            n_pos = int(pd.read_csv(edge_pos, sep="\t").shape[0]) if edge_pos and Path(edge_pos).exists() else None
            n_neg = int(pd.read_csv(edge_neg, sep="\t").shape[0]) if edge_neg and Path(edge_neg).exists() else None
            manifest_rows.append(
                {
                    "sample": sample,
                    "scratch_dir": str(scratch),
                    "status": "ok",
                    "n_gene_edges_pos": n_pos,
                    "n_gene_edges_neg": n_neg,
                    **{k: str(v) for k, v in paths.items()},
                }
            )
        except Exception as exc:
            logger.exception("gene_modules cohort failed sample=%s", sample)
            manifest_rows.append(
                {
                    "sample": sample,
                    "scratch_dir": str(scratch),
                    "status": "failed",
                    "error": str(exc),
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "gene_modules_manifest.tsv.gz"
    manifest.to_csv(manifest_path, sep="\t", index=False, compression="gzip")
    schema_path = out_dir / "gene_modules_manifest.schema.json"
    schema = {
        "description": "Row-wise manifest for gene module discovery outputs per sample.",
        "columns": {
            "sample": "Sample identifier (derived from scratch dir name).",
            "scratch_dir": "Input scratch directory containing train/val spearman.hdf5.",
            "status": "ok|failed.",
            "n_gene_edges_pos": "Number of persisted positive gene-gene edges (if available).",
            "n_gene_edges_neg": "Number of persisted negative gene-gene edges (if available).",
            "gene_modules": "Path to gene_modules.tsv.gz",
            "gene_edges_pos": "Path to gene_edges_pos.tsv.gz (optional)",
            "gene_edges_neg": "Path to gene_edges_neg.tsv.gz (optional)",
            "gene_module_antagonism": "Path to gene_module_antagonism.tsv.gz",
            "module_stats": "Path to module_stats.json",
            "error": "Failure text when status=failed.",
        },
    }
    schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _load_sample_gene_modules(sample_out_dir: Path) -> pd.DataFrame:
    path = sample_out_dir / "gene_modules.tsv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path, sep="\t")
    required = {"gene_id", "module_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df[["gene_id", "module_id"]].copy()
    df["module_id"] = df["module_id"].astype(int)
    return df


def _build_module_nodes_from_cohort_outputs(
    out_dir: Union[str, Path],
    *,
    manifest: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    if manifest is None:
        manifest_path = out_dir / "gene_modules_manifest.tsv.gz"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Expected {manifest_path}")
        manifest = pd.read_csv(manifest_path, sep="\t")

    ok = manifest[manifest["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame(columns=["node_id", "sample", "module_id", "n_genes"])

    rows: List[Dict[str, Any]] = []
    for sample in ok["sample"].astype(str).tolist():
        sample_out = out_dir / sample
        mods = _load_sample_gene_modules(sample_out)
        sizes = mods.groupby("module_id", as_index=True)["gene_id"].nunique().to_dict()
        for module_id, n_genes in sizes.items():
            rows.append({"sample": sample, "module_id": int(module_id), "n_genes": int(n_genes)})
    nodes = pd.DataFrame(rows)
    if nodes.empty:
        return pd.DataFrame(columns=["node_id", "sample", "module_id", "n_genes"])
    nodes = nodes.sort_values(["sample", "module_id"]).reset_index(drop=True)
    nodes.insert(0, "node_id", np.arange(nodes.shape[0], dtype=int))
    return nodes


def _compute_module_overlap_edges(
    out_dir: Union[str, Path],
    nodes: pd.DataFrame,
    *,
    jaccard_min: float = 0.3,
    require_different_samples: bool = True,
) -> pd.DataFrame:
    """Compute a weighted module-overlap graph using Jaccard between (sample,module_id) nodes."""
    out_dir = Path(out_dir)
    if nodes.empty:
        return pd.DataFrame(columns=["i", "j", "jaccard", "n_intersection"])

    # Build a map: (sample,module_id) -> node_id
    key_to_node: Dict[Tuple[str, int], int] = {
        (r["sample"], int(r["module_id"])): int(r["node_id"]) for r in nodes.to_dict(orient="records")
    }
    node_sizes = nodes.set_index("node_id")["n_genes"].to_dict()
    node_to_sample = nodes.set_index("node_id")["sample"].to_dict()

    # Inverted index: gene -> list of node_ids where present (typically one per sample)
    gene_to_nodes: Dict[str, List[int]] = {}
    for sample in nodes["sample"].astype(str).unique().tolist():
        mods = _load_sample_gene_modules(out_dir / sample)
        for gene_id, module_id in mods[["gene_id", "module_id"]].itertuples(index=False, name=None):
            node_id = key_to_node.get((sample, int(module_id)))
            if node_id is None:
                continue
            gene_to_nodes.setdefault(str(gene_id), []).append(int(node_id))

    # Count intersections for candidate pairs via shared genes.
    inter_counts: Dict[Tuple[int, int], int] = {}
    for node_list in gene_to_nodes.values():
        if len(node_list) <= 1:
            continue
        uniq = sorted(set(node_list))
        for a_idx, a in enumerate(uniq):
            for b in uniq[a_idx + 1 :]:
                if require_different_samples and node_to_sample.get(a) == node_to_sample.get(b):
                    continue
                key = (a, b) if a < b else (b, a)
                inter_counts[key] = inter_counts.get(key, 0) + 1

    if not inter_counts:
        return pd.DataFrame(columns=["i", "j", "jaccard", "n_intersection"])

    rows = []
    for (a, b), inter in inter_counts.items():
        size_a = int(node_sizes.get(a, 0))
        size_b = int(node_sizes.get(b, 0))
        union = size_a + size_b - int(inter)
        if union <= 0:
            continue
        jac = float(inter) / float(union)
        if jac >= float(jaccard_min):
            rows.append({"i": int(a), "j": int(b), "jaccard": float(jac), "n_intersection": int(inter)})
    return pd.DataFrame(rows)


def run_replicated_gene_modules_for_cohort(
    out_dir: Union[str, Path],
    *,
    jaccard_min: float = 0.3,
    resolution: float = 1.0,
) -> Dict[str, Path]:
    """Cluster per-sample positive modules into cohort-level replicated modules.

    Output is *annotated* with `support_n_samples` so sample-unique structure is preserved
    without splitting into separate reporting formats.
    """
    from .utils import perform_leiden_clustering

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "gene_modules_manifest.tsv.gz"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Expected {manifest_path}")
    manifest = pd.read_csv(manifest_path, sep="\t")

    nodes = _build_module_nodes_from_cohort_outputs(out_dir, manifest=manifest)
    edges = _compute_module_overlap_edges(out_dir, nodes, jaccard_min=jaccard_min, require_different_samples=True)

    if nodes.empty:
        replicated_modules_path = out_dir / "replicated_modules.tsv.gz"
        empty_support = pd.DataFrame(columns=["replicated_module_id", "gene_id", "support_n_samples"])
        empty_support.to_csv(
            replicated_modules_path, sep="\t", index=False, compression="gzip"
        )
        instances_path = out_dir / "replicated_module_instances.tsv.gz"
        pd.DataFrame(columns=["sample", "module_id", "replicated_module_id", "n_genes"]).to_csv(
            instances_path, sep="\t", index=False, compression="gzip"
        )
        report_path = out_dir / "replicated_modules.report.json"
        _write_report_json(
            report_path,
            {
                "artifacts": {
                    "replicated_modules_tsv_gz": str(replicated_modules_path),
                    "replicated_module_instances_tsv_gz": str(instances_path),
                },
                "params": {"jaccard_min": float(jaccard_min), "resolution": float(resolution)},
                "summary": {"n_replicated_modules": 0, "n_gene_rows": 0},
            },
        )
        return {
            "replicated_modules": replicated_modules_path,
            "replicated_module_instances": instances_path,
            "report_json": report_path,
        }

    # Build module-overlap adjacency; fall back to all-singletons if no edges.
    if edges.empty:
        labels = np.arange(nodes.shape[0], dtype=int)
    else:
        graph = edges_to_sparse_adjacency(
            edges.rename(columns={"jaccard": "rho"}), n_nodes=int(nodes.shape[0]), weight_col="rho"
        )
        _, _, labels = perform_leiden_clustering(graph, resolution_parameter=float(resolution))
        labels = np.asarray(labels, dtype=int)

    nodes = nodes.copy()
    nodes["replicated_module_id"] = labels.astype(int)
    instances_path = out_dir / "replicated_module_instances.tsv.gz"
    nodes[["sample", "module_id", "replicated_module_id", "n_genes"]].to_csv(
        instances_path, sep="\t", index=False, compression="gzip"
    )

    # Gene-level support per replicated_module_id.
    ok_samples = manifest.loc[manifest["status"] == "ok", "sample"].astype(str).tolist()
    gene_rows: List[Tuple[int, str, str]] = []  # (rep_mod_id, gene_id, sample)
    for sample in ok_samples:
        sample_out = out_dir / sample
        mods = _load_sample_gene_modules(sample_out)
        module_to_rep = nodes[nodes["sample"] == sample].set_index("module_id")["replicated_module_id"].to_dict()
        for gene_id, module_id in mods[["gene_id", "module_id"]].itertuples(index=False, name=None):
            rep_id = module_to_rep.get(int(module_id))
            if rep_id is None:
                continue
            gene_rows.append((int(rep_id), str(gene_id), str(sample)))

    gene_df = pd.DataFrame(gene_rows, columns=["replicated_module_id", "gene_id", "sample"])
    support = (
        gene_df.drop_duplicates()
        .groupby(["replicated_module_id", "gene_id"], as_index=False)["sample"]
        .nunique()
        .rename(columns={"sample": "support_n_samples"})
    )
    replicated_modules_path = out_dir / "replicated_modules.tsv.gz"
    support.to_csv(replicated_modules_path, sep="\t", index=False, compression="gzip")

    report_path = out_dir / "replicated_modules.report.json"
    _write_report_json(
        report_path,
        {
            "artifacts": {
                "replicated_modules_tsv_gz": str(replicated_modules_path),
                "replicated_module_instances_tsv_gz": str(instances_path),
            },
            "params": {"jaccard_min": float(jaccard_min), "resolution": float(resolution)},
            "summary": {
                "n_replicated_modules": int(support["replicated_module_id"].nunique()),
                "n_gene_rows": int(support.shape[0]),
            },
        },
    )
    return {
        "replicated_modules": replicated_modules_path,
        "replicated_module_instances": instances_path,
        "report_json": report_path,
    }


def _load_sample_module_antagonism(sample_out_dir: Path) -> pd.DataFrame:
    path = sample_out_dir / "gene_module_antagonism.tsv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path, sep="\t")
    required = {"module_a", "module_b", "n_edges", "mean_rho", "edge_density", "size_a", "size_b"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df[list(required)].copy()
    df["module_a"] = df["module_a"].astype(int)
    df["module_b"] = df["module_b"].astype(int)
    df["n_edges"] = df["n_edges"].astype(int)
    df["size_a"] = df["size_a"].astype(int)
    df["size_b"] = df["size_b"].astype(int)
    df["mean_rho"] = df["mean_rho"].astype(np.float32)
    df["edge_density"] = df["edge_density"].astype(np.float32)
    return df


def run_replicated_module_antagonism_for_cohort(
    out_dir: Union[str, Path],
    *,
    replicated_instances_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Aggregate within-sample module antagonism into replicated-module antagonism across samples.

    This strictly uses *negative* edges that were summarized between positive modules, then maps
    sample module IDs -> replicated_module_id using `replicated_module_instances.tsv.gz`.
    """
    out_dir = Path(out_dir)
    manifest_path = out_dir / "gene_modules_manifest.tsv.gz"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Expected {manifest_path}")
    manifest = pd.read_csv(manifest_path, sep="\t")
    ok_samples = manifest.loc[manifest["status"] == "ok", "sample"].astype(str).tolist()

    if replicated_instances_path is None:
        replicated_instances_path = out_dir / "replicated_module_instances.tsv.gz"
    replicated_instances_path = Path(replicated_instances_path)
    if not replicated_instances_path.exists():
        raise FileNotFoundError(f"Expected {replicated_instances_path}")
    inst = pd.read_csv(replicated_instances_path, sep="\t")
    required = {"sample", "module_id", "replicated_module_id"}
    missing = required - set(inst.columns)
    if missing:
        raise ValueError(f"{replicated_instances_path} missing columns: {sorted(missing)}")
    inst["sample"] = inst["sample"].astype(str)
    inst["module_id"] = inst["module_id"].astype(int)
    inst["replicated_module_id"] = inst["replicated_module_id"].astype(int)

    rows: List[Dict[str, Any]] = []
    for sample in ok_samples:
        sample_out = out_dir / sample
        antagonism = _load_sample_module_antagonism(sample_out)
        module_to_rep = inst[inst["sample"] == sample].set_index("module_id")["replicated_module_id"].to_dict()
        if antagonism.empty:
            continue
        rep_a = antagonism["module_a"].map(module_to_rep)
        rep_b = antagonism["module_b"].map(module_to_rep)
        keep = rep_a.notna() & rep_b.notna()
        if not bool(keep.any()):
            continue
        tmp = antagonism.loc[keep].copy()
        tmp["rep_a"] = rep_a.loc[keep].astype(int).to_numpy()
        tmp["rep_b"] = rep_b.loc[keep].astype(int).to_numpy()
        tmp["rep_module_a"] = np.minimum(tmp["rep_a"], tmp["rep_b"]).astype(int)
        tmp["rep_module_b"] = np.maximum(tmp["rep_a"], tmp["rep_b"]).astype(int)
        tmp["sample"] = sample
        rows.append(
            tmp[
                [
                    "sample",
                    "rep_module_a",
                    "rep_module_b",
                    "n_edges",
                    "mean_rho",
                    "edge_density",
                    "size_a",
                    "size_b",
                ]
            ]
        )

    if not rows:
        out_path = out_dir / "replicated_module_antagonism.tsv.gz"
        empty = pd.DataFrame(
            columns=[
                "replicated_module_a",
                "replicated_module_b",
                "n_samples_support",
                "mean_mean_rho",
                "median_mean_rho",
                "mean_edge_density",
                "mean_n_edges",
            ]
        )
        empty.to_csv(out_path, sep="\t", index=False, compression="gzip")
        report_path = out_dir / "replicated_module_antagonism.report.json"
        _write_report_json(
            report_path,
            {
                "artifacts": {"replicated_module_antagonism_tsv_gz": str(out_path)},
                "summary": {"n_edges": 0, "max_samples_support": 0},
            },
        )
        return out_path

    all_df = pd.concat(rows, axis=0, ignore_index=True)
    all_df = all_df.rename(columns={"rep_module_a": "replicated_module_a", "rep_module_b": "replicated_module_b"})

    agg = (
        all_df.groupby(["replicated_module_a", "replicated_module_b"], as_index=False)
        .agg(
            n_samples_support=("sample", "nunique"),
            mean_mean_rho=("mean_rho", "mean"),
            median_mean_rho=("mean_rho", "median"),
            mean_edge_density=("edge_density", "mean"),
            mean_n_edges=("n_edges", "mean"),
        )
        .sort_values(["n_samples_support", "mean_mean_rho"], ascending=[False, True])
        .reset_index(drop=True)
    )
    agg["mean_mean_rho"] = agg["mean_mean_rho"].astype(np.float32)
    agg["median_mean_rho"] = agg["median_mean_rho"].astype(np.float32)
    agg["mean_edge_density"] = agg["mean_edge_density"].astype(np.float32)
    agg["mean_n_edges"] = agg["mean_n_edges"].astype(np.float32)

    out_path = out_dir / "replicated_module_antagonism.tsv.gz"
    agg.to_csv(out_path, sep="\t", index=False, compression="gzip")
    report_path = out_dir / "replicated_module_antagonism.report.json"
    _write_report_json(
        report_path,
        {
            "artifacts": {"replicated_module_antagonism_tsv_gz": str(out_path)},
            "summary": {
                "n_edges": int(agg.shape[0]),
                "max_samples_support": int(agg["n_samples_support"].max()) if not agg.empty else 0,
            },
        },
    )
    return out_path


def run_gene_module_meta_analysis_for_cohort(
    scratch_dirs: Sequence[Union[str, Path]],
    *,
    out_dir: Union[str, Path],
    split_mode: str = "union",
    chunk_rows: int = 512,
    module_resolution: float = 1.0,
    k_gene: Optional[int] = None,
    min_k_gene: int = 10,
    max_k_gene: int = 200,
    persist_edges: bool = True,
    reuse_existing_edges: bool = True,
    replication_jaccard_min: float = 0.3,
    replication_resolution: float = 1.0,
) -> Dict[str, Path]:
    """One-call cohort runner: per-sample modules -> replicated modules -> replicated antagonism."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = run_gene_modules_for_cohort(
        scratch_dirs,
        out_dir=out_dir,
        split_mode=split_mode,
        chunk_rows=chunk_rows,
        resolution=module_resolution,
        k_gene=k_gene,
        min_k_gene=min_k_gene,
        max_k_gene=max_k_gene,
        persist_edges=persist_edges,
        reuse_existing_edges=reuse_existing_edges,
    )
    rep_paths = run_replicated_gene_modules_for_cohort(
        out_dir,
        jaccard_min=replication_jaccard_min,
        resolution=replication_resolution,
    )
    antagonism_path = run_replicated_module_antagonism_for_cohort(
        out_dir,
        replicated_instances_path=rep_paths["replicated_module_instances"],
    )
    report_path = out_dir / "gene_module_meta_analysis.report.json"
    _write_report_json(
        report_path,
        {
            "artifacts": {
                "gene_modules_manifest_tsv_gz": str(out_dir / "gene_modules_manifest.tsv.gz"),
                "replicated_modules_tsv_gz": str(rep_paths["replicated_modules"]),
                "replicated_module_instances_tsv_gz": str(rep_paths["replicated_module_instances"]),
                "replicated_module_antagonism_tsv_gz": str(antagonism_path),
                "replicated_modules_report_json": str(rep_paths.get("report_json", "")),
                "replicated_module_antagonism_report_json": str(out_dir / "replicated_module_antagonism.report.json"),
            },
            "params": {
                "split_mode": split_mode,
                "chunk_rows": int(chunk_rows),
                "module_resolution": float(module_resolution),
                "k_gene": int(k_gene) if k_gene is not None else None,
                "min_k_gene": int(min_k_gene),
                "max_k_gene": int(max_k_gene),
                "persist_edges": bool(persist_edges),
                "reuse_existing_edges": bool(reuse_existing_edges),
                "replication_jaccard_min": float(replication_jaccard_min),
                "replication_resolution": float(replication_resolution),
            },
        },
    )
    return {
        "gene_modules_manifest": out_dir / "gene_modules_manifest.tsv.gz",
        "replicated_modules": rep_paths["replicated_modules"],
        "replicated_module_instances": rep_paths["replicated_module_instances"],
        "replicated_module_antagonism": antagonism_path,
        "report_json": report_path,
    }


def _l2_normalize_rows(mat: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (mat / norms).astype(np.float32)


def compute_cluster_gene_module_scores(
    exprs_cells_by_genes: Union[np.ndarray, sp.spmatrix],
    *,
    gene_ids: Sequence[str],
    cluster_labels: Sequence[Union[int, str]],
    replicated_modules: pd.DataFrame,
    min_support_n_samples: int = 1,
    score: str = "mean",
) -> pd.DataFrame:
    """Compute per-cluster module scores from a gene-module membership table.

    This is intentionally *not* Scanpy-dependent and expects the caller to pass the
    intended expression space (e.g., pf-log output used elsewhere in sc_robust).

    Parameters
    - exprs_cells_by_genes: cells×genes expression (dense or CSR/CSC).
    - gene_ids: length==n_genes, matching exprs column order.
    - cluster_labels: length==n_cells cluster assignment.
    - replicated_modules: DataFrame with at least columns:
        - replicated_module_id (int)
        - gene_id (str)
        - support_n_samples (int)
    - min_support_n_samples: filter genes to those observed in >= this many samples.
    - score:
        - "mean": module score = mean expression of member genes.

    Returns
    - DataFrame with columns: cluster, replicated_module_id, score
    """
    if score != "mean":
        raise ValueError("Only score='mean' is supported currently.")
    X = exprs_cells_by_genes
    n_cells, n_genes = X.shape
    if len(gene_ids) != n_genes:
        raise ValueError("gene_ids must match exprs_cells_by_genes.shape[1].")
    if len(cluster_labels) != n_cells:
        raise ValueError("cluster_labels must match exprs_cells_by_genes.shape[0].")

    mods = replicated_modules.copy()
    required = {"replicated_module_id", "gene_id", "support_n_samples"}
    missing = required - set(mods.columns)
    if missing:
        raise ValueError(f"replicated_modules missing columns: {sorted(missing)}")
    mods = mods[mods["support_n_samples"].astype(int) >= int(min_support_n_samples)].copy()
    if mods.empty:
        return pd.DataFrame(columns=["cluster", "replicated_module_id", "score"])
    mods["replicated_module_id"] = mods["replicated_module_id"].astype(int)
    mods["gene_id"] = mods["gene_id"].astype(str)

    gene_to_idx = {str(g): int(i) for i, g in enumerate(gene_ids)}
    mods["gene_idx"] = mods["gene_id"].map(gene_to_idx)
    mods = mods.dropna(subset=["gene_idx"]).copy()
    if mods.empty:
        return pd.DataFrame(columns=["cluster", "replicated_module_id", "score"])
    mods["gene_idx"] = mods["gene_idx"].astype(int)

    module_ids = sorted(mods["replicated_module_id"].unique().tolist())
    module_to_col = {int(m): int(i) for i, m in enumerate(module_ids)}
    mods["module_col"] = mods["replicated_module_id"].map(module_to_col).astype(int)

    # Build genes×modules membership; use 1/|module| weights so (X @ M) yields per-cell mean.
    sizes = mods.groupby("module_col", as_index=True)["gene_idx"].nunique().to_dict()
    w = mods["module_col"].map(lambda c: 1.0 / float(sizes[int(c)])).to_numpy(dtype=np.float32, copy=False)
    gi = mods["gene_idx"].to_numpy(dtype=np.int64, copy=False)
    mj = mods["module_col"].to_numpy(dtype=np.int64, copy=False)
    membership = sp.coo_matrix((w, (gi, mj)), shape=(n_genes, len(module_ids)), dtype=np.float32).tocsr()

    if sp.issparse(X):
        cell_mod = (X @ membership).astype(np.float32)
        cell_mod = cell_mod.toarray()
    else:
        cell_mod = (np.asarray(X, dtype=np.float32) @ membership.toarray()).astype(np.float32)

    labels = np.asarray(cluster_labels)
    clusters = pd.Series(labels).astype(str).to_numpy()
    df = pd.DataFrame(cell_mod, columns=[str(m) for m in module_ids])
    df.insert(0, "cluster", clusters)
    cluster_means = df.groupby("cluster", as_index=False).mean(numeric_only=True)

    long = cluster_means.melt(id_vars=["cluster"], var_name="replicated_module_id", value_name="score")
    long["replicated_module_id"] = long["replicated_module_id"].astype(int)
    long["score"] = long["score"].astype(np.float32)
    return long


def compute_meta_cluster_module_profiles_cosine(
    cluster_module_scores: pd.DataFrame,
    *,
    cluster_to_meta: Dict[str, str],
) -> pd.DataFrame:
    """Compute cosine-style meta-cluster module profiles from per-cluster module scores.

    The intent is to compare geometry (direction) rather than magnitude:
    - Per cluster: build a module-score vector and L2 normalize it.
    - Per meta-cluster: average the normalized vectors and renormalize.
    """
    required = {"cluster", "replicated_module_id", "score"}
    missing = required - set(cluster_module_scores.columns)
    if missing:
        raise ValueError(f"cluster_module_scores missing columns: {sorted(missing)}")

    df = cluster_module_scores.copy()
    df["cluster"] = df["cluster"].astype(str)
    df["replicated_module_id"] = df["replicated_module_id"].astype(int)
    df["score"] = df["score"].astype(np.float32)
    df["meta_cluster"] = df["cluster"].map(lambda c: cluster_to_meta.get(str(c)))
    df = df.dropna(subset=["meta_cluster"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["meta_cluster", "replicated_module_id", "profile_weight", "n_clusters"])
    df["meta_cluster"] = df["meta_cluster"].astype(str)

    wide = df.pivot_table(
        index="cluster", columns="replicated_module_id", values="score", aggfunc="mean", fill_value=0.0
    ).astype(np.float32)
    wide_norm = _l2_normalize_rows(wide.to_numpy())
    norm_df = pd.DataFrame(wide_norm, index=wide.index, columns=wide.columns)
    norm_df["meta_cluster"] = norm_df.index.to_series().map(lambda c: cluster_to_meta.get(str(c))).astype(str)

    # Average direction per meta-cluster, then renormalize.
    meta_means = norm_df.groupby("meta_cluster", as_index=True).mean(numeric_only=True)
    meta_norm = _l2_normalize_rows(meta_means.to_numpy())
    meta_norm_df = pd.DataFrame(meta_norm, index=meta_means.index, columns=meta_means.columns)
    meta_norm_df["n_clusters"] = norm_df.groupby("meta_cluster").size().astype(int)

    out = meta_norm_df.reset_index().melt(
        id_vars=["meta_cluster", "n_clusters"], var_name="replicated_module_id", value_name="profile_weight"
    )
    out["replicated_module_id"] = out["replicated_module_id"].astype(int)
    out["profile_weight"] = out["profile_weight"].astype(np.float32)
    out["n_clusters"] = out["n_clusters"].astype(int)
    return out


def build_publication_gene_sets(
    *,
    de_table: pd.DataFrame,
    replicated_modules: pd.DataFrame,
    meta_cluster_module_profiles: pd.DataFrame,
    meta_cluster_col: str = "meta_cluster",
    gene_col: str = "gene_id",
    de_p_col: str = "pvalue",
    de_effect_col: str = "stat",
    de_alpha: float = 0.05,
    support_min: int = 1,
    top_modules_per_meta_cluster: int = 5,
    core_max_genes: int = 50,
    extended_max_genes: int = 200,
) -> pd.DataFrame:
    """Create "core" and "extended" marker gene sets per meta-cluster.

    This is a *rubric* helper: it does not run DE, and it does not assume a
    particular DE back-end. It simply intersects:
      1) DE-consistent genes (p<=alpha, effect>0 by default)
      2) membership in top-ranked replicated modules for a meta-cluster
      3) gene-level replication support from `replicated_modules.support_n_samples`

    Returns a long table with columns:
      - meta_cluster
      - tier: core|extended
      - gene_id
      - support_n_samples
      - de_pvalue
      - de_effect
      - replicated_module_id
      - module_profile_weight
    """
    # Validate inputs
    for col in (meta_cluster_col, gene_col, de_p_col, de_effect_col):
        if col not in de_table.columns:
            raise ValueError(f"de_table missing column: {col}")
    for col in ("replicated_module_id", "gene_id", "support_n_samples"):
        if col not in replicated_modules.columns:
            raise ValueError(f"replicated_modules missing column: {col}")
    for col in ("meta_cluster", "replicated_module_id", "profile_weight"):
        if col not in meta_cluster_module_profiles.columns:
            raise ValueError(f"meta_cluster_module_profiles missing column: {col}")

    de = de_table[[meta_cluster_col, gene_col, de_p_col, de_effect_col]].copy()
    de = de.rename(
        columns={
            meta_cluster_col: "meta_cluster",
            gene_col: "gene_id",
            de_p_col: "de_pvalue",
            de_effect_col: "de_effect",
        }
    )
    de["meta_cluster"] = de["meta_cluster"].astype(str)
    de["gene_id"] = de["gene_id"].astype(str)
    de["de_pvalue"] = pd.to_numeric(de["de_pvalue"], errors="coerce")
    de["de_effect"] = pd.to_numeric(de["de_effect"], errors="coerce")
    de = de.dropna(subset=["de_pvalue", "de_effect"]).copy()

    # Default notion of "marker": significant and positive direction in the DE contrast.
    de = de[(de["de_pvalue"] <= float(de_alpha)) & (de["de_effect"] > 0)].copy()
    if de.empty:
        return pd.DataFrame(
            columns=[
                "meta_cluster",
                "tier",
                "gene_id",
                "support_n_samples",
                "de_pvalue",
                "de_effect",
                "replicated_module_id",
                "module_profile_weight",
            ]
        )

    rm = replicated_modules[["replicated_module_id", "gene_id", "support_n_samples"]].copy()
    rm["replicated_module_id"] = rm["replicated_module_id"].astype(int)
    rm["gene_id"] = rm["gene_id"].astype(str)
    rm["support_n_samples"] = rm["support_n_samples"].astype(int)
    rm = rm[rm["support_n_samples"] >= int(support_min)].copy()
    if rm.empty:
        return pd.DataFrame(
            columns=[
                "meta_cluster",
                "tier",
                "gene_id",
                "support_n_samples",
                "de_pvalue",
                "de_effect",
                "replicated_module_id",
                "module_profile_weight",
            ]
        )

    prof = meta_cluster_module_profiles[["meta_cluster", "replicated_module_id", "profile_weight"]].copy()
    prof["meta_cluster"] = prof["meta_cluster"].astype(str)
    prof["replicated_module_id"] = prof["replicated_module_id"].astype(int)
    prof["profile_weight"] = pd.to_numeric(prof["profile_weight"], errors="coerce").fillna(0.0).astype(np.float32)

    # Pick top modules (by profile_weight) per meta_cluster.
    top_modules = (
        prof.sort_values(["meta_cluster", "profile_weight"], ascending=[True, False])
        .groupby("meta_cluster", as_index=False)
        .head(int(top_modules_per_meta_cluster))
        .copy()
    )
    if top_modules.empty:
        return pd.DataFrame(
            columns=[
                "meta_cluster",
                "tier",
                "gene_id",
                "support_n_samples",
                "de_pvalue",
                "de_effect",
                "replicated_module_id",
                "module_profile_weight",
            ]
        )

    # Candidate genes: DE markers that are in one of the top modules for that meta_cluster.
    candidates = de.merge(rm, on="gene_id", how="inner")
    candidates = candidates.merge(top_modules, on=["meta_cluster", "replicated_module_id"], how="inner")
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "meta_cluster",
                "tier",
                "gene_id",
                "support_n_samples",
                "de_pvalue",
                "de_effect",
                "replicated_module_id",
                "module_profile_weight",
            ]
        )

    candidates = candidates.rename(columns={"profile_weight": "module_profile_weight"})
    # Rank: module profile weight (desc), then stronger DE effect (desc), then lower p (asc).
    candidates = candidates.sort_values(
        ["meta_cluster", "module_profile_weight", "de_effect", "de_pvalue"],
        ascending=[True, False, False, True],
    )

    def _take_tier(df: pd.DataFrame, max_n: int) -> pd.DataFrame:
        return df.groupby("meta_cluster", as_index=False).head(int(max_n)).copy()

    core = _take_tier(candidates, core_max_genes)
    core["tier"] = "core"
    extended = _take_tier(candidates, extended_max_genes)
    extended["tier"] = "extended"

    out = pd.concat([core, extended], axis=0, ignore_index=True)
    out = out[
        [
            "meta_cluster",
            "tier",
            "gene_id",
            "support_n_samples",
            "de_pvalue",
            "de_effect",
            "replicated_module_id",
            "module_profile_weight",
        ]
    ].copy()
    return out


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
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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
    sign_discordant_mask = both & (~merged["agree_sign"])
    n_sign_discordant = int(sign_discordant_mask.sum())
    merged = merged[~sign_discordant_mask].copy()

    merged["confidence"] = np.where(both, "both_splits", "single_split")
    stats = {"n_sign_discordant_dropped": n_sign_discordant}
    return merged, stats


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
    prov_train = _safe_json_loads(art_train.provenance_json)
    prov_val = _safe_json_loads(art_val.provenance_json)
    prov_fields_train = _extract_provenance_fields(prov_train)
    prov_fields_val = _extract_provenance_fields(prov_val)
    if art_train.feature_ids_kept != art_val.feature_ids_kept:
        raise ValueError("train/val feature_ids_kept differ; align feature selection before module discovery.")

    edges_train = extract_thresholded_edges(train_h5, chunk_rows=chunk_rows, upper_triangle_only=True)
    edges_val = extract_thresholded_edges(val_h5, chunk_rows=chunk_rows, upper_triangle_only=True)

    if split_mode not in {"train", "val", "union"}:
        raise ValueError("split_mode must be one of: 'train', 'val', 'union'")

    merge_stats: Dict[str, Any] = {}
    if split_mode == "train":
        edges = edges_train.copy()
        edge_meta = {"mode": "train"}
    elif split_mode == "val":
        edges = edges_val.copy()
        edge_meta = {"mode": "val"}
    else:
        edges, merge_stats = _union_split_edges(edges_train, edges_val)
        edge_meta = {"mode": "union"}

    # Weight edges by "excess beyond cutoff" (log1p), per-split, then average across splits
    # (missing treated as 0) to match our union merge semantics.
    if "rho_train" in edges.columns:
        edges["w_train"] = _edge_weight_from_excess(
            rho=edges["rho_train"].fillna(0.0).to_numpy(),
            sign=edges["sign"].to_numpy(),
            cpos=art_train.cpos,
            cneg=art_train.cneg,
        )
    if "rho_val" in edges.columns:
        edges["w_val"] = _edge_weight_from_excess(
            rho=edges["rho_val"].fillna(0.0).to_numpy(),
            sign=edges["sign"].to_numpy(),
            cpos=art_val.cpos,
            cneg=art_val.cneg,
        )
    if "w_train" in edges.columns or "w_val" in edges.columns:
        edges["w_train"] = edges.get("w_train", 0.0)
        edges["w_val"] = edges.get("w_val", 0.0)
        edges["w_mean"] = ((edges["w_train"].astype(np.float32) + edges["w_val"].astype(np.float32)) / 2.0).astype(
            np.float32
        )

    pos, neg = _split_edges_by_sign(edges)

    weight_col = "w_mean" if "w_mean" in pos.columns else ("rho_mean" if "rho_mean" in pos.columns else "rho")
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
            "provenance_fields": prov_fields_train,
            "cpos": art_train.cpos,
            "cneg": art_train.cneg,
        },
        "val": {
            "path": str(art_val.path),
            "schema_version": art_val.schema_version,
            "provenance_sha256": art_val.provenance_sha256,
            "provenance_fields": prov_fields_val,
            "cpos": art_val.cpos,
            "cneg": art_val.cneg,
        },
        "edges": edge_meta,
        "merge_stats": merge_stats,
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

    report_path = out_dir / "gene_modules.report.json"
    _write_report_json(
        report_path,
        {
            "artifacts": {k: v for k, v in artifact_paths.items() if v},
            "params": params,
            "summary": {
                "n_genes": int(art_train.n_features),
                "n_positive_edges": int(pos.shape[0]),
                "n_negative_edges": int(neg.shape[0]),
                "n_modules": int(pd.Series(labels).nunique()),
                "merge_stats": merge_stats,
            },
            "source": {
                "scratch_dir": str(scratch_dir),
                "train_spearman_h5": str(train_h5),
                "val_spearman_h5": str(val_h5),
            },
        },
    )

    out_paths: Dict[str, Path] = {
        "gene_modules": modules_path,
        "module_stats": stats_path,
        "gene_module_antagonism": antagonism_path,
        "report_json": report_path,
    }
    if persist_edges:
        out_paths.update({"gene_edges_pos": edge_pos_path, "gene_edges_neg": edge_neg_path})
    return out_paths
