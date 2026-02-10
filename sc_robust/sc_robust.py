import dill
import anndata as ad
from copy import copy, deepcopy
import inspect
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from count_split.count_split import multi_split
from anticor_features.anticor_features import get_anti_cor_genes
import numpy as np
from .normalization import *
from .find_consensus import find_pcs, find_consensus_graph
from importlib import metadata
import hashlib
import datetime
import json
import time
import warnings

logger = logging.getLogger(__name__)


def _call_get_anti_cor_genes(*args, **kwargs):
    """Call `anticor_features.get_anti_cor_genes` filtering kwargs to supported keys.

    This keeps sc_robust compatible across anticor_features versions as the API evolves.
    """
    sig = inspect.signature(get_anti_cor_genes)
    allowed = set(sig.parameters)
    filtered = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    return get_anti_cor_genes(*args, **filtered)


@contextmanager
def _temporary_numpy_seed(seed: int):
    """Temporarily set numpy's global RNG seed and restore the previous state."""
    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(state)


def _anticor_features_failure_message(
    exc: Exception,
    *,
    species: str,
    split_label: str,
    anticor_options: Dict[str, Any],
) -> str:
    """Produce an actionable error message for anticor_features failures."""
    exc_type = type(exc).__name__
    exc_msg = str(exc)
    exc_lower = exc_msg.lower()

    offline_mode = bool(anticor_options.get("offline_mode", False))
    use_live = bool(anticor_options.get("use_live_pathway_lookup", False))
    pre_remove_pathways = anticor_options.get("pre_remove_pathways", None)

    lines = [
        f"anticor_features failed during feature selection ({split_label}).",
        f"species={species!r} offline_mode={offline_mode} use_live_pathway_lookup={use_live}",
        f"error_type={exc_type} error={exc_msg}",
    ]

    hints: List[str] = []

    if offline_mode and use_live:
        hints.append(
            "You requested both offline_mode=True and use_live_pathway_lookup=True; disable live lookup or offline mode."
        )

    # Heuristics for common offline / network failures.
    networkish = any(tok in exc_lower for tok in ("g:profiler", "gprofiler", "connection", "timed out", "network", "proxy"))
    id_bankish = any(tok in exc_lower for tok in ("id bank", "idbank", "bank not found", "idbanknotfound"))

    if networkish:
        hints.append(
            "This looks like a network-dependent pathway lookup (g:Profiler) in an environment without internet access."
        )
        hints.append(
            "Fix: set use_live_pathway_lookup=False (recommended) and rely on shipped ID banks, or set pre_remove_pathways=[] to skip pathway-based removal."
        )
        hints.append(
            "If you need custom pathway removals offline, generate/provide an ID bank and pass id_bank_dir=... ."
        )

    if id_bankish:
        hints.append(
            "The offline ID bank appears missing for this species or configuration."
        )
        hints.append(
            "Fix: install/regenerate the anticor_features ID bank for your species, or pass id_bank_dir=... ."
        )
        if not offline_mode:
            hints.append(
                "Alternatively, allow live lookup by setting use_live_pathway_lookup=True (requires internet)."
            )

    if offline_mode and pre_remove_pathways not in (None, []) and not networkish and not id_bankish:
        hints.append(
            "With offline_mode=True, custom pre_remove_pathways may be unsupported unless you provide an ID bank; consider pre_remove_pathways=[] or id_bank_dir=... ."
        )

    if hints:
        lines.append("Hints:")
        lines.extend([f"- {hint}" for hint in hints])
    else:
        lines.append("Hints:")
        lines.append("- Try setting offline_mode=True for clearer offline-only behavior, or set use_live_pathway_lookup=False to avoid live lookups.")
        lines.append("- If using pathway-based removal offline, ensure an ID bank exists for your species or pass id_bank_dir=... .")

    return "\n".join(lines)


class robust(object):
    """Split-based robust graph builder and clustering helper.

    This class orchestrates a split-then-validate pipeline:
      1) Split counts into train/val(/test) via count splitting
      2) Normalize per split (configurable)
      3) Feature select via anti-correlation gene finder
      4) Dimensionality reduction (SVD) with cross-split PC validation
      5) KNN graphs per split with local masking and consensus merge

    Attributes like `graph`, `indices`, `distances`, and `weights` are
    populated after initialization for downstream clustering.

    Parameters:
      in_ad: AnnData-like object (expects `.X`, `.var`, `.obs`).
      gene_ids: Optional list of gene identifiers to use; if None, inferred
        from `.var['gene_ids']`, `.var['gene_name']`, or `.var.index`.
      splits: Two or three proportions for train/val(/test) (sum ≈ 1.0).
      pc_max: Max components for truncated SVD before validation.
      norm_function: One of keys in `NORM` (e.g., 'pf_log').
      species: Species key forwarded to anti-correlation feature selection.
      initial_k: Optional K for initial KNN search (defaults to round(log n)).
      do_plot: If True, enables diagnostic plots during PC validation.
      seed: Random seed used for deterministic count-splitting.
      scratch_dir: Optional directory for anticor_features to persist artifacts.
      anticor_kwargs: Optional extra kwargs forwarded to anticor_features.
    """
    def __init__(self, 
                 in_ad: Any,
                 gene_ids: Optional[List] = None,
                 splits: Optional[List] = [0.325,0.325,0.35],
                 pc_max: Optional[int] = 250,
                 norm_function: Optional[str] = "pf_log",
                 species: Optional[str] = "hsapiens",
                 scratch_dir: Optional[Union[str, Path]] = None,
                 pre_remove_features: Optional[List[str]] = None,
                 pre_remove_pathways: Optional[List[str]] = None,
                 offline_mode: Optional[bool] = None,
                 id_bank_dir: Optional[Union[str, Path]] = None,
                 use_live_pathway_lookup: Optional[bool] = None,
                 anticor_kwargs: Optional[Dict[str, Any]] = None,
                 initial_k: Optional[int] = None,
                 do_plot: Optional[bool] = False,
                 seed = 123456) -> None:
        self.seed = int(seed)
        self.gene_ids = gene_ids
        self.initial_k = initial_k
        self.original_ad = in_ad
        self.splits = splits
        if self.splits is not None and len(self.splits) == 2:
            warnings.warn(
                "splits has length 2: sc_robust will copy the validation split into `test` (no held-out DE split). "
                "If you build clusters/graphs using the validation split, you must NOT reuse `test` for DE, "
                "as it is not independent and re-introduces the double-dipping problem. "
                "Use a 3-way split (train/val/test) for independent downstream DE.",
                UserWarning,
                stacklevel=2,
            )
        self.pc_max = pc_max
        self.species = species
        self.scratch_dir = Path(scratch_dir) if scratch_dir is not None else None
        self.anticor_options: Dict[str, Any] = dict(anticor_kwargs or {})
        self.anticor_options.setdefault("pre_remove_features", pre_remove_features)
        self.anticor_options.setdefault("pre_remove_pathways", pre_remove_pathways)
        self.anticor_options.setdefault("offline_mode", offline_mode)
        self.anticor_options.setdefault("id_bank_dir", str(id_bank_dir) if id_bank_dir is not None else None)
        self.anticor_options.setdefault("use_live_pathway_lookup", use_live_pathway_lookup)
        if bool(self.anticor_options.get("offline_mode", False)) and bool(self.anticor_options.get("use_live_pathway_lookup", False)):
            raise ValueError("offline_mode=True is incompatible with use_live_pathway_lookup=True.")
        self.do_plot = do_plot
        if norm_function not in NORM:
            raise AssertionError("norm_function arg must be one of:"+", ".join(sorted(list(NORM.keys()))))
        self.norm_function = norm_function

        self.provenance = self._build_provenance()
        start = time.perf_counter()
        with _temporary_numpy_seed(self.seed):
            self.do_splits()
            logger.info("step=do_splits elapsed_s=%.3f", time.perf_counter() - start)
            self.provenance["splits_out"] = {
                "train_shape": tuple(getattr(self.train, "shape", ())),
                "val_shape": tuple(getattr(self.val, "shape", ())),
                "test_shape": tuple(getattr(self.test, "shape", ())),
            }
            self.normalize()
            logger.info("step=normalize elapsed_s=%.3f", time.perf_counter() - start)
            self.feature_select()
            logger.info("step=feature_select elapsed_s=%.3f", time.perf_counter() - start)
        train_feat_idxs = getattr(self, "train_feat_idxs", None)
        val_feat_idxs = getattr(self, "val_feat_idxs", None)
        self.provenance["feature_selection"] = {
            "train_selected_n": int(len(train_feat_idxs)) if train_feat_idxs is not None else 0,
            "val_selected_n": int(len(val_feat_idxs)) if val_feat_idxs is not None else 0,
            "scratch_dir": str(self.scratch_dir) if self.scratch_dir is not None else None,
        }
        self.find_reproducible_pcs()
        logger.info("step=find_reproducible_pcs elapsed_s=%.3f", time.perf_counter() - start)
        self.provenance["pcs"] = {
            "train_pcs_shape": tuple(getattr(self.train_pcs, "shape", ())),
            "val_pcs_shape": tuple(getattr(self.val_pcs, "shape", ())),
        }
        if getattr(self.train_pcs, "shape", (0, 0))[1] == 0 or getattr(self.val_pcs, "shape", (0, 0))[1] == 0:
            # Null / no-structure case: the empirical validation found no reproducible PCs.
            # Keep the algorithmic outcome (0 PCs), but report it gracefully and stop.
            self.no_reproducible_pcs = True
            self.indices = None
            self.distances = None
            self.weights = None
            self.graph = None
            self.provenance["status"] = "no_reproducible_pcs"
            logger.warning(
                "No reproducible PCs found across splits; graph construction skipped. "
                "This is expected on null/no-structure datasets."
            )
            return
        self.get_consensus_graph()
        logger.info("step=get_consensus_graph elapsed_s=%.3f", time.perf_counter() - start)
        self.provenance.setdefault("graph", {})
        self.provenance["graph"].update(
            {
                "shape": tuple(getattr(self.graph, "shape", ())),
                "nnz": int(getattr(self.graph, "nnz", 0)),
            }
        )
        return
    #
    #
    def save(self, f):
        """Save the robust object to a file using dill."""
        with open(f, 'wb') as file:
            dill.dump(self, file)

    def _hash_strings(self, values: List[str]) -> str:
        h = hashlib.sha256()
        for v in values:
            h.update(v.encode("utf-8", errors="ignore"))
            h.update(b"\0")
        return h.hexdigest()

    def _safe_pkg_version(self, name: str) -> Optional[str]:
        try:
            return metadata.version(name)
        except Exception:
            return None

    def _build_provenance(self) -> Dict[str, Any]:
        """Capture lightweight provenance for reproducibility and debugging."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        adata = self.original_ad
        n_obs = getattr(adata, "n_obs", None)
        n_vars = getattr(adata, "n_vars", None)
        x = getattr(adata, "X", None)
        x_shape = tuple(getattr(x, "shape", ())) if x is not None else None

        obs_names = getattr(adata, "obs_names", None)
        var_names = getattr(adata, "var_names", None)
        obs_hash = None
        var_hash = None
        try:
            if obs_names is not None:
                obs_hash = self._hash_strings([str(x) for x in list(obs_names)])
            if var_names is not None:
                var_hash = self._hash_strings([str(x) for x in list(var_names)])
        except Exception:
            # Hashing is best-effort; skip if AnnData-like does not support iteration safely.
            pass

        return {
            "created_utc": now,
            "seed": self.seed,
            "splits": list(self.splits) if self.splits is not None else None,
            "pc_max": self.pc_max,
            "norm_function": self.norm_function,
            "species": self.species,
            "initial_k": self.initial_k,
            "do_plot": bool(self.do_plot),
            "adata": {
                "n_obs": n_obs,
                "n_vars": n_vars,
                "X_shape": x_shape,
                "obs_names_sha256": obs_hash,
                "var_names_sha256": var_hash,
            },
            "deps": {
                "sc_robust": self._safe_pkg_version("sc_robust"),
                "anticor_features": self._safe_pkg_version("anticor_features"),
                "count_split": self._safe_pkg_version("count_split"),
                "numpy": self._safe_pkg_version("numpy"),
                "scipy": self._safe_pkg_version("scipy"),
                "torch": self._safe_pkg_version("torch"),
                "faiss-cpu": self._safe_pkg_version("faiss-cpu"),
                "faiss": self._safe_pkg_version("faiss"),
                "igraph": self._safe_pkg_version("igraph"),
                "leidenalg": self._safe_pkg_version("leidenalg"),
            },
            "anticor_options": dict(self.anticor_options),
        }

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)

    def _record_feature_selection_manifest(
        self,
        *,
        split: str,
        feature_df,
        selected_indices: np.ndarray,
        subset_idxs: np.ndarray,
        scratch_dir: Optional[Path],
    ) -> None:
        if scratch_dir is None:
            return
        try:
            feature_ids_order = [str(x) for x in feature_df.index.tolist()]
        except Exception:
            feature_ids_order = []
        kept_features_order = []
        try:
            kept_features_order = [str(feature_df.index[i]) for i in selected_indices.tolist()]
        except Exception:
            kept_features_order = []

        manifest = {
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "split": split,
            "seed": self.seed,
            "species": self.species,
            "subset_n": int(len(subset_idxs)),
            "n_features_total": int(len(feature_df)),
            "n_features_selected": int(len(selected_indices)),
            "feature_ids_order": feature_ids_order,
            "kept_features_order": kept_features_order,
            "anticor_options": dict(self.anticor_options),
        }
        out_path = scratch_dir / "kept_features_manifest.json"
        self._write_json(out_path, manifest)
        self.provenance.setdefault("artifacts", {})[f"feature_selection_manifest_{split}"] = str(out_path)
    #
    #
    def do_splits(self):
        """Split counts into train/val(/test) via count splitting.

        Notes:
          - The incoming data are assumed to be cells x genes; count_split
            expects samples in columns, hence the transpose adjustments.
        """
        if len(self.splits)==3:
            self.train, self.val, self.test = multi_split(self.original_ad.X.T, percent_vect=self.splits, bin_size = 1000)
        elif len(self.splits)==2:
            # count_split expects samples in columns (cells) and variables in rows (genes).
            # AnnData stores X as cells×genes, so we pass X.T here for consistency with 3-way splits.
            self.train, self.val = multi_split(self.original_ad.X.T, percent_vect=self.splits, bin_size = 1000)
            self.test = copy(self.val)
        else:
            raise AssertionError("Number of splits must be 2 or 3.")
        # The count splitting assumes samples (cells) are in columns, but convention has flipped now
        self.train = self.train.T
        self.val = self.val.T
        self.test = self.test.T
        self.train_counts = self.train.copy()
        self.val_counts = self.val.copy()
        self.test_counts = self.test.copy()
        return
    #
    #
    def normalize(self):
        """Normalize train/val/test matrices using the selected function.

        Default is `pf_log` (pseudocount-normalize then log1p), applied per
        split. If only two splits are provided, `test` is a copy of `val`.
        """
        logger.info("normalizing_splits splits=%s norm_function=%s", self.splits, self.norm_function)
        self.train = NORM[self.norm_function](self.train)
        self.val = NORM[self.norm_function](self.val)
        if len(self.splits)==2:
            ## TODO: Check if this is necessary or if we can pass
            self.test = copy(self.val)
        else:
            self.test = NORM[self.norm_function](self.test)
        return
    #
    #
    def feature_select(self, subset_idxs = None):
        """Run anti-correlation feature selection on each split.

        Parameters:
          subset_idxs: Optional index array to subset features prior to
            selection (default uses all features).
        Side effects:
          - Populates `train_feature_df`, `val_feature_df` (pandas DataFrames)
            and `train_feat_idxs`, `val_feat_idxs` (numpy indices of selected).
        """
        if subset_idxs is None:
            subset_idxs = np.arange(self.train.shape[0])
        logger.info("feature_selection start subset_n=%s", len(subset_idxs))
        if type(self.gene_ids)!=list:
            if "gene_ids" in self.original_ad.var:
                self.gene_ids = self.original_ad.var["gene_ids"].tolist()
            elif "gene_name" in self.original_ad.var:
                self.gene_ids = self.original_ad.var["gene_name"].tolist()
            else:
                self.gene_ids = self.original_ad.var.index.tolist()
        else:
            # Otherwise, we assume the user has provided good IDs
            pass
        train_scratch_dir = None
        val_scratch_dir = None
        if self.scratch_dir is not None:
            train_scratch_dir = self.scratch_dir / "train"
            val_scratch_dir = self.scratch_dir / "val"
            train_scratch_dir.mkdir(parents=True, exist_ok=True)
            val_scratch_dir.mkdir(parents=True, exist_ok=True)
        ## TODO: Handle case in which nothing was selected at FDR = 0.05 in train and/or val
        try:
            self.train_feature_df = _call_get_anti_cor_genes(
                self.train[subset_idxs, :].T,
                self.gene_ids,
                species=self.species,
                scratch_dir=str(train_scratch_dir) if train_scratch_dir is not None else None,
                **self.anticor_options,
            )
        except Exception as exc:
            raise RuntimeError(
                _anticor_features_failure_message(
                    exc,
                    species=self.species,
                    split_label="train",
                    anticor_options=self.anticor_options,
                )
            ) from exc
        self.train_feat_idxs = np.where(self.train_feature_df["selected"]==True)[0]
        self._record_feature_selection_manifest(
            split="train",
            feature_df=self.train_feature_df,
            selected_indices=self.train_feat_idxs,
            subset_idxs=np.asarray(subset_idxs),
            scratch_dir=train_scratch_dir,
        )
        try:
            self.val_feature_df = _call_get_anti_cor_genes(
                self.val[subset_idxs, :].T,
                self.gene_ids,
                species=self.species,
                scratch_dir=str(val_scratch_dir) if val_scratch_dir is not None else None,
                **self.anticor_options,
            )
        except Exception as exc:
            raise RuntimeError(
                _anticor_features_failure_message(
                    exc,
                    species=self.species,
                    split_label="val",
                    anticor_options=self.anticor_options,
                )
            ) from exc
        self.val_feat_idxs = np.where(self.val_feature_df["selected"]==True)[0]
        self._record_feature_selection_manifest(
            split="val",
            feature_df=self.val_feature_df,
            selected_indices=self.val_feat_idxs,
            subset_idxs=np.asarray(subset_idxs),
            scratch_dir=val_scratch_dir,
        )
    #
    #
    def find_reproducible_pcs(self):
        """Compute SVD PCs and keep cross-split reproducible components."""
        self.train_pcs, self.val_pcs = find_pcs(
            self.train[:,self.train_feat_idxs], 
            self.val[:,self.val_feat_idxs], 
            pc_max = self.pc_max,
            do_plot = self.do_plot,
            random_state=self.seed + 3,
        )
        return
    #
    #
    def get_consensus_graph(self):
        """Build KNNs per split, locally mask, and merge into a consensus graph.

        Populates:
          - `indices`, `distances`, `weights`: per-node lists
          - `graph`: scipy.sparse COO adjacency for downstream clustering
        """
        self.indices, self.distances, self.weights, self.graph = find_consensus_graph(
            self.train_pcs, 
            self.val_pcs, 
            self.initial_k, 
            cosine = True, use_gpu = False)
        n = int(self.train_pcs.shape[0])
        k = self.initial_k
        if k is None:
            k = int(round(np.log(n), 0))
        k = min(int(k), 200, n)
        k = max(int(k), min(10, n))
        self.provenance.setdefault("graph", {})
        self.provenance["graph"]["metric"] = "cosine"
        self.provenance["graph"]["k_used"] = int(k)
        return
