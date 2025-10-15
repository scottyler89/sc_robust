import dill
import anndata as ad
from copy import copy, deepcopy
from typing import List, Any, Optional
from count_split.count_split import multi_split
from anticor_features.anticor_features import get_anti_cor_genes
from .normalization import *
from .find_consensus import find_pcs, find_consensus_graph


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
    """
    def __init__(self, 
                 in_ad: Any,
                 gene_ids: Optional[List] = None,
                 splits: Optional[List] = [0.325,0.325,0.35],
                 pc_max: Optional[int] = 250,
                 norm_function: Optional[str] = "pf_log",
                 species: Optional[str] = "hsapiens",
                 initial_k: Optional[int] = None,
                 do_plot: Optional[bool] = False,
                 seed = 123456) -> None:
        np.random.seed(seed)
        self.gene_ids = gene_ids
        self.initial_k = initial_k
        self.original_ad = in_ad
        self.splits = splits
        self.pc_max = pc_max
        self.species = species
        self.do_plot = do_plot
        if norm_function not in NORM:
            raise AssertionError("norm_function arg must be one of:"+", ".join(sorted(list(NORM.keys()))))
        self.norm_function = norm_function
        self.do_splits()
        self.normalize()
        self.feature_select()
        self.find_reproducible_pcs()
        self.get_consensus_graph()
        return
    #
    #
    def save(self, f):
        """Save the robust object to a file using dill."""
        with open(f, 'wb') as file:
            dill.dump(self, file)
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
            self.train, self.val = multi_split(self.original_ad, percent_vect=self.splits)
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
        print("normalizing the three splits")
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
        print("performing featue selection")
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
        ## TODO: Handle case in which nothing was selected at FDR = 0.05 in train and/or val
        self.train_feature_df = get_anti_cor_genes(self.train[subset_idxs,:].T,
                                              self.gene_ids,
                                              species=self.species)
        self.train_feat_idxs = np.where(self.train_feature_df["selected"]==True)[0]
        self.val_feature_df = get_anti_cor_genes(self.val[subset_idxs,:].T,
                                              self.gene_ids,
                                              species=self.species)
        self.val_feat_idxs = np.where(self.val_feature_df["selected"]==True)[0]
    #
    #
    def find_reproducible_pcs(self):
        """Compute SVD PCs and keep cross-split reproducible components."""
        self.train_pcs, self.val_pcs = find_pcs(
            self.train[:,self.train_feat_idxs], 
            self.val[:,self.val_feat_idxs], 
            pc_max = self.pc_max,
            do_plot = self.do_plot)
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
        return


