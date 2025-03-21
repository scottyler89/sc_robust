import anndata as ad
from copy import copy, deepcopy
from typing import List, Any, Optional
from count_split.count_split import multi_split
from anticor_features.anticor_features import get_anti_cor_genes
from .normalization import *
from .find_consensus import find_pcs, find_consensus_graph


class robust(object):
    def __init__(self, 
                 in_ad: Any,
                 splits: Optional[List] = [0.325,0.325,0.35],
                 pc_max: Optional[int] = 250,
                 norm_function: Optional[str] = "cpm_log",
                 species: Optional[str] = "hsapiens",
                 initial_k: Optional[int] = None,
                 do_plot: Optional[bool] = False,
                 seed = 123456) -> None:
        np.random.seed(seed)
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
    def do_splits(self):
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
        return
    #
    #
    def normalize(self):
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
        if subset_idxs is None:
            subset_idxs = np.arange(self.train.shape[0])
        print("performing featue selection")
        if "gene_ids" in self.original_ad.var:
            gene_ids = self.original_ad.var["gene_ids"].tolist()
        elif "gene_name" in self.original_ad.var:
            gene_ids = self.original_ad.var["gene_name"].tolist()
        else:
            gene_ids = self.original_ad.var.index.tolist()
        self.train_feature_df = get_anti_cor_genes(self.train[subset_idxs,:].T,
                                              gene_ids,
                                              species=self.species)
        self.train_feat_idxs = np.where(self.train_feature_df["selected"]==True)[0]
        self.val_feature_df = get_anti_cor_genes(self.val[subset_idxs,:].T,
                                              gene_ids,
                                              species=self.species)
        self.val_feat_idxs = np.where(self.val_feature_df["selected"]==True)[0]
    #
    #
    def find_reproducible_pcs(self):
        self.train_pcs, self.val_pcs = find_pcs(
            self.train[:,self.train_feat_idxs], 
            self.val[:,self.val_feat_idxs], 
            pc_max = self.pc_max,
            do_plot = self.do_plot)
        return
    #
    #
    def get_consensus_graph(self):
        self.indices, self.distances, self.weights, self.graph = find_consensus_graph(
            self.train_pcs, 
            self.val_pcs, 
            self.initial_k, 
            cosine = True, use_gpu = False)
        return

