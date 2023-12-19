import anndata as ad
from copy import copy, deepcopy
from typing import List, Any, Optional, Float
from count_split.count_split import multi_split
from .normalization import *
from .find_consensus import find_pcs, find_graph

class robust(object):
    def __init__(self, 
                 in_ad: Any,
                 splits: Optional[List] = [0.39,0.26,0.35],
                 pc_max: Optional[int] = 250,
                 pc_threshold: Optional[Float] = 0.1,
                 norm_function: Optional[str] = "cpm_log") -> None:
        self.original_ad = in_ad
        self.splits = splits
        self.pc_max = pc_max
        self.pc_threshold = pc_threshold
        if norm_function not in norm_dict:
            raise AssertionError("norm_function arg must be one of:"+", ".join(sorted(list(NORM.keys()))))
        self.do_splits()
        self.normalize()
        self.find_reproducible_pcs()
        self.get_consensus_graph()
        return
    #
    #
    def do_splits(self):
        if len(self.splits)!=3:
            self.train, self.val, self.test = multi_split(self.original_ad.X.T, percent_vect=self.splits)
        elif len(self.splits)!=2:
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
    def find_reproducible_pcs(self):
        train_pc, val_pc = find_pcs(
            self.train, 
            self.val, 
            pc_max = self.pc_max,
            pc_threshold = self.pc_threshold)
        return
    #
    #
    def get_consensus_graph(self):
        

