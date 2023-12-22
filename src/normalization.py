import numpy as np
#from pysctransform import vst
#from st_rle.rle import do_depth_normalization


def depth_norm(in_mat,target=None):
    # Normalizes the row
    loading = np.sum(in_mat,axis=1)
    if target is None:
        # Default to counts/avg total counts
        target = np.mean(loading)
    loading_factors = loading/target
    return(in_mat / loading_factors[:,None])


def tenk_norm(in_mat):
    return(depth_norm(in_mat, target=10000))

#def sctransform_vst(in_mat):
#    vst_out_3k = vst(umi = in_mat,
#                    gene_names=["gene_"+str(i) for i in range(in_mat.shape[0])],
#                    cell_names=["cell_"+str(i) for i in range(in_mat.shape[1])],
#                    method="fix-slope",
#                    exclude_poisson=True
#                    )


#def rle_norm(in_mat):
#    return(do_depth_normalization(in_mat))

#############################################
# Taken from Sina's script:
# https://github.com/pachterlab/BHGP_2022/blob/main/scripts/norm_sparse.py
import sys
import os
from scipy.io import mmread, mmwrite

import numpy as np
import scipy as sp

def do_pf(mtx, sf = None):
    pf = mtx.sum(axis=1).A.ravel()
    if not sf:
        sf = pf.mean()
    pf = sp.sparse.diags(sf/pf) @ mtx
    return pf

def norm_raw(mtx):
    return mtx

def norm_pf(mtx):
    return do_pf(mtx)

def norm_log(mtx):
    return np.log1p(mtx)

def norm_pf_log(mtx):
    pf_log = np.log1p(do_pf(mtx))
    return pf_log

def norm_pf_log_pf(mtx):
    pf_log_pf = do_pf(np.log1p(do_pf(mtx)))
    return pf_log_pf

def norm_cpm_log(mtx):
    cpm_log = np.log1p(do_pf(mtx, sf=1e6))
    return cpm_log

def norm_cp10k_log(mtx):
    cp10k_log =  np.log1p(do_pf(mtx, sf=1e4))
    return cp10k_log

def norm_sqrt(mtx):
    sqrt = np.sqrt(mtx)
    return sqrt

NORM = {
    "raw": norm_raw,
    ##"sc_transform": sctransform_vst,
    #"rle": rle_norm,
    "pf": norm_pf,
    "log": norm_log,
    "pf_log": norm_pf_log,
    "pf_log_pf": norm_pf_log_pf,
    "cpm_log": norm_cpm_log,
    "cp10k_log": norm_cp10k_log,
    "sqrt": norm_sqrt
}



