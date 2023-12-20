import faiss
import torch
import numpy as np
from copy import deepcopy
from typing import Int, Optional, Any, List, Float
from scipy.stats import norm
from sklearn import metrics
from sklearn.metrics import pairwise
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances as euc
from anticor_features.anticor_stats import no_p_pear as pearson

def find_pcs(train_mat: Any, 
             val_mat: Any,
             max_pcs: Optional[Int] = 250,
             k: Optional[Float]=None):
    if k is None:
        # Default to the log of the number of observations
        k = max([200,int(np.log(train_mat.shape[0]))])
    # Keep only the genes that are expressed in both
    train_idxs = sorted(list(set(train_mat.indices)))
    val_idxs = sorted(list(set(val_mat.indices)))
    both_idxs = [idx for idx in val_idxs if idx in train_idxs]
    train_pc = tsvd(train_mat, npcs=max_pcs)
    val_pc = tsvd(val_mat, npcs=max_pcs)
    train_keep, val_keep = keep_correlated_pcs(train_pc, val_pc)
    train_pc = train_pc[:,train_keep]
    val_pc = val_pc[:,val_keep]
    return train_pc, val_pc



def tsvd(temp_mat, npcs=250):
    pca = TruncatedSVD(n_components=npcs,n_iter=7, random_state=42)
    out_pcs = pca.fit_transform(temp_mat)
    return(out_pcs)


def keep_correlated_pcs(train_pc, val_pc, alpha=0.01, n_boot = 50, do_plot = False):
    adjusted_alpha = alpha/(train_pc.shape[1]*val_pc.shape[1])/2 # /2 for 2-tail
    z_cutoff = norm.ppf(1-adjusted_alpha)
    ## get the real correlations
    pear_res = pearson(train_pc,val_pc)[:train_pc.shape[1],train_pc.shape[1]:]
    if do_plot:
        sns.heatmap(pear_res)
        plt.show()
    boot_mat = np.zeros(
        (pear_res.shape[0],
        pear_res.shape[1],
        n_boot))
    train_shuff = deepcopy(train_pc)
    for b in range(n_boot):
        ## get the null distributions
        for col in range(train_pc.shape[1]):
            # go through the columns (ie: pcs)
            # & shuffle them 
            train_shuff[:,col]=np.random.permutation(train_shuff[:,col])
        boot_mat[:,:,b] = pearson(train_shuff,val_pc)[:train_shuff.shape[1],train_shuff.shape[1]:]
    mean_b = np.mean(boot_mat, axis=2)
    sd_b = np.std(boot_mat, axis=2)
    std_res = (pear_res-mean_b)/sd_b
    if do_plot:
        sns.heatmap(std_res)
        plt.show()
    above_cutoff = std_res>z_cutoff
    below_cutoff = std_res< -1*z_cutoff
    include_bool = above_cutoff + below_cutoff
    if do_plot:
        sns.heatmap(include_bool)
        plt.show()
    include_train = np.max(include_bool,axis=1)
    include_val = np.max(include_bool,axis=0)
    print("train shape:",include_train.shape)
    print("val shape:",include_val.shape)
    return include_train, include_val






################################################
## Find the graphs
def find_consensus_graph(train_pcs, val_pcs, cosine = True, use_gpu = False):
    train_neigh, train_dist = find_one_graph(train_pcs, cosine = cosine, use_gpu = use_gpu)
    val_neigh, val_dist = find_one_graph(val_pcs, cosine = cosine, use_gpu = use_gpu)
    return

def merge_graphs(t_i,
                 t_d,
                 t_m,
                 v_i,
                 v_d,
                 v_m):
    # i: indices, d: distances, m: mask
    


def find_one_graph(pcs, k, cosine=True, use_gpu = False):
    index = get_faiss_idx(pcs, cosine = cosine, use_gpu = use_gpu)
    indexes, distances = find_k_nearest_neighbors(index, pcs, k)
    ## filter using slicer method to get mask
    local_mask = mask_knn_local_diff_dist(torch.tensor(distances))
    return(indexes, distances, local_mask) 


def get_faiss_idx(vectors, cosine=True, use_gpu = False):
    ## Normalize the PCs to make it cosine distance
    if cosine:
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product (IP) for cosine similarity
    if use_gpu:
        # Using GPU resources if available and desired
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(vectors)
    return(index)


def find_k_nearest_neighbors(index, vectors, k):
    """ Find k-nearest neighbors for each vector. """
    cos_sim, indices = index.search(vectors, k)
    ## 1-cosine similarity is cosine distance
    return indices, 1-cos_sim


##############################################################
# KNN masking functions
def masked_mad(input_tensor, mask, epsilon = 1e-8):
    # Make sure the mask is a bool tensor
    mask = mask.bool()
    # Apply mask
    masked_tensor = input_tensor * mask
    # Calculate median
    med = torch.median(masked_tensor, dim=1).values.unsqueeze(1)
    # Calculate Median Absolute Deviation
    mad = torch.median(torch.abs(masked_tensor - med),
                       dim=1).values.unsqueeze(1)
    mad[mad < epsilon] = epsilon
    return med, mad


def get_mad_ratio(input_tensor, mask):
    med, mad = masked_mad(input_tensor, mask)
    return (input_tensor - med)/mad


def mask_knn_local_diff_dist(dists, prior_mask=None, cutoff_threshold=3, min_k=10):
    """
    Generate a mask for k nearest neighbors based on a cutoff threshold.
    
    Parameters:
    ----------
    obs_knn_dist : numpy ndarray
        A matrix of the distances to the k nearest neighbors in the observed data.

    k : int
        The number of nearest neighbors to consider.

    cutoff_threshold : float
        The relative gap threshold for considering a difference between sorted order distances.
        This cutoff is the multiple of the mean sort-order difference for considering the distance.
        "too big" to be kept. Getting to this point or farther away will be masked.
        
    Returns:
    -------
    mask : numpy ndarray
        A binary mask matrix indicating which distances should be considered (1) and which should be ignored (0).

    Examples:
    --------
    >>> obs_knn_dist = np.array(...)
    >>> k = 10
    >>> cutoff_threshold = 3.0
    >>> mask = mask_knn(obs_knn_dist, k, cutoff_threshold)
    """
    if prior_mask is None:
        prior_mask = torch.ones_like(dists)
    print("dists.shape:",dists.shape)
    print("min_k:",min_k)
    print("mean number connections BEFORE local masking:")
    print(torch.mean(torch.sum(prior_mask.float(), dim=1)))
    # Create a boolean tensor with ones (True) of the same shape as dists
    diff_mask = torch.ones_like(dists, dtype=torch.bool)
    # Calculate the discrete difference of the relevant slice of dists
    discrete_diff = dists[:, (min_k+1):] - dists[:, min_k:-1]
    # Now subset the prior mask to make it compatible
    prior_mask_diff = prior_mask[:, (min_k+1):]
    #print("discrete_diff:",discrete_diff)
    # Standardize the discrete difference
    #print("discrete_diff.shape:",discrete_diff.shape)
    #print("prior_mask_diff.shape:",prior_mask_diff.shape)
    discrete_diff = get_mad_ratio(discrete_diff, prior_mask_diff)
    #print("standardized discrete_diff:")
    #print(discrete_diff)
    # Create a temporary mask for values below the cutoff threshold
    temp_diff_mask_cutoff = discrete_diff < cutoff_threshold
    # Apply prior_mask to temp_diff_mask_cutoff
    temp_diff_mask_cutoff = temp_diff_mask_cutoff & prior_mask_diff
    #print("temp_diff_mask_cutoff with prior mask applied:")
    #print(temp_diff_mask_cutoff)
    # Find indices with a jump (where there is a false)
    idxs_with_diff_mask = (torch.min(temp_diff_mask_cutoff,
                                     dim=1).values == 0).nonzero(as_tuple=True)[0]
    for idx in idxs_with_diff_mask:
        # Then mask everything that is farther than the jump
        # Where are they false, then which index is lowest that is false
        temp_gap_idx = (temp_diff_mask_cutoff[idx, :] == False).nonzero(
            as_tuple=True)[0].min()
        temp_diff_mask_cutoff[idx, temp_gap_idx:] = False
    # Apply prior_mask to diff_mask
    diff_mask = diff_mask & prior_mask
    diff_mask[:, (min_k+1):] = temp_diff_mask_cutoff
    #print("diff_mask after local diff masking")
    #print(diff_mask)
    print("mean number connections after local masking:")
    print(torch.mean(torch.sum(diff_mask.float(), dim=1)))
    return diff_mask


#########################################################################
