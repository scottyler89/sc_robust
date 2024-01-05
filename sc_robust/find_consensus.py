import faiss
import torch
import numpy as np
import seaborn as sns
from math import ceil
from matplotlib import pyplot as plt
from copy import deepcopy
from scipy.stats import norm
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from anticor_features.anticor_stats import no_p_pear as pearson
from typing import Optional, Any
import warnings


def find_pcs(train_mat: Any, 
             val_mat: Any,
             pc_max: Optional[int] = 250,
             do_plot: Optional[bool] = False):
    print("Decomposing training and validation matrices")
    ## Sanity check
    pc_max = min(pc_max,min(train_mat.shape), min(val_mat.shape))
    # Keep only the genes that are expressed in both
    train_idxs = sorted(list(set(train_mat.indices)))
    val_idxs = sorted(list(set(val_mat.indices)))
    both_idxs = [idx for idx in val_idxs if idx in train_idxs]
    train_pc = tsvd(train_mat, npcs=pc_max)
    val_pc = tsvd(val_mat, npcs=pc_max)
    print("\tperforming empirical validation")
    train_keep, val_keep = keep_correlated_pcs(train_pc, val_pc, do_plot=do_plot)
    train_pc = train_pc[:,train_keep]
    val_pc = val_pc[:,val_keep]
    print("\tfinal dimensionality:")
    print("\t\ttraining:",train_pc.shape)
    print("\t\tvalidation:",val_pc.shape)
    return train_pc, val_pc


def tsvd(temp_mat, npcs=250):
    pca = TruncatedSVD(n_components=npcs,n_iter=7, random_state=42)
    out_pcs = pca.fit_transform(temp_mat)
    # Not sure why there are some nans..
    out_pcs[np.isnan(out_pcs)]=0.
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
    return include_train, include_val




################################################
################################################
################################################
## Find the graphs
def find_consensus_graph(train_pcs, val_pcs, initial_k, cosine = True, use_gpu = False):
    train_index, train_distance, train_mask = find_one_graph(train_pcs, initial_k, cosine = cosine, use_gpu = use_gpu)
    val_index, val_distance, val_mask = find_one_graph(val_pcs, initial_k, cosine = cosine, use_gpu = use_gpu)
    merged_indices, merged_distances, merged_weights, merged_graph = merge_knns(train_index, train_distance, train_mask, val_index, val_distance, val_mask)
    return merged_indices, merged_distances, merged_weights, merged_graph


############################################################################
def calculate_maximum_distances(distance_matrix, mask_matrix):
    """ Calculates the maximum included distance for each node """
    masked_distances = distance_matrix.masked_fill(~mask_matrix, float('-inf'))
    return masked_distances.max(dim=1).values


## Helper functions for finding which edges are replicated in both, vs unique to one or the other
def select_valid_indices(index, mask):
    # These are lists of tensors that contain only the pertinent non-masked indices
    return [index[row][mask[row]] for row in range(len(index))]


def get_intersection(i1,i2):
    combined = torch.cat((i1, i2))
    uniques, counts = combined.unique(return_counts=True, sorted=False)
    # calculate symmetric difference and intersect
    sym_diff = uniques[counts == 1]
    intersection = uniques[counts > 1]
    return(intersection, sym_diff)


def find_common_indices(train_indices, val_indices):
    # Returns a list of tensors that contain only the common indices
    all_common = []
    all_unique = []
    for t_indices, v_indices in zip(train_indices, val_indices):
        temp_common, temp_unique = get_intersection(t_indices, v_indices)
        if min(temp_common)<0:
            print("terrible badness2:")
            print(t_indices, v_indices)
            print(temp_common)
            print(temp_unique)
            print(poo)
        else:
            pass
            #print("that good good:")
            #print(t_indices, v_indices)
            #print(temp_common)
            #print(temp_unique)
        all_common.append(temp_common)
        all_unique.append(temp_unique)
    return all_common, all_unique


def find_unique_indices(train_indices, unique_indices):
    # Returns a list of tensors that contain only the common indices
    train_unique = []
    val_unique = []
    for t_indices, u_indices in zip(train_indices, unique_indices):
        # We use the same as above, except counting if training appears in unique indices
        # Things that only appear once are val specific b/c we know they're all already unique
        temp_train, temp_val = get_intersection(t_indices, u_indices)
        train_unique.append(temp_train)
        val_unique.append(temp_val)
    return train_unique, val_unique


def get_min_valid(in_vect):
    temp_min =9999999
    for i in range(len(in_vect)):
        if len(in_vect[i])<temp_min:
            temp_min = len(in_vect[i])
    return(temp_min)



##
def identify_common_unique_edges(train_index, val_index, train_mask, val_mask):
    """
    Identifies common and unique neighbor indices for each node based on the training and validation sets.

    Args:
    - train_index (Tensor[n, k]): Index matrix for the training set.
    - val_index (Tensor[n, k]): Index matrix for the validation set.
    - train_mask (Tensor[n, k]): Mask matrix for the training set.
    - val_mask (Tensor[n, k]): Mask matrix for the validation set.

    Returns:
    - Tuple[List[Tensor], List[Tensor], List[Tensor]]: Three lists of tensors containing common indices,
      unique training indices, and unique validation indices for each node, respectively.
    """
    # Step 1: Select valid indices from the training and validation sets
    valid_train_indices = select_valid_indices(train_index, train_mask)
    valid_val_indices = select_valid_indices(val_index, val_mask)
    # Find common indices between the training and validation sets for each node
    common_indices, unique_indices = find_common_indices(valid_train_indices, valid_val_indices)
    # Find indices unique to the training set for each node
    unique_train_indices, unique_val_indices = find_unique_indices(valid_train_indices, unique_indices)
    return common_indices, unique_train_indices, unique_val_indices


###################
def compute_average_common_distances(common_edges, train_index, val_index, train_distance, val_distance):
    avg_common_distances = []
    for node, ti, vi, td, vd in zip(common_edges, train_index, val_index, train_distance, val_distance):
        temp_tensor = torch.zeros_like(node, dtype=torch.float32)
        for idx, common_neighbor in enumerate(node):
            train_pos = (ti == common_neighbor).nonzero(as_tuple=True)[0]
            val_pos = (vi == common_neighbor).nonzero(as_tuple=True)[0]
            if train_pos.numel() > 0 and val_pos.numel() > 0:
                temp_tensor[idx] = (td[train_pos] + vd[val_pos]) / 2
            else:
                warnings.warn("Something might have gone wrong because we couldn't find the indices in what should be common neighbors")
        avg_common_distances.append(temp_tensor)
    return avg_common_distances


def compute_average_unique_distances(unique_index, ref_index, ref_distance, other_distance):
    avg_unique_distances = []
    for node, ri, rd, od in zip(unique_index, ref_index, ref_distance, other_distance):
        temp_tensor = torch.zeros_like(node, dtype=torch.float32)
        for idx, unique_neighbor in enumerate(node):
            unique_pos = (ri == unique_neighbor).nonzero(as_tuple=True)[0]
            if unique_pos.numel() > 0:
                temp_tensor[idx] = (rd[unique_pos] + od.max()) / 2  # Using the maximum distance from the 'other' set
        avg_unique_distances.append(temp_tensor)
    return avg_unique_distances


########################

def distances_to_weights(temp_dist, eps = 1e-3):
    # The first is self connection, so skip it
    # It's also already sorted, so the minimum non-self is at 1
    # and the maximum is the final entry
    temp_min = temp_dist[1].clone()
    temp_dist[1:] -= (temp_min-eps)
    temp_max = temp_dist[-1].clone()
    temp_dist[1:] /= (temp_max+eps)
    # invert it
    temp_dist[1:] *= -1. 
    # shift it positive
    temp_dist[1:] += 1.
    temp_dist[0] = 1.
    return(temp_dist)


def indices_and_weights_to_graph(indices, weights, length):
    r_lin = np.zeros((length),dtype=np.int64)
    c_lin = np.zeros((length),dtype=np.int64)
    w_lin = np.zeros((length),dtype=np.float16)
    counter = 0
    for node in range(len(indices)):
        idxs = indices[node]
        ws = weights[node]
        for i, w in zip(idxs, ws):
            r_lin[counter]=node
            c_lin[counter]=i
            w_lin[counter]=w
            if i<0:
                print("terrible badness")
                print(idxs)
                print(ws)
                print(poo)
            counter+=1
    return coo_matrix((w_lin,(r_lin, c_lin)))


def combine_and_sort_distances(common_edges,
                               unique_train_edges,
                               unique_val_edges,
                               avg_common_distances, 
                               avg_unique_train_distances, 
                               avg_unique_val_distances):
    n_edges_per_node = []
    merged_index_list = []
    merged_dist_list = []
    merged_weight_list = []
    for ci, ti, vi, cd, td, vd in zip(common_edges,
                        unique_train_edges,
                        unique_val_edges,
                        avg_common_distances, 
                        avg_unique_train_distances, 
                        avg_unique_val_distances):
        temp_idx = torch.cat((ci, ti, vi))
        if min(temp_idx)<0:
            print("terrible badness1:")
            print(ci)
            print(ti)
            print(vi)
            print(poo)
        temp_dist = torch.cat((cd, td, vd))
        # resort based on the merged distances
        new_order = torch.argsort(temp_dist)
        temp_idx = temp_idx[new_order]
        temp_dist = temp_dist[new_order]
        # Now append them
        merged_index_list.append(temp_idx)
        merged_dist_list.append(temp_dist)
        merged_weight_list.append(distances_to_weights(temp_dist))
        # Add to the running total so we know how big to make the destination matrices
        n_edges_per_node.append(temp_idx.shape[0])
    #n_nodes = common_edges.shape[0]
    print("min edges found were:",min(n_edges_per_node))
    print("maximum edges found were:",max(n_edges_per_node))
    merged_graph = indices_and_weights_to_graph(merged_index_list, merged_weight_list, sum(n_edges_per_node))
    ## Now re-collate them into their final destination format
    #merged_indices = torch.zeros((n_nodes,max_edges),dtype=torch.int64)
    #merged_distances = torch.zeros((n_nodes,max_edges),dtype=torch.float32)
    #merged_mask = torch.zeros((n_nodes,max_edges),dtype=torch.bool)
    #return #merged_indices, merged_distances, merged_weight_list
    return merged_index_list, merged_dist_list, merged_weight_list, merged_graph


def fix_edge(cur_index, cur_distance, cur_mask, ref_index, ref_distance, ref_mask):
    for i in range(cur_index.shape[0]):
        if min(cur_index[i])<0:
            print("found weird edge case:",i)
            # This means that svd failed and propogated through to nearest neighbor
            # which returns -1s if there are nans
            cur_index[i]=ref_index[i]
            cur_distance[i]=ref_distance[i]
            cur_mask[i]=ref_mask[i]
    return(cur_index, cur_distance, cur_mask)


def merge_knns(train_index, train_distance, train_mask, val_index, val_distance, val_mask):
    ## first find and fix edge cases where svd failed & copy results from other split
    train_index, train_distance, train_mask = fix_edge(train_index, train_distance, train_mask, 
                                                       val_index, val_distance, val_mask)
    val_index, val_distance, val_mask = fix_edge(val_index, val_distance, val_mask,
                                                 train_index, train_distance, train_mask)
    print("\tmerging the knns")
    for i in range(len(val_index)):
        if min(val_index[i])<0:
            print("mega bad0:")
            print(val_index[i])
            print(poo)
    num_observations, k = train_index.shape
    max_train_distance = calculate_maximum_distances(train_distance, train_mask)
    max_val_distance = calculate_maximum_distances(val_distance, val_mask)
    # Step 1: Identify common and unique edges per node
    common_edges, unique_train_edges, unique_val_edges = identify_common_unique_edges(train_index, val_index, train_mask, val_mask)
    
    # Step 2: Compute average distances for common edges
    avg_common_distances = compute_average_common_distances(common_edges, train_index, val_index, train_distance, val_distance)
    
    # Step 3: Compute averages for unique edges (train and validation)
    avg_unique_train_distances = compute_average_unique_distances(unique_train_edges, train_index, train_distance, max_val_distance)
    avg_unique_val_distances = compute_average_unique_distances(unique_val_edges, val_index, val_distance, max_train_distance)
    
    # Step 4: Combine and sort all of them
    merged_indices, merged_distances, merged_weights, merged_graph = combine_and_sort_distances(common_edges,
                                                  unique_train_edges,
                                                  unique_val_edges,
                                                  avg_common_distances, 
                                                  avg_unique_train_distances, 
                                                  avg_unique_val_distances)
    return merged_indices, merged_distances, merged_weights, merged_graph



#merged_indices, merged_distances, merged_weights = merge_knns(train_index, train_distance, train_mask, val_index, val_distance, val_mask)


############################################################################


def find_one_graph(pcs, k, cosine=True, use_gpu = False):
    if k is None:
        # Default to the log of the number of observations
        # The mi
        k= int(round(np.log(pcs.shape[0]),
                     0)
               )
    # bound k between 20 and 200, but follow the above heuristic or user guidance otherwise
    k = min(k, 200)
    index = get_faiss_idx(pcs, cosine = cosine, use_gpu = use_gpu)
    indexes, distances = find_k_nearest_neighbors(index, pcs, k)
    ## filter using slicer method to get mask
    local_mask = mask_knn_local_diff_dist(
        torch.tensor(distances), 
        min_k=max(ceil(k/2),3)
    )
    #for i in range(len(indexes)):
    #    if min(indexes[i])<0:
    #        print("mega bad -1:")
    #        print(indexes[i])
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
    index.add(torch.tensor(vectors))
    return(index)


def find_k_nearest_neighbors(index, vectors, k):
    """ Find k-nearest neighbors for each vector. """
    cos_sim, indices = index.search(vectors, k)
    count=0
    ## 1-cosine similarity is cosine distance
    return torch.tensor(indices), 1-torch.tensor(cos_sim)


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
    temp_diff_mask_cutoff = temp_diff_mask_cutoff.bool() & prior_mask_diff.bool()
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
    diff_mask = diff_mask.bool() & prior_mask.bool()
    diff_mask[:, (min_k+1):] = temp_diff_mask_cutoff
    #print("diff_mask after local diff masking")
    #print(diff_mask)
    print("mean number connections after local masking:")
    print(torch.mean(torch.sum(diff_mask.float(), dim=1)))
    print("min number connections after local masking:")
    print(torch.min(torch.sum(diff_mask.float(), dim=1)))
    return diff_mask


#########################################################################


#find_consensus_graph(ro.train_pc, ro.val_pc, ro.initial_k, cosine = True, use_gpu = False)


