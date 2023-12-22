import faiss
import torch
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from copy import deepcopy
from scipy.stats import norm
from sklearn.decomposition import TruncatedSVD
from anticor_features.anticor_stats import no_p_pear as pearson
from typing import Optional, Any


def find_pcs(train_mat: Any, 
             val_mat: Any,
             pc_max: Optional[int] = 250,
             k: Optional[float]=None):
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
    train_keep, val_keep = keep_correlated_pcs(train_pc, val_pc)
    train_pc = train_pc[:,train_keep]
    val_pc = val_pc[:,val_keep]
    print("\tfinal dimensionality:")
    print("\t\ttraining:",train_pc.shape)
    print("\t\tvalidation:",val_pc.shape)
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
    return include_train, include_val




################################################
################################################
################################################
## Find the graphs
def find_consensus_graph(train_pcs, val_pcs, initial_k, cosine = True, use_gpu = False):
    train_index, train_distance, train_mask = find_one_graph(train_pcs, initial_k, cosine = cosine, use_gpu = use_gpu)
    val_index, val_distance, val_mask = find_one_graph(val_pcs, initial_k, cosine = cosine, use_gpu = use_gpu)
    #merge_index, merge_distance, merge_mask = merge_knns(train_index, train_distance, train_mask, val_index, val_distance, val_mask)
    return #merge_index, merge_distance, merge_mask


############################################################################
def calculate_maximum_distances(distance_matrix, mask_matrix):
    """ Calculates the maximum included distance for each node """
    masked_distances = distance_matrix.masked_fill(~mask_matrix, float('-inf'))
    return masked_distances.max(dim=1).values


## Helper functions for finding which edges are replicated in both, vs unique to one or the other
def select_valid_indices(index, mask):
    # These are lists of tensors that contain only the pertinent non-masked indices
    return [index[row][mask[row]] for row, _ in enumerate(index)]


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
        all_common.append(temp_common)
        all_unique.append(temp_unique)
    return all_common, all_unique


def find_unique_indices(train_indices, val_indices, common_indices):
    unique_train_indices = [torch.setdiff1d(t_indices, c_indices) for t_indices, c_indices in zip(train_indices, common_indices)]
    unique_val_indices = [torch.setdiff1d(v_indices, c_indices) for v_indices, c_indices in zip(val_indices, common_indices)]
    return unique_train_indices, unique_val_indices

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
    common_indices = find_common_indices(valid_train_indices, valid_val_indices)
    # Find indices unique to the training set for each node
    unique_train_indices, unique_val_indices = find_unique_indices(valid_train_indices, valid_val_indices, common_indices)
    return common_indices, unique_train_indices, unique_val_indices


##
def merge_knns(train_index, train_distance, train_mask, val_index, val_distance, val_mask):
    num_observations, k = train_index.shape
    max_train_distance = calculate_maximum_distances(train_distance, train_mask)
    max_val_distance = calculate_maximum_distances(val_distance, val_mask)
    # Step 1: Identify common and unique edges per node
    common_edges, unique_train_edges, unique_val_edges = identify_common_unique_edges(train_index, val_index, train_mask, val_mask)
    
    # Step 2: Compute average distances for common edges
    avg_common_distances = compute_average_common_distances(common_edges, train_distance, val_distance)
    
    # Step 3: Compute averages for unique edges (train and validation)
    max_train_distance = calculate_maximum_distances(train_distance, train_mask)
    avg_unique_train_distances = compute_average_unique_distances(unique_train_edges, train_distance, max_train_distance)
    
    max_val_distance = calculate_maximum_distances(val_distance, val_mask)
    avg_unique_val_distances = compute_average_unique_distances(unique_val_edges, val_distance, max_val_distance)
    
    # Step 4: Combine and sort distances
    sorted_distances = combine_and_sort_distances(avg_common_distances, avg_unique_train_distances, avg_unique_val_distances)
    
    # ... Continue with similar logic for indices and masks
    
    return sorted_distances  # Placeholder, replace with actual outputs (indices, distances, masks)

############################################################################


def find_one_graph(pcs, k, cosine=True, use_gpu = False):
    if k is None:
        # Default to the log of the number of observations
        # The mi
        k= int(round(np.log(pcs.shape[0])/2 +
                     np.sqrt(pcs.shape[0])/2,
                     0)
               )
    # bound k between 20 and 200, but follow the above heuristic or user guidance otherwise
    k = min(max([20,k]), 200)
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
    return diff_mask


#########################################################################


find_consensus_graph(ro.train_pc, ro.val_pc, ro.initial_k, cosine = True, use_gpu = False)


