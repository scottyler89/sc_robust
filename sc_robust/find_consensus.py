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
import logging

# Gated debug printing for this module
DEBUG = False
logger = logging.getLogger(__name__)


def find_pcs(train_mat: Any, 
             val_mat: Any,
             pc_max: Optional[int] = 250,
             do_plot: Optional[bool] = False,
             random_state: Optional[int] = None):
    if DEBUG:
        logger.debug("Decomposing training and validation matrices")
    ## Sanity check
    if DEBUG:
        logger.debug("train_mat.shape=%s", getattr(train_mat, "shape", None))
        logger.debug("val_mat.shape=%s", getattr(val_mat, "shape", None))
        logger.debug("pc_max=%s", pc_max)
    pc_max = min(pc_max,min(train_mat.shape), min(val_mat.shape))
    # Keep only the genes that are expressed in both
    train_idxs = sorted(list(set(train_mat.indices)))
    val_idxs = sorted(list(set(val_mat.indices)))
    both_idxs = [idx for idx in val_idxs if idx in train_idxs]
    train_pc = tsvd(train_mat, npcs=pc_max)
    val_pc = tsvd(val_mat, npcs=pc_max)
    if DEBUG:
        logger.debug("performing empirical validation")
    train_keep, val_keep = keep_correlated_pcs(train_pc, val_pc, do_plot=do_plot, random_state=random_state)
    train_pc = train_pc[:, train_keep]
    val_pc = val_pc[:, val_keep]
    if DEBUG:
        logger.debug("final dimensionality training=%s validation=%s", train_pc.shape, val_pc.shape)
    return train_pc, val_pc


def tsvd(temp_mat, npcs=250):
    """Compute truncated SVD components for a matrix (dense or sparse).

    Parameters:
      temp_mat: Array-like or sparse matrix (samples x features).
      npcs: Number of components to compute (default 250).

    Returns:
      np.ndarray of shape (n_samples, npcs) with NaNs replaced by zeros.
    """
    pca = TruncatedSVD(n_components=npcs, n_iter=7, random_state=42)
    out_pcs = pca.fit_transform(temp_mat)
    # Guard against rare NaNs
    out_pcs[np.isnan(out_pcs)] = 0.0
    return out_pcs


def keep_correlated_pcs(train_pc, val_pc, alpha=0.01, n_boot = 50, do_plot = False, random_state: Optional[int] = None):
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
    rng = np.random.default_rng(random_state) if random_state is not None else np.random.default_rng()
    for b in range(n_boot):
        ## get the null distributions
        for col in range(train_pc.shape[1]):
            # go through the columns (ie: pcs)
            # & shuffle them 
            train_shuff[:,col]=rng.permutation(train_shuff[:,col])
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
    """Build a consensus graph from train/val PCs via KNN, masking, and merging.

    Parameters:
      train_pcs: Array-like (n_cells, n_dims) for the training split PCs/embedding.
      val_pcs: Array-like (n_cells, n_dims) for the validation split PCs/embedding.
      initial_k: Int for initial KNN size; defaults to round(log(n)) when None.
      cosine: If True, use cosine similarity (inner product on normalized rows).
      use_gpu: If True and FAISS GPU is available, perform GPU search.

    Returns:
      (indices, distances, weights, graph): lists of merged neighbors/distances/weights
      per node, and a weighted COO adjacency matrix.
    """
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
        if temp_common.numel() > 0 and torch.min(temp_common) < 0:
            raise ValueError("Negative neighbor index detected during intersection; check K against dataset size.")
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
    """Convert a sorted per-node distance vector to weights via linear rescaling.

    Assumes the first entry is the self-distance; leaves it at weight 1.
    Non-self distances are shifted to start near 0, scaled by the max, inverted,
    and shifted to (0,1], yielding larger weights for nearer neighbors.
    """
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


def process_idx_dist_mask_to_g(indexes, distances, local_mask):
    """
    Process the input indexes, distances, and local_mask arrays to filter out unwanted
    connections, compute weights, and prepare the data for building a sparse graph.
    This version uses PyTorch tensors and returns a torch.sparse_coo_tensor.
    
    Parameters:
        indexes (list): A list of arrays/lists or torch tensors, each containing
            neighbor indices for each node.
        distances (list): A list of arrays/lists or torch tensors, each containing
            the distances corresponding to the neighbor indices.
        local_mask (list): A list of boolean arrays/lists or torch tensors with the
            same shapes as the corresponding indexes and distances. True values indicate
            the connection should be kept.
    
    Returns:
        scipy.sparse.coo_matrix: Weighted adjacency with shape (n_nodes, n_nodes).
    """
    processed_indices = []
    processed_weights = []
    n_edges_per_node = []
    
    for idx, dist, mask in zip(indexes, distances, local_mask):
        # Convert to torch tensors if they aren't already
        if not isinstance(idx, torch.Tensor):
            idx_tensor = torch.tensor(idx, dtype=torch.int64)
        else:
            idx_tensor = idx
        
        if not isinstance(dist, torch.Tensor):
            dist_tensor = torch.tensor(dist, dtype=torch.float32)
        else:
            dist_tensor = dist
        
        if not isinstance(mask, torch.Tensor):
            mask_tensor = torch.tensor(mask, dtype=torch.bool)
        else:
            mask_tensor = mask
        
        # Apply the local mask to filter indices and distances
        filtered_idx = idx_tensor[mask_tensor]
        filtered_dist = dist_tensor[mask_tensor]
        
        # Compute weights from the filtered distances.
        # Assumes distances_to_weights is a torch-compatible function.
        weights = distances_to_weights(filtered_dist)
        
        processed_indices.append(filtered_idx)
        processed_weights.append(weights)
        n_edges_per_node.append(filtered_idx.shape[0])
    
    # Total number of edges across all nodes
    total_edges = sum(n_edges_per_node)
    
    # Create and return the sparse graph using the helper function.
    graph = indices_and_weights_to_graph(processed_indices, processed_weights, total_edges)
    return graph


def indices_and_weights_to_graph(indices, weights, length):
    """Assemble a COO adjacency from per-node neighbor indices and weights.

    Skips self-edges and truncates preallocated arrays to the actual edge count
    to avoid zero-initialized artifacts.
    """
    r_lin = np.zeros((length), dtype=np.int64)
    c_lin = np.zeros((length), dtype=np.int64)
    w_lin = np.zeros((length), dtype=np.float32)
    counter = 0
    for node in range(len(indices)):
        idxs = indices[node]
        ws = weights[node]
        for i, w in zip(idxs, ws):
            ## exclude self connections
            if node != i:
                r_lin[counter]=node
                c_lin[counter]=i
                w_lin[counter]=w
                if i < 0:
                    raise ValueError("Negative neighbor index encountered while building graph.")
                counter+=1
    # Truncate arrays to actual edge count to avoid zero-weight artifacts
    r_lin = r_lin[:counter]
    c_lin = c_lin[:counter]
    w_lin = w_lin[:counter]
    return coo_matrix((w_lin,(r_lin, c_lin)))


def combine_and_sort_distances(common_edges,
                               unique_train_edges,
                               unique_val_edges,
                               avg_common_distances, 
                               avg_unique_train_distances, 
                               avg_unique_val_distances):
    """Combine per-node common/unique edges, sort by merged distances, and weight."""
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
        if temp_idx.numel() > 0 and torch.min(temp_idx) < 0:
            raise ValueError("Negative neighbor index encountered while merging KNNs.")
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
    if DEBUG:
        logger.debug("min edges found=%s max edges found=%s", min(n_edges_per_node), max(n_edges_per_node))
    merged_graph = indices_and_weights_to_graph(merged_index_list, merged_weight_list, sum(n_edges_per_node))
    ## Now re-collate them into their final destination format
    #merged_indices = torch.zeros((n_nodes,max_edges),dtype=torch.int64)
    #merged_distances = torch.zeros((n_nodes,max_edges),dtype=torch.float32)
    #merged_mask = torch.zeros((n_nodes,max_edges),dtype=torch.bool)
    #return #merged_indices, merged_distances, merged_weight_list
    return merged_index_list, merged_dist_list, merged_weight_list, merged_graph


def fix_edge(cur_index, cur_distance, cur_mask, ref_index, ref_distance, ref_mask):
    """Replace invalid (-1) neighbor rows using the reference split.

    This mitigates rare SVD/FAISS edge cases where NaNs propagate and yield -1 indices.
    """
    for i in range(cur_index.shape[0]):
        if min(cur_index[i])<0:
            if DEBUG:
                logger.debug("found weird edge case i=%s", i)
            # This means that svd failed and propogated through to nearest neighbor
            # which returns -1s if there are nans
            cur_index[i]=ref_index[i]
            cur_distance[i]=ref_distance[i]
            cur_mask[i]=ref_mask[i]
    return(cur_index, cur_distance, cur_mask)


def merge_knns(train_index, train_distance, train_mask, val_index, val_distance, val_mask):
    """Merge train/val KNNs by averaging common edges and penalizing uniques.

    Unique edges are averaged with the other split's max distance to downweight
    edges not reproduced across splits.
    """
    ## first find and fix edge cases where svd failed & copy results from other split
    train_index, train_distance, train_mask = fix_edge(train_index, train_distance, train_mask, 
                                                       val_index, val_distance, val_mask)
    val_index, val_distance, val_mask = fix_edge(val_index, val_distance, val_mask,
                                                 train_index, train_distance, train_mask)
    if DEBUG:
        logger.debug("merging the knns")
    for i in range(len(val_index)):
        if torch.min(val_index[i]) < 0:
            raise ValueError("Negative index in validation KNN; ensure k <= n and inputs have no NaNs.")
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


def find_one_graph(pcs, k=None, metric: Optional[str] = None, cosine: bool = True, use_gpu: bool = False):
    """Find per-row KNN neighbors, distances, and a local-difference mask.

    Parameters:
      pcs: Array-like (n_samples, n_dims).
      k: Optional int; defaults to round(log(n)), capped by 200 and n.
      metric: 'cosine' | 'l2' | 'ip'. If None, falls back to `cosine` flag.
      cosine: Legacy switch for backward compatibility (ignored if `metric` given).
      use_gpu: Use FAISS GPU if available.

    Returns:
      (indices, distances, mask): torch tensors for neighbors, distances, and kept edges.
    """
    if k is None:
        # Default to the log of the number of observations
        k= int(round(np.log(pcs.shape[0]), 0))
    n = int(pcs.shape[0])
    k = int(k)
    k = min(k, 200, n)
    k = max(k, min(10, n))
    # Determine metric for backward compatibility: cosine flag controls default unless metric provided
    chosen_metric = metric if metric is not None else ("cosine" if cosine else "ip")
    index = get_faiss_idx(pcs, metric=chosen_metric, cosine=cosine, use_gpu=use_gpu)
    indexes, distances = find_k_nearest_neighbors(index, pcs, k, metric=chosen_metric, cosine=cosine)
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


def get_faiss_idx(vectors, metric: Optional[str] = None, cosine: bool = True, use_gpu: bool = False):
    """Create a FAISS IndexFlatIP for cosine/dot-product search.

    When `cosine=True`, rows are L2-normalized with zero-norm rows guarded.
    FAISS expects `np.float32` arrays; torch tensors are converted.
    """
    # Ensure numpy float32 for FAISS
    if isinstance(vectors, torch.Tensor):
        vec = vectors.detach().cpu().numpy().astype(np.float32)
    else:
        vec = np.asarray(vectors, dtype=np.float32)
    chosen_metric = metric if metric is not None else ("cosine" if cosine else "ip")
    if chosen_metric == "cosine":
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        safe = norms.copy()
        safe[safe == 0] = 1.0
        vec = vec / safe
    dimension = vec.shape[1]
    if chosen_metric == "l2":
        index = faiss.IndexFlatL2(dimension)
    else:
        # 'cosine' or 'ip' both use inner product index
        index = faiss.IndexFlatIP(dimension)
    if use_gpu:
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(vec)
    return index


def find_k_nearest_neighbors(index, vectors, k, metric: Optional[str] = None, cosine: bool = True):
    """Find k-nearest neighbors via FAISS; returns indices and distances.

    Distances depend on the metric:
      - 'cosine': 1 - cosine_similarity
      - 'l2': squared Euclidean distance returned by FAISS
      - 'ip': negative inner product (pseudo-distance)
    """
    if isinstance(vectors, torch.Tensor):
        vec = vectors.detach().cpu().numpy().astype(np.float32)
    else:
        vec = np.asarray(vectors, dtype=np.float32)
    chosen_metric = metric if metric is not None else ("cosine" if cosine else "ip")
    if chosen_metric == "l2":
        # FAISS L2 returns squared L2 distances directly
        dists, indices = index.search(vec, k)
        return torch.tensor(indices, dtype=torch.int64), torch.tensor(dists, dtype=torch.float32)
    else:
        sims, indices = index.search(vec, k)
        if chosen_metric == "cosine":
            dists = 1.0 - sims
        else:  # 'ip' raw inner product -> pseudo-distance
            dists = -sims
        return torch.tensor(indices, dtype=torch.int64), torch.tensor(dists, dtype=torch.float32)


##############################################################
# KNN masking functions
def masked_mad(input_tensor, mask, epsilon = 1e-8):
    """Compute median and MAD per-row with a boolean mask (approximate).

    Note: uses element-wise multiplication by the mask; for very sparse masks,
    medians may skew toward 0. Good enough for relative thresholding of diffs.
    """
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
    """Standardize input by row-wise (masked) median/MAD for robust cutoffing."""
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
    if DEBUG:
        logger.debug("dists.shape=%s min_k=%s", getattr(dists, "shape", None), min_k)
        logger.debug(
            "mean connections before masking=%.3f",
            float(torch.mean(torch.sum(prior_mask.float(), dim=1)).item()),
        )
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
    if DEBUG:
        logger.debug(
            "mean connections after masking=%.3f min connections=%s",
            float(torch.mean(torch.sum(diff_mask.float(), dim=1)).item()),
            int(torch.min(torch.sum(diff_mask.float(), dim=1)).item()),
        )
    return diff_mask


#########################################################################


#find_consensus_graph(ro.train_pc, ro.val_pc, ro.initial_k, cosine = True, use_gpu = False)
