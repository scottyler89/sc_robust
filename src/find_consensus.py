import faiss
import numpy as np
from typing import Int, Optional, Any, List, Float
from sklearn import metrics
from sklearn.metrics import pairwise
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances as euc


def find_pcs(train_mat: Any, 
             val_mat: Any, 
             genes: List,
             max_pcs: Optional[Int] = 250,
             pc_threshold: Optional[Float] = 0.1,
             k: Optional[Float]=None):
    if k is None:
        # Default to the log of the number of observations
        k = max([200,int(np.log(train_mat.shape[0]))])
    # Keep only the genes that are expressed in both
    train_idxs = sorted(list(set(train_mat.indices)))
    val_idxs = sorted(list(set(val_mat.indices)))
    both_idxs = [idx for idx in val_idxs if idx in train_idxs]
    train_pc = tsvd(train_mat[:,both_idxs], npcs=max_pcs)
    val_pc = tsvd(val_mat[:,both_idxs], npcs=max_pcs)
    
    return



def tsvd(temp_mat, npcs=250):
    pca = TruncatedSVD(n_components=npcs,n_iter=7, random_state=42)
    out_pcs = pca.fit_transform(temp_mat.T)
    return(out_pcs)


def keep_correlated_pcs(train_pcs, val_pcs):
    ## 
    
    return


def find_consensus_graph(train_pcs, val_pcs, cosine = True, use_gpu = False):
    train_neigh, train_dist = find_one_graph(train_pcs, cosine = cosine, use_gpu = use_gpu)
    val_neigh, val_dist = find_one_graph(train_pcs, cosine = cosine, use_gpu = use_gpu)
    return


def find_one_graph(pcs, k, cosine=True, use_gpu = False):
    index = get_faiss_idx(pcs, cosine = cosine, use_gpu = use_gpu)
    indexes, distances = find_k_nearest_neighbors(index, pcs, k)
    ## filter using slicer method to get mask
    local_mask = 
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
    distances, indices = index.search(vectors, k)
    return indices, distances



# Example usage
point_cloud_data = np.random.random((100000, 250))  # Example data
k = 10  # Number of nearest neighbors
neighbor_graph, distance_matrix = find_one_graph(point_cloud_data, k)

