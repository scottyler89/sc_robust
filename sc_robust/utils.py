import pandas as pd
import numpy as np
import scipy.sparse as sp
import igraph as ig
import leidenalg



#def dynamic_fs_df_reprocessing(fs_df):
#    ## If selected is zero, relax selection criteria to 
#    if fs_df[""]
#    (fs_df["num_sig_pos_cor"]>=10) & (fs_df["FDR"]<0.10)
    


def coo_matrix_to_igraph(coo_mat):
    """
    Convert a weighted COO matrix to an igraph Graph.
    Assumes the COO matrix represents an undirected graph.
    """
    # Ensure the matrix is in COO format
    if not sp.isspmatrix_coo(coo_mat):
        coo_mat = coo_mat.tocoo()
    
    # Extract row, col, and data arrays from the COO matrix
    sources = coo_mat.row.tolist()
    targets = coo_mat.col.tolist()
    weights = coo_mat.data.tolist()
    
    # Create an undirected graph with the number of vertices equal to the matrix dimension
    n_vertices = coo_mat.shape[0]
    g = ig.Graph(n_vertices, directed=False)
    
    # Add edges to the graph
    edges = list(zip(sources, targets))
    g.add_edges(edges)
    
    # Assign edge weights
    g.es['weight'] = weights
    
    return g

def perform_leiden_clustering(coo_mat, resolution_parameter=1.0):
    """
    Convert the COO matrix to an igraph graph and perform Leiden clustering.
    
    Parameters:
      coo_mat: A scipy.sparse COO matrix representing the weighted adjacency matrix.
      resolution_parameter: A float that controls the resolution of the clustering.
    
    Returns:
      clusters: A list of lists, where each inner list contains the node indices of a cluster.
      partition: The partition object returned by the leidenalg library.
    """
    # Convert the COO matrix into an igraph graph
    g = coo_matrix_to_igraph(coo_mat)
    
    # Run Leiden clustering
    partition = leidenalg.find_partition(
        g, 
        leidenalg.RBConfigurationVertexPartition, 
        weights='weight', 
        resolution_parameter=resolution_parameter
    )
    
    # Extract clusters from the partition object
    clusters = [cluster for cluster in partition]
    
    # Create a NumPy array for cluster labels per node
    labels = np.empty(g.vcount(), dtype=int)
    for cluster_idx, cluster in enumerate(partition):
        for node in cluster:
            labels[node] = cluster_idx
    
    return clusters, partition, labels


def load_ro(f):
    """
    Load a robust object from a file using dill.
    """
    with open(f, 'rb') as file:
        ro = dill.load(file)
    return ro
