import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import pymetis

def make_symmetric(in_graph):
    """
    Ensure the input COO matrix is symmetric by taking the elementwise maximum
    between the matrix and its transpose.
    
    Parameters:
      in_graph (coo_matrix): Input weighted adjacency matrix in COO format.
    
    Returns:
      coo_matrix: A symmetric COO matrix.
    """
    csr = in_graph.tocsr()
    symmetric_csr = csr.maximum(csr.transpose())
    return symmetric_csr.tocoo()

def filter_top_edges(csr, top_n):
    """
    For each row in the CSR matrix, retain only the top_n highest-weight edges.
    
    Parameters:
      csr (csr_matrix): Input graph in CSR format.
      top_n (int): Number of top edges to retain per row.
      
    Returns:
      csr_matrix: A new CSR matrix with each row filtered to top_n edges.
    """
    new_data = []
    new_indices = []
    new_indptr = [0]
    
    for i in range(csr.shape[0]):
        row_start = csr.indptr[i]
        row_end = csr.indptr[i+1]
        row_data = csr.data[row_start:row_end]
        row_indices = csr.indices[row_start:row_end]
        
        if len(row_data) > top_n:
            # Select indices for the top_n weights (largest first)
            order = np.argsort(row_data)[::-1][:top_n]
            filtered_data = row_data[order]
            filtered_indices = row_indices[order]
        else:
            filtered_data = row_data
            filtered_indices = row_indices
        
        new_data.extend(filtered_data)
        new_indices.extend(filtered_indices)
        new_indptr.append(len(new_data))
        
    return csr_matrix((np.array(new_data), np.array(new_indices), np.array(new_indptr)), shape=csr.shape)

def aggregate_cell_meta(meta_df, cell_indices, weights):
    """
    Aggregate cell-level metadata for a set of cells using provided weights.
    
    For numeric columns, compute a weighted average (return 0 if total weight is 0).
    For categorical columns, perform a weighted one-hot encoding.
      For each unique category in the column, create a new key with the format
      "agg_<column>__<category>" and set its value to the weighted fraction of cells
      in that category.
    
    Parameters:
      meta_df (pd.DataFrame): Cell-level metadata.
      cell_indices (list): List of cell indices in the current partition.
      weights (np.array): Array of weights (e.g., total count per cell) for the cells.
    
    Returns:
      dict: Aggregated metadata with keys prefixed by "agg_".
    """
    agg = {}
    sub_meta = meta_df.iloc[cell_indices]
    total_weight = weights.sum()
    
    for col in meta_df.columns:
        if pd.api.types.is_numeric_dtype(meta_df[col]):
            vals = sub_meta[col].astype(float)
            agg_val = (vals * weights).sum() / total_weight if total_weight > 0 else 0
            agg["agg_" + col] = agg_val
        else:
            unique_categories = sub_meta[col].unique()
            for cat in unique_categories:
                mask = (sub_meta[col].values == cat)
                cat_weight = weights[mask].sum()
                fraction = cat_weight / total_weight if total_weight > 0 else 0
                key = f"agg_{col}__{cat}"
                agg[key] = fraction
    return agg

def prep_sample_pseudobulk(in_graph, X, cells_per_pb=10, sample_vect=None, cluster_vect=None,
                           gene_ids=None, coords=None, cell_meta=None):
    """
    Prepares pseudobulk samples by partitioning cells using METIS and aggregating gene expression counts.
    
    Parameters:
      in_graph (COO matrix): Weighted adjacency matrix of cells in COO format.
      X (np.ndarray): Gene expression matrix (cells x genes).
      cells_per_pb (int): Target number of cells per pseudobulk sample.
      sample_vect (np.ndarray or list): Array of sample labels for each cell.
      cluster_vect (np.ndarray or list): Array of cluster labels for each cell.
      gene_ids (list or array, optional): Gene identifiers corresponding to the columns of X.
      coords (np.ndarray, optional): A 2D array of shape (n_cells, 2) with 2D coordinates (e.g., UMAP/t-SNE) for each cell.
      cell_meta (pd.DataFrame, optional): Cell-level metadata DataFrame with row order corresponding to cells.
    
    Returns:
      pb_exprs: Aggregated pseudobulk expression matrix.
                If gene_ids is provided, returns a pd.DataFrame with columns labeled by gene_ids;
                otherwise, returns a numpy array.
      annotation_df (pd.DataFrame): DataFrame with metadata for each pseudobulk sample including:
                                    - pb_id: partition ID (contiguous 0-indexed)
                                    - cell_n: number of cells in the partition
                                    - sample_proportions: relative contribution of sample labels
                                    - cluster_proportions: relative contribution of cluster labels
                                    - source_cells: list of cell indices contributing to the partition
                                    - total_count_sum: total count sum aggregated from all cells in the partition
                                    - pb_coord_x, pb_coord_y (if coords provided): weighted average coordinates
                                    - Aggregated cell meta (if cell_meta provided): aggregated numeric and one-hot encoded categorical fields.
    """
    x_shape = getattr(X, "shape", None)
    if getattr(X, "ndim", 2) != 2 or not x_shape or len(x_shape) != 2:
        raise ValueError(f"X must be a 2D matrix with shape (n_cells, n_genes); got shape={x_shape}.")
    n_cells = X.shape[0]
    
    # Check in_graph format.
    if not isinstance(in_graph, coo_matrix):
        raise ValueError("in_graph must be a COO matrix. Please convert it using scipy.sparse.coo_matrix.")
    
    # Make the graph symmetric.
    in_graph = make_symmetric(in_graph)
    
    # Pre-filter the symmetric graph to include only top (cells_per_pb) edges per row.
    csr_graph = in_graph.tocsr()
    csr_graph = filter_top_edges(csr_graph, max([2,int(cells_per_pb/2)]))
    in_graph = make_symmetric(csr_graph.tocoo())
    
    # Basic input checks.
    if in_graph.shape[0] != in_graph.shape[1]:
        raise ValueError(f"The adjacency graph is not square: shape={in_graph.shape}.")
    if in_graph.shape[0] != n_cells:
        raise ValueError(
            "Expression matrix X must be cells×genes with the same number of rows as the graph; "
            f"graph.shape={in_graph.shape} X.shape={x_shape}. "
            "If you provided genes×cells, transpose X."
        )
    
    # Process sample and cluster vectors.
    if sample_vect is None:
        sample_vect = np.array(["NA"] * n_cells)
    else:
        sample_vect = np.array(sample_vect)
    if cluster_vect is None:
        cluster_vect = np.array(["NA"] * n_cells)
    else:
        cluster_vect = np.array(cluster_vect)
    
    # Check gene_ids if provided.
    if gene_ids is not None:
        if len(gene_ids) != X.shape[1]:
            raise ValueError(
                f"Length of gene_ids ({len(gene_ids)}) must match the number of columns in X ({X.shape[1]})."
            )
    
    # If coords is provided, check its shape.
    if coords is not None:
        if coords.shape[0] != n_cells or coords.shape[1] != 2:
            raise ValueError("coords must be of shape (n_cells, 2).")
    
    # Determine number of partitions.
    n_parts = int(np.ceil(n_cells / cells_per_pb))
    
    # Convert filtered graph to CSR.
    csr_graph = in_graph.tocsr()
    xadj = csr_graph.indptr.tolist()
    adjncy = csr_graph.indices.tolist()
    adjwgt = csr_graph.data.tolist()
    
    # Scale edge weights to integers.
    scale_factor = 1000  # Scale weights from [1e-3, 1] roughly to [1, 1000]
    adjwgt_int = [int(round(w * scale_factor)) for w in adjwgt]
    
    # Partition the graph using METIS.
    edgecut, partitioning = pymetis.part_graph(n_parts, xadj=xadj, adjncy=adjncy, eweights=adjwgt_int)
    
    # Remap partition labels to contiguous range.
    unique_labels = sorted(set(partitioning))
    mapping = {old: new for new, old in enumerate(unique_labels)}
    new_partitioning = [mapping[p] for p in partitioning]
    
    # Group cell indices by partition using the remapped labels.
    partitions = {}
    for cell_idx, part in enumerate(new_partitioning):
        partitions.setdefault(part, []).append(cell_idx)
    
    # Precompute per-cell total counts (used for weighting coordinates and cell meta).
    cell_total_counts = X.sum(axis=1)  # shape: (n_cells,)
    
    pb_ids = []
    pb_expr_list = []
    annotation_list = []
    
    # Iterate over partitions.
    for pb_id, cell_indices in partitions.items():
        pb_ids.append(pb_id)
        cell_indices = np.array(cell_indices)
        cell_n = len(cell_indices)
        
        # Sum the expression counts for these cells.
        pb_expr = X[cell_indices, :].sum(axis=0)
        #pb_expr = np.asarray(X[cell_indices, :]).sum(axis=0).ravel()
        pb_expr_list.append(pb_expr)
        
        # Total count sum for this partition.
        total_count_sum = pb_expr.sum()
        
        # Calculate sample and cluster proportions.
        sample_counts = pd.Series(sample_vect[cell_indices]).value_counts(normalize=True).to_dict()
        cluster_counts = pd.Series(cluster_vect[cell_indices]).value_counts(normalize=True).to_dict()
        
        # Compute weighted average coordinates if provided.
        pb_coord_x = pb_coord_y = 0
        if coords is not None:
            weights = cell_total_counts[cell_indices]
            if weights.sum() > 0:
                pb_coord_x = (coords[cell_indices, 0] * weights).sum() / weights.sum()
                pb_coord_y = (coords[cell_indices, 1] * weights).sum() / weights.sum()
            # Else, remain 0.
        
        # Aggregate cell meta if provided.
        agg_meta = {}
        if cell_meta is not None:
            weights = cell_total_counts[cell_indices]
            agg_meta = aggregate_cell_meta(cell_meta, cell_indices, weights)
        
        # Build annotation dictionary.
        ann = {
            "pb_id": pb_id,
            "cell_n": cell_n,
            "sample_proportions": sample_counts,
            "cluster_proportions": cluster_counts,
            "source_cells": list(cell_indices),
            "total_count_sum": total_count_sum,
        }
        if coords is not None:
            ann["pb_coord_x"] = pb_coord_x
            ann["pb_coord_y"] = pb_coord_y
        if cell_meta is not None:
            ann.update(agg_meta)
        
        annotation_list.append(ann)
    
    # Create pseudobulk expression matrix.
    pb_exprs_arr = np.vstack(pb_expr_list)
    if gene_ids is not None:
        pb_exprs = pd.DataFrame(pb_exprs_arr, index=pb_ids, columns=gene_ids)
    else:
        pb_exprs = pb_exprs_arr
    
    annotation_df = pd.DataFrame(annotation_list)
    annotation_df[annotation_df.isna()]=0.0
    annotation_df.index = annotation_df["pb_id"].tolist()
    sorted_pb_ids = sorted(annotation_df.index.tolist())
    pb_exprs = pb_exprs.loc[sorted_pb_ids,:]
    annotation_df = annotation_df.loc[sorted_pb_ids,:]
    return pb_exprs, annotation_df



def filter_edges_within_clusters(adj: coo_matrix, clusters: list) -> coo_matrix:
    """
    Filters out the edges in a COO-format sparse matrix (weighted adjacency matrix)
    that connect nodes from different clusters. Returns a new COO matrix retaining
    only the edges within the same cluster.

    Parameters:
        adj (coo_matrix): The input sparse adjacency matrix.
        clusters (list): A list of cluster labels corresponding to each node in the matrix.
    
    Returns:
        coo_matrix: A new sparse COO matrix with edges only between nodes of the same cluster.
    """
    # Convert clusters to a numpy array for vectorized indexing
    clusters_arr = np.array(clusters)
    # Extract row and column indices along with the data from the COO matrix
    rows = adj.row
    cols = adj.col
    data = adj.data
    # Create a boolean mask where True indicates nodes from the same cluster
    same_cluster_mask = clusters_arr[rows] == clusters_arr[cols]
    # Use the mask to filter the rows, columns, and data
    new_rows = rows[same_cluster_mask]
    new_cols = cols[same_cluster_mask]
    new_data = data[same_cluster_mask]    
    # Return a new COO matrix containing only the intra-cluster edges
    return coo_matrix((new_data, (new_rows, new_cols)), shape=adj.shape)


