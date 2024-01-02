from copy import deepcopy
import scanpy as sc
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Set the seed for reproducibility
np.random.seed(123456)

# The prefix sc_ is used here for scanpy, while ro_ is for robust
sc_adata = sc.read_10x_h5("c:/Users/styler/Documents/data/toy_sc_rna/data/Breast_Cancer_3p_filtered_feature_bc_matrix.h5", 
                       gex_only=False)
ro_adata = sc.read_10x_h5("c:/Users/styler/Documents/data/toy_sc_rna/data/Breast_Cancer_3p_filtered_feature_bc_matrix.h5", 
                       gex_only=False)

#adata.var["symbol"]=adata.var.index
#adata.var.index = adata.var["gene_ids"]

###########################################
# Basic pre-processing
def do_preprocess(local_adata):
    # SCANPY DEVS: STOP USING GLOBAL VARIABLES. IT MAKES WELL WRITTEN CODE BREAK...
    # The "adata" leakage is terrible <you're killing me Small's gif>
    local_adata.var_names_make_unique()
    sc.pp.filter_cells(local_adata, min_genes=200)
    sc.pp.filter_genes(local_adata, min_cells=3)
    local_adata.var['mt'] = local_adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(local_adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pl.violin(local_adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                jitter=0.4, multi_panel=True)
    sc.pl.scatter(local_adata, x='total_counts', y='pct_counts_mt')
    sc.pl.scatter(local_adata, x='total_counts', y='n_genes_by_counts')
    local_adata = local_adata[local_adata.obs.pct_counts_mt < 25, :]
    local_adata.obs= local_adata.obs[local_adata.obs.pct_counts_mt < 25,:]
    return local_adata


# process them
sc_adata = do_preprocess(sc_adata)
sc_adata = sc_adata[sc_adata.obs.pct_counts_mt < 25, :]
sc_adata.obs= sc_adata.obs[sc_adata.obs.pct_counts_mt < 25,:]
# I loathe the duplicated code, but the global variable leakage
# makes you need to do it all separate
ro_adata = do_preprocess(ro_adata)
ro_adata = ro_adata[ro_adata.obs.pct_counts_mt < 25, :]
ro_adata.obs= ro_adata.obs[ro_adata.obs.pct_counts_mt < 25,:]

###########################################
# Let the robust pipeline do it's thing...
ro = robust(ro_adata)
ro.train_feature_df.index=ro_adata.var.index
ro.val_feature_df.index=ro_adata.var.index
# Make a spring embedding from it
G = nx.from_scipy_sparse_array(ro.graph)
pos = nx.spring_layout(G)
pos_array = np.array([v for k, v in pos.items()])

# To use scanpy's plotting function's later, we'll store it
ro_adata.obsm["X_ro_spring"]=pos_array
# And now do leiden on the robust graph
sc.tl.leiden(ro_adata, adjacency=ro.graph.tocsr())


##############
# Run the standard scanpy tutorial code
sc.pp.normalize_total(sc_adata, target_sum=1e4)
sc.pp.log1p(sc_adata)
sc.pp.highly_variable_genes(sc_adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc_adata = sc_adata[:, sc_adata.var.highly_variable]
sc.pp.regress_out(sc_adata, ['total_counts', 'pct_counts_mt'])
sc.pp.scale(sc_adata, max_value=10)
sc.tl.pca(sc_adata, svd_solver='arpack')
sc.pl.pca(sc_adata, color='HER2')
sc.pp.neighbors(sc_adata, n_neighbors=10, n_pcs=40)

# Calculate the umap
sc.tl.draw_graph(sc_adata)
sc.tl.umap(sc_adata)
sc.tl.leiden(sc_adata)

##############################################
# include it in the ro_adata for simplicity
ro_adata.obsm["X_sc_umap"]=sc_adata.obsm["X_umap"]
ro_adata.obsm["X_sc_spring"]=sc_adata.obsm["X_draw_graph_fr"]
ro_adata.obs["sc_leiden"]=sc_adata.obs["leiden"]
sc.tl.rank_genes_groups(ro_adata, 'leiden', method='logreg')
sc.pl.rank_genes_groups_heatmap(ro_adata, n_genes=3)
##

# plot
goi=["EPCAM", "ERBB2", "XBP1", "HIF1A", "MYC", "EGFR", "VWF", "FCER1G", "LYZ"]
#sc.pl.scatter(ro_adata, basis="ro_spring", color=["leiden"]+goi)


#sc.pl.scatter(ro_adata, basis="sc_umap", color=["leiden"], show=False)
#sc.pl.scatter(ro_adata, basis="sc_spring", color=["leiden"], show=False)
#sc.pl.scatter(ro_adata, basis="ro_spring", color=["sc_leiden"], show=False)

sc.pl.scatter(ro_adata, basis="sc_umap", color=["sc_leiden"]+goi, show=False)
#sc.pl.scatter(ro_adata, basis="sc_spring", color=["sc_leiden"]+goi, show=False)
sc.pl.scatter(ro_adata, basis="ro_spring", color=["leiden"]+goi)




