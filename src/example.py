import scanpy as sc


adata = sc.read_10x_h5("c:/Users/styler/Documents/data/toy_sc_rna/data/Breast_Cancer_3p_filtered_feature_bc_matrix.h5", 
                       gex_only=False)

#adata.var["symbol"]=adata.var.index
#adata.var.index = adata.var["gene_ids"]


ro = robust(adata)


