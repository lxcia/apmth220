import scanpy as sc
import squidpy as sq
import numpy as np
import pandas as pd
from anndata import AnnData
import pathlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import skimage
import seaborn as sns
import nottangram as tg

sc.logging.print_header()
print(f"squidpy=={sq.__version__}")

# LOAD ACTUAL DATA

# adata_st = sq.datasets.visium_fluo_adata_crop()
# adata_st = adata_st[
#     adata_st.obs.cluster.isin([f"Cortex_{i}" for i in np.arange(1, 5)])
# ].copy()
# # img = sq.datasets.visium_fluo_image_crop()

# # adata_sc = sq.datasets.sc_mouse_cortex()

# # save fresh copy of adata_st
# # adata_st.write_h5ad("adata_st.h5ad")

# # LOAD ST EMBEDDINGS

# adata_st_embeddings = sc.read_h5ad("/n/data1/hms/dbmi/zitnik/lab/users/lucia1215/tangram/mouse_embeddings_st_embeddings_20250421_131135.h5ad")
# print(f"Number of rows in embeddings matrix: {adata_st_embeddings.shape[0]}")

# # sc.tl.rank_genes_groups(adata_sc, groupby="cell_subclass", use_raw=False)
# # markers_df = pd.DataFrame(adata_sc.uns["rank_genes_groups"]["names"]).iloc[0:100, :]
# # markers = list(np.unique(markers_df.melt().value.values))
# # len(markers)

# # # Create the augmented matrices
# # # For ST data
# # H_st = adata_st_embeddings.X.toarray() if not isinstance(adata_st_embeddings.X, np.ndarray) else adata_st_embeddings.X
# # H_st_emb = adata_st_embeddings.obsm["X_scGPT"]
# # H_st_augmented = np.hstack([H_st, H_st_emb])

# # Get the original data and embeddings
# H_st = adata_st.X.toarray() if not isinstance(adata_st.X, np.ndarray) else adata_st.X
# H_st_emb = adata_st_embeddings.obsm["X_scGPT"]

# # Print the shapes to verify
# print(f"Original data shape: {H_st.shape}")
# print(f"Embeddings shape: {H_st_emb.shape}")

# # Create the augmented matrix
# H_st_augmented = np.hstack([H_st, H_st_emb])
# print(f"Augmented matrix shape: {H_st_augmented.shape}")

# # Create a new AnnData with the correct dimensions
# adata_st_augmented = sc.AnnData(X=H_st_augmented)

# # Create new var names with the CORRECT number of embedding dimensions
# n_genes = H_st.shape[1]
# n_emb_dims = H_st_emb.shape[1]
# print(f"Number of genes: {n_genes}")
# print(f"Number of embedding dimensions: {n_emb_dims}")

# # Create variable names that match the number of columns
# var_names = list(adata_st.var_names) + [f"emb_dim_{i}" for i in range(n_emb_dims)]
# print(f"Total number of variable names: {len(var_names)}")

# # Verify that the length matches
# if len(var_names) != H_st_augmented.shape[1]:
#     print("ERROR: Variable names length doesn't match number of columns!")
#     print(f"Variable names length: {len(var_names)}")
#     print(f"Columns in augmented data: {H_st_augmented.shape[1]}")
# else:
#     # Only assign if lengths match
#     adata_st_augmented.var_names = pd.Index(var_names)
#     print("Variable names assigned successfully")

# # Save the augmented AnnData object
# adata_st_augmented.write_h5ad("adata_st_augmented.h5ad")

# print("here")


# SINGLE CELL DATA HERE 

adata_sc = sq.datasets.sc_mouse_cortex()

# LOAD SC EMBEDDINGS

adata_sc_embeddings = sc.read_h5ad("/n/data1/hms/dbmi/zitnik/lab/users/lucia1215/tangram/mouse_embeddings_sc_embeddings_20250421_122340.h5ad")
print(f"Number of rows in embeddings matrix: {adata_sc_embeddings.shape[0]}")

# Get the original data and embeddings
H_sc = adata_sc.X.toarray() if not isinstance(adata_sc.X, np.ndarray) else adata_sc.X
H_sc_emb = adata_sc_embeddings.obsm["X_scGPT"]

# Print the shapes to verify
print(f"Original scdata shape: {H_sc.shape}")
print(f"scEmbeddings shape: {H_sc_emb.shape}")

# Create the augmented matrix
H_sc_augmented = np.hstack([H_sc, H_sc_emb])
print(f"Augmented matrix shape: {H_sc_augmented.shape}")

# Create a new AnnData with the correct dimensions
adata_sc_augmented = sc.AnnData(X=H_sc_augmented)

# Create new var names with the CORRECT number of embedding dimensions
n_genes = H_sc.shape[1]
n_emb_dims = H_sc_emb.shape[1]
print(f"Number of genes: {n_genes}")
print(f"Number of embedding dimensions: {n_emb_dims}")

# Create variable names that match the number of columns
var_names = list(adata_sc.var_names) + [f"emb_dim_{i}" for i in range(n_emb_dims)]
print(f"Total number of variable names: {len(var_names)}")

# Verify that the length matches
if len(var_names) != H_sc_augmented.shape[1]:
    print("ERROR: Variable names length doesn't match number of columns!")
    print(f"Variable names length: {len(var_names)}")
    print(f"Columns in augmented data: {H_sc_augmented.shape[1]}")
else:
    # Only assign if lengths match
    adata_sc_augmented.var_names = pd.Index(var_names)
    print("Variable names assigned successfully")

# Save the augmented AnnData object
adata_sc_augmented.write_h5ad("adata_sc_augmented.h5ad")
print("saved adata_sc_augmented.h5ad")

print("here")





