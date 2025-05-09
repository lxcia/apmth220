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



# load the augmented data files

adata_sc_augmented = sc.read_h5ad("adata_sc_augmented.h5ad")
adata_st_augmented = sc.read_h5ad("adata_st_augmented.h5ad")

# get the gene markers

sc.tl.rank_genes_groups(adata_sc_augmented, groupby="cell_subclass", use_raw=False)
markers_df = pd.DataFrame(adata_sc_augmented.uns["rank_genes_groups"]["names"]).iloc[0:100, :]
markers = list(np.unique(markers_df.melt().value.values))

tg.pp_adatas(adata_sc_augmented, adata_st_augmented, genes=markers)

# doing the actual mapping

ad_map_augmented = tg.map_cells_to_space(adata_sc_augmented, adata_st_augmented,
    mode="cells",
#     mode="clusters",
#     cluster_label='cell_subclass',  # .obs field w cell types
    density_prior='rna_count_based',
    num_epochs=500,
    # device="cuda:0",
    device='cpu',
)