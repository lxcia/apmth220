import scanpy as sc

adata = sc.read_h5ad("/n/data1/hms/dbmi/zitnik/lab/users/kel331/virtualcellsim/PINNACLE/data/preprocessed/downsampled2_adata_normalized_log.h5ad")

import pickle
with open('/n/data1/hms/dbmi/zitnik/lab/users/kel331/virtualcellsim/PINNACLE/data/networks/uns_dictionary.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

adata.uns = loaded_dict
adata.var = adata.var.reset_index().set_index("names")
adata.write_h5ad("/n/data1/hms/dbmi/zitnik/lab/users/kel331/virtualcellsim/PINNACLE/data/networks/ranked_CELLxGENE.h5ad")