import torch
import scanpy as sc
import numpy as np
from scgpt.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.utils import load_pretrained
from pathlib import Path
import os
import sys
from scipy import sparse

def normalize_data(adata):
    """
    Normalize the data and add it to adata.layers['counts'].
    
    Args:
        adata: AnnData object containing raw counts in X
        
    Returns:
        adata: AnnData object with normalized counts in layers['counts']
    """
    # Make a copy of the raw counts
    adata.layers["raw_counts"] = adata.X.copy()
    
    # Normalize the data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Store normalized data in layers['counts']
    adata.layers["counts"] = adata.X.copy()
    
    # Restore raw counts to X
    adata.X = adata.layers["raw_counts"]
    
    return adata

def validate_adata(adata):
    """
    Validate that the AnnData object meets the requirements for scGPT.
    
    Args:
        adata: AnnData object to validate
        
    Returns:
        bool: True if validation passes, raises ValueError otherwise
    """
    # Check if adata is an AnnData object
    if not isinstance(adata, sc.AnnData):
        raise ValueError("Input must be an AnnData object")
    
    # Check for raw counts in X
    if sparse.issparse(adata.X):
        # For sparse matrices, check if there are any negative values
        if (adata.X.data < 0).any():
            raise ValueError("X layer should contain raw counts (non-negative values)")
    else:
        # For dense matrices, use the original check
        if not np.all(adata.X >= 0):
            raise ValueError("X layer should contain raw counts (non-negative values)")
    
    # Check for normalized counts in layers["counts"]
    if "counts" not in adata.layers:
        print("Normalized counts not found in layers['counts']. Normalizing data...")
        adata = normalize_data(adata)
    
    # Check gene names
    if len(adata.var_names) == 0:
        raise ValueError("Gene names must be set in var_names")
    
    # Check cell IDs
    if len(adata.obs_names) == 0:
        raise ValueError("Cell IDs must be set in obs_names")
    
    # Check for batch information
    if "batch_id" not in adata.obs:
        print("Warning: No batch information found in obs['batch_id']. "
              "This may affect model performance if you have multiple batches.")
    
    # Check data types
    if not isinstance(adata.X, (np.ndarray, sparse.spmatrix)):
        raise ValueError("X layer must be a numpy array or sparse matrix")
    
    if not isinstance(adata.layers["counts"], (np.ndarray, sparse.spmatrix)):
        raise ValueError("layers['counts'] must be a numpy array or sparse matrix")
    
    return True

def save_embeddings(adata, output_dir="embeddings", prefix="scGPT"):
    """
    Save the AnnData object with embeddings to a file.
    
    Args:
        adata: AnnData object containing the embeddings
        output_dir: Directory to save the embeddings
        prefix: Prefix for the output filename
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{prefix}_embeddings_{timestamp}.h5ad"
    
    # Save the AnnData object
    adata.write_h5ad(output_file)
    print(f"Saved embeddings to {output_file}")
    
    # Also save the embeddings matrix separately as a numpy file
    embeddings_file = output_dir / f"{prefix}_embeddings_matrix_{timestamp}.npy"
    np.save(embeddings_file, adata.obsm["X_scGPT"])
    print(f"Saved embeddings matrix to {embeddings_file}")

def setup_device():
    """
    Set up the device for PyTorch, handling both CPU and GPU cases.
    """
    if torch.cuda.is_available():
        try:
            # Try to get GPU information
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Using GPU: {gpu_name}")
            return torch.device("cuda")
        except Exception as e:
            print(f"GPU available but error accessing it: {e}")
            print("Falling back to CPU")
            return torch.device("cpu")
    else:
        print("No GPU available, using CPU")
        return torch.device("cpu")

# Load your data
# adata = sc.read_h5ad("mouse_data/adata_sc_filtered.h5ad")  # Your scRNA-seq data
adata = sc.read_h5ad("adata_st.h5ad")  # Your scRNA-seq data

# Validate the data
validate_adata(adata)

# Set up device with error handling
device = setup_device()

# Create vocabulary dictionary
vocab = {
    "<pad>": 0,
    "<mask>": 1,
    "<cls>": 2,
    "<unk>": 3
}

# Load pretrained model
model = TransformerModel(
    ntoken=20000,  # vocabulary size
    d_model=512,   # embedding dimension
    nhead=8,       # number of attention heads
    d_hid=2048,    # dimension of the feedforward network
    nlayers=6,     # number of transformer layers
    dropout=0.1,   # dropout rate
    vocab=vocab,   # vocabulary dictionary
    pad_token="<pad>",  # padding token
    do_mvc=True,   # enable masked value prediction
    do_dab=False,  # disable domain adaptation since we don't have batch info
    use_batch_labels=False,  # disable batch labels
    domain_spec_batchnorm=False,  # disable domain-specific batch norm
    n_input_bins=51,  # number of input bins
    ecs_threshold=0.8,  # elastic cell similarity threshold
    explicit_zero_prob=True,  # explicit zero probability
    use_fast_transformer=True,  # use fast transformer
    pre_norm=False,  # pre-normalization
)

# Load pretrained weights
try:
    # First load on CPU
    pretrained_weights = torch.load("mouse_data/best_model.pt", map_location="cpu")
    load_pretrained(model, pretrained_weights)
    # Then move model to GPU
    model = model.to(device)
except FileNotFoundError:
    print("Error: Could not find the model file at mouse_data/best_model.pt")
    sys.exit(1)
except RuntimeError as e:
    print(f"Error loading model: {e}")
    print("Trying to run on CPU instead...")
    device = torch.device("cpu")
    model = model.to(device)

# Prepare data
if sparse.issparse(adata.layers["counts"]):
    print("Converting sparse matrix to dense format...")
    counts_data = adata.layers["counts"].toarray()
else:
    counts_data = adata.layers["counts"]

# Get gene IDs
gene_ids = np.array([vocab.get(gene, vocab["<unk>"]) for gene in adata.var_names])

tokenized_data = tokenize_and_pad_batch(
    counts_data,
    gene_ids,
    max_len=1536,
    vocab=vocab,
    pad_token="<pad>",
    pad_value=0,
    append_cls=True,
    include_zero_gene=True,  # Include zero genes in the input
)

# Generate embeddings
model.eval()
with torch.no_grad():
    input_gene_ids = torch.tensor(tokenized_data["genes"]).to(device)
    input_values = torch.tensor(tokenized_data["values"]).to(device)
    src_key_padding_mask = input_gene_ids.eq(vocab["<pad>"])
    
    # Get cell embeddings using encode_batch
    cell_embeddings = model.encode_batch(
        input_gene_ids,
        input_values.float(),
        src_key_padding_mask=src_key_padding_mask,
        batch_size=64,  # Adjust based on your GPU memory
        batch_labels=None,  # No batch labels needed
        time_step=0,
        return_np=True,
    )

# Add embeddings to your AnnData object
adata.obsm["X_scGPT"] = cell_embeddings

# Save the embeddings
save_embeddings(adata, output_dir="mouse_data_new", prefix="mouse_embeddings_st")