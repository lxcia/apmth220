# making this file to preprocess the data for scGPT

# preprocess_data.py

import copy
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import List, Tuple, Dict, Union, Optional
import warnings
import logging

import torch
from anndata import AnnData
import scanpy as sc
import numpy as np
import wandb
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
import psutil

sys.path.append("../")
import scgpt as scg
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer.gene_tokenizer import GeneVocab

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

def set_seed(seed):
    """Set random seed for reproducibility"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process()
    memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
    return memory_gb

def read_scb_file(data_path: Path, batch_size: int = 10000) -> AnnData:
    """Read .scb format data and convert to AnnData"""
    # Read counts from parquet file
    counts_file = data_path / "counts.datatable.parquet"
    if not counts_file.exists():
        raise FileNotFoundError(f"Counts file not found: {counts_file}")
    
    # Read gene vocabulary
    vocab_file = data_path / "gene_vocab.json"
    if not vocab_file.exists():
        raise FileNotFoundError(f"Gene vocabulary file not found: {vocab_file}")
        
    with open(vocab_file, 'r') as f:
        gene_vocab = json.load(f)
    
    # Read manifest for metadata
    manifest_file = data_path / "manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_file}")
        
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)
    
    # Read the counts data in batches
    counts_df = pd.read_parquet(counts_file)
    
    # Get dimensions
    n_cells = len(counts_df)
    n_genes = len(gene_vocab)
    logger.info(f"Number of cells: {n_cells}")
    logger.info(f"Number of genes: {n_genes}")
    
    # Process in batches
    sparse_matrices = []
    for start_idx in range(0, n_cells, batch_size):
        end_idx = min(start_idx + batch_size, n_cells)
        logger.info(f"Processing batch {start_idx//batch_size + 1}/{(n_cells-1)//batch_size + 1}")
        
        batch_df = counts_df.iloc[start_idx:end_idx]
        
        # Create lists for sparse matrix construction
        rows = []
        cols = []
        data = []
        
        # Fill the sparse matrix data
        for cell_idx, (genes, exprs) in enumerate(zip(batch_df['genes'], batch_df['expressions'])):
            if len(genes) != len(exprs):
                logger.warning(f"Mismatch in genes and expressions length for cell {start_idx + cell_idx}")
                continue
            rows.extend([cell_idx] * len(genes))
            cols.extend(genes)
            data.extend(exprs)
        
        # Create sparse matrix in COO format for this batch
        batch_matrix = sparse.coo_matrix(
            (data, (rows, cols)),
            shape=(end_idx - start_idx, n_genes)
        ).tocsr()
        
        sparse_matrices.append(batch_matrix)
        
        # Free memory
        del rows, cols, data
        gc.collect()
    
    # Combine all batches
    expressions_matrix = sparse.vstack(sparse_matrices)
    
    # Create AnnData object
    adata = AnnData(X=expressions_matrix)
    
    # Set gene names using the gene vocabulary
    adata.var_names = pd.Index(gene_vocab)
    
    # Set cell names using the id column
    adata.obs_names = counts_df['id'].astype(str)
    
    # Add metadata from manifest if available
    if manifest.get('metadata'):
        for key, value in manifest['metadata'].items():
            adata.obs[key] = value
    
    logger.info(f"Created AnnData with shape: {adata.shape}")
    logger.info(f"Matrix sparsity: {1.0 - expressions_matrix.nnz / (n_cells * n_genes):.2%}")
    return adata

def process_dataset(data_path: str, save_dir: str):
    """Process a single dataset following the authors' approach"""
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = scg.logger
    log_file = save_dir / "preprocessing.log"
    scg.utils.add_file_handler(logger, log_file)
    
    logger.info(f"Starting processing of {data_path}")
    logger.info(f"Saving results to {save_dir}")
    
    # Load data
    logger.info(f"Loading data from: {data_path}")
    try:
        adata = read_scb_file(Path(data_path))
        logger.info(f"Successfully loaded data with shape: {adata.shape}")
        
        # Save raw data before preprocessing
        raw_file = save_dir / "raw_data.h5ad"
        logger.info(f"Saving raw data to {raw_file}")
        adata.write_h5ad(raw_file, compression='gzip')
        
    except Exception as e:
        logger.error(f"Error reading data from {data_path}: {e}")
        raise
    
    # Process the data
    preprocessor = Preprocessor()
    try:
        logger.info("Starting preprocessing steps...")
        processed_dict = preprocessor(adata)
        
        # Check if preprocessing returned None
        if processed_dict is None:
            logger.error("Preprocessor returned None instead of dict")
            raise ValueError("Preprocessor returned None")
        
        logger.info("Preprocessing completed")
        
        # Save the processed data dictionary
        processed_file = save_dir / "processed_data.h5ad"
        logger.info(f"Saving processed AnnData to {processed_file}")
        adata.write_h5ad(processed_file, compression='gzip')
        
        # Save additional processed layers/attributes
        for key, value in processed_dict.items():
            if isinstance(value, np.ndarray):
                np.save(save_dir / f"{key}.npy", value)
            elif isinstance(value, sparse.spmatrix):
                sparse.save_npz(save_dir / f"{key}.npz", value)
            elif isinstance(value, (list, dict)):
                with open(save_dir / f"{key}.json", 'w') as f:
                    json.dump(value, f)
            logger.info(f"Saved {key} to {save_dir}")
            
        # Save metadata
        metadata = {
            'n_cells': adata.n_obs,
            'n_genes': adata.n_vars,
            'sparsity': float(1.0 - (adata.X.nnz / (adata.n_obs * adata.n_vars)) 
                            if issparse(adata.X) else 'dense'),
            'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'processed_keys': list(processed_dict.keys())
        }
        
        meta_file = save_dir / "metadata.json"
        logger.info(f"Saving metadata to {meta_file}")
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info("All files saved successfully")
        
        logger.info(f"Preprocessor returned dictionary with keys: {list(processed_dict.keys())}")
        
    except Exception as e:
        logger.error(f"Error during preprocessing or saving: {e}")
        logger.error(f"Exception details: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
        
    finally:
        logger.info("Processing completed")
    
    return processed_dict

def main(cell_types_to_process=None):
    # Directory containing .scb files
    data_dir = Path("data/cellxgene/batch_download_scb")
    save_dir = Path("data/processed_all")
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
        
    # Process each dataset
    for cell_type_dir in data_dir.iterdir():
        if not cell_type_dir.is_dir():
            continue
            
        # Skip if this cell type is not in our list to process
        if cell_types_to_process and cell_type_dir.name not in cell_types_to_process:
            logger.info(f"Skipping {cell_type_dir.name} - not in processing list")
            continue
            
        logger.info(f"Processing {cell_type_dir.name}")
        cell_type_save_dir = save_dir / cell_type_dir.name
        
        # Keep track of failures
        failures = []
        
        for partition_dir in cell_type_dir.iterdir():
            if not partition_dir.is_dir() or not partition_dir.name.endswith('.scb'):
                continue
                
            try:
                partition_save_dir = cell_type_save_dir / partition_dir.name
                logger.info(f"Processing partition: {partition_dir}")
                
                # Skip if already processed
                if (partition_save_dir / "processed_data.h5ad").exists():
                    logger.info(f"Skipping {partition_dir} - already processed")
                    continue
                    
                process_dataset(
                    partition_dir,
                    partition_save_dir
                )
                logger.info(f"Successfully processed {partition_dir}")
                
            except Exception as e:
                logger.error(f"Error processing {partition_dir}: {e}")
                failures.append((partition_dir, str(e)))
                continue
        
        # Log summary of failures
        if failures:
            logger.error("The following partitions failed to process:")
            for partition, error in failures:
                logger.error(f"  - {partition}: {error}")

if __name__ == "__main__":
    # Settings from original authors
    hyperparameter_defaults = dict(
        seed=42,
        n_hvg=1200,  # number of highly variable genes
        n_bins=51,
    )

    config = hyperparameter_defaults
    
    try:
        set_seed(config['seed'])
    except Exception as e:
        logger.warning(f"Warning: Could not set seed: {e}")
    
    # Special tokens settings
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    mask_value = -1
    pad_value = -2
    
    # Specify which cell types to process
    cell_types_to_process = ["others", "pancreas"]
    # cell_types_to_process = ["blood"]
    main(cell_types_to_process)