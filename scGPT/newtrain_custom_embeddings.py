import wandb
import torch
import numpy as np
import pandas as pd
import logging
import sys
import os
from pathlib import Path
import json
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
import warnings
import time
import copy
import gc
from typing import List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('train_custom_embeddings.log')
    ]
)
logger = logging.getLogger(__name__)

# Check CUDA availability
logger.info(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    logger.info(f'CUDA device: {torch.cuda.get_device_name(0)}')
    logger.info(f'CUDA version: {torch.version.cuda}')
    device = torch.device('cuda')
else:
    logger.error('CUDA is not available. Please check your GPU setup.')
    raise RuntimeError('CUDA is not available')

try:
    from scgpt import prepare_data, prepare_dataloader, train, evaluate, define_wandb_metrcis
    from scgpt.model.model_custom_embeddings import TransformerModel
    from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
    from scgpt.loss import masked_mse_loss, criterion_neg_log_bernoulli
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import scanpy as sc
    import anndata as ad
except ImportError as e:
    logger.error(f'Failed to import required modules: {e}')
    raise

# Initialize wandb
logger.info('Initializing wandb...')
hyperparameter_defaults = dict(
    seed=42,
    do_train=True,
    # mask_ratio=0.4,
    epochs=6,
    n_bins=51,
    GEPC=True,
    ecs_thres=0.6,  # ECS threshold for expression consistency - not specified in paper, using default value
    dab_weight=1.0,
    lr=1e-4,
    batch_size=4,  # Reduced from 32 to handle memory constraints
    layer_size=512,
    nlayers=12,
    nhead=8,
    dropout=0.2,
    schedule_ratio=0.9,
    save_eval_interval=5,
    log_interval=100,
    fast_transformer=True,
    pre_norm=False,
    amp=True,
    explicit_zero_prob=True,
    use_batch_labels=True,
    domain_spec_batchnorm='dsbn',
    input_emb_style='continuous',
    cell_emb_style='cls',
    mvc_decoder_style='inner product',
    pad_token='<pad>',
    pad_value=-2,
    mask_value=-1,
    include_zero_gene=False,
    max_seq_len=1200,
    task='integration',
    chunk_size=5000,  # Reduced from 25000 to prevent OOM errors
    CLS=False,
    ESC=False,
    DAR=True,
    GEP=True,
    use_mod=False,
    DSBN=True,
    input_layer_key='counts'
)

try:
    run = wandb.init(
        config=hyperparameter_defaults,
        project='scGPT-custom-embeddings',
        reinit=True,
        settings=wandb.Settings(start_method='fork', allow_val_change=True)
    )
    config = wandb.config
    # Ensure use_mod is False since we're not using multi-omics data
    config.use_mod = False
    logger.info(f'wandb initialized with config: {config}')
except Exception as e:
    logger.error(f'Failed to initialize wandb: {e}')
    raise

class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def load_custom_embeddings(embeddings_path: str, gene_ids: List[str], target_dim: int = 128) -> torch.Tensor:
    """
    Load custom embeddings from .npz file and ensure they match the vocabulary size.
    
    Args:
        embeddings_path: Path to the .npz file containing embeddings
        gene_ids: List of gene IDs in the vocabulary
        target_dim: Target dimension for the embeddings
        
    Returns:
        torch.Tensor: Embedding matrix of shape (vocab_size, target_dim)
    """
    try:
        # Load the embeddings
        data = np.load(embeddings_path)
        embeddings = data['embeddings']
        gene_names = data['gene_names']
        
        logger.info(f'Loaded embeddings with shape: {embeddings.shape}')
        logger.info(f'Target dimension: {target_dim}')
        
        # Handle dimension mismatch
        if embeddings.shape[1] != target_dim:
            logger.info(f'Resizing embeddings from {embeddings.shape[1]} to {target_dim} dimensions')
            # Use linear interpolation to resize embeddings
            from scipy.interpolate import interp1d
            resized_embeddings = np.zeros((embeddings.shape[0], target_dim))
            for i in range(embeddings.shape[0]):
                f = interp1d(np.linspace(0, 1, embeddings.shape[1]), 
                            embeddings[i], 
                            kind='linear',
                            fill_value='extrapolate')
                resized_embeddings[i] = f(np.linspace(0, 1, target_dim))
            embeddings = resized_embeddings
            logger.info(f'Resized embeddings shape: {embeddings.shape}')
        
        # Create mapping from gene names to indices
        gene_to_idx = {gene: i for i, gene in enumerate(gene_names)}
        
        # Initialize full embedding matrix with zeros
        vocab_size = len(gene_ids) + 3  # Add 3 for special tokens
        full_embeddings = torch.zeros((vocab_size, target_dim))
        
        # Fill in custom embeddings where available
        found_genes = 0
        for i, gene in enumerate(gene_ids):
            if gene in gene_to_idx:
                full_embeddings[i] = torch.from_numpy(embeddings[gene_to_idx[gene]])
                found_genes += 1
        
        logger.info(f'Found embeddings for {found_genes} out of {len(gene_ids)} genes')
        
        # Initialize special token embeddings with small random values
        for i in range(len(gene_ids), vocab_size):
            full_embeddings[i] = torch.randn(target_dim) * 0.02
        
        # Try to move to GPU with error handling
        try:
            torch.cuda.empty_cache()
            full_embeddings = full_embeddings.to('cuda')
            logger.info('Successfully moved embeddings to GPU')
        except Exception as e:
            logger.warning(f'Failed to move embeddings to GPU: {str(e)}')
            logger.warning('Continuing with CPU tensors')
        
        logger.info(f'Final embedding matrix shape: {full_embeddings.shape}')
        return full_embeddings
        
    except Exception as e:
        logger.error(f'Error loading custom embeddings: {str(e)}')
        raise

def tokenize_in_chunks(counts, gene_ids, chunk_size=10000):
    """Tokenize data in smaller chunks with memory management"""
    tokenized_data = []
    for i in range(0, len(counts), chunk_size):
        chunk = counts[i:i+chunk_size]
        logger.info(f'Tokenizing chunk {i//chunk_size + 1} of {len(counts)//chunk_size + 1}')
        
        # Convert to dense if sparse and clear memory
        if issparse(chunk):
            chunk = chunk.toarray()
        
        # Convert gene names to indices using the vocab
        gene_indices = np.array([vocab[gene] for gene in gene_ids])
        
        # Tokenize the chunk
        tokenized_chunk = tokenize_and_pad_batch(
            chunk,
            gene_indices,
            max_len=config.max_seq_len,
            vocab=vocab,
            pad_token=config.pad_token,
            pad_value=config.pad_value,
            append_cls=True,
            include_zero_gene=config.include_zero_gene
        )
        
        # Clear memory
        del chunk
        del gene_indices
        torch.cuda.empty_cache()
        gc.collect()
        
        # Process tokenized chunk
        for key in tokenized_chunk:
            if isinstance(tokenized_chunk[key], torch.Tensor):
                tokenized_chunk[key] = tokenized_chunk[key].cpu()  # Move to CPU
        
        tokenized_data.append(tokenized_chunk)
        
        # Clear memory again
        torch.cuda.empty_cache()
        gc.collect()
    
    return tokenized_data

def process_data_streaming(data_dir, chunk_size=10000):
    """Process data in a streaming fashion, yielding one partition at a time"""
    logger.info('Loading data from partitions...')
    
    # Create a mapping from original partition IDs to sequential integers
    partition_id_map = {}
    current_id = 0
    
    try:
        # Process all partitions
        for partition_dir in sorted(data_dir.glob('partition_*.scb')):
            logger.info(f'Processing partition: {partition_dir.name}')
            
            # Get original partition ID
            original_partition_id = int(partition_dir.name.split('_')[1].split('.')[0])
            
            # Map to sequential ID if not already mapped
            if original_partition_id not in partition_id_map:
                partition_id_map[original_partition_id] = current_id
                current_id += 1
            
            # Read metadata
            with open(partition_dir / 'metadata.json', 'r') as f:
                metadata = json.load(f)
            
            # Read gene names
            with open(partition_dir / 'var_names.json', 'r') as f:
                gene_names = json.load(f)
            
            # Read cell names
            with open(partition_dir / 'obs_names.json', 'r') as f:
                cell_names = json.load(f)
            
            # Read normalized counts in chunks
            import scipy.sparse as sp
            counts = sp.load_npz(partition_dir / 'normalized_counts.npz')
            
            # Process in smaller chunks
            for i in range(0, counts.shape[0], chunk_size):
                chunk = counts[i:i+chunk_size]
                if issparse(chunk):
                    chunk = chunk.toarray()
                
                partition_adata = ad.AnnData(
                    X=chunk,
                    obs=pd.DataFrame(index=cell_names[i:i+chunk_size]),
                    var=pd.DataFrame(index=gene_names)
                )
                
                # Use mapped partition ID for batch labels
                mapped_partition_id = partition_id_map[original_partition_id]
                partition_adata.obs['batch_id'] = mapped_partition_id
                partition_adata.obs['partition_id'] = original_partition_id
                
                # Add modality type if use_mod is True
                if config.use_mod:
                    partition_adata.obs['mod_type'] = 0  # Default modality type
                
                # Yield the processed chunk immediately
                yield partition_adata, mapped_partition_id, original_partition_id
                
                logger.info(f'Processed chunk {i//chunk_size + 1} of partition {original_partition_id} (mapped to {mapped_partition_id}) with {len(partition_adata)} cells')
                
                # Clear memory
                del chunk
                del partition_adata
                torch.cuda.empty_cache()
                gc.collect()
    
    except Exception as e:
        logger.error(f'Failed to load data: {e}')
        raise
    
    logger.info(f'Partition ID mapping: {partition_id_map}')

def train_in_chunks(model, data_dirs, config, device):
    """Train model on data processed in chunks with memory management"""
    best_val_loss = float('inf')
    best_model = None
    
    # Create save directory if it doesn't exist
    save_dir = Path('save')
    save_dir.mkdir(exist_ok=True)
    
    # Define possible mask ratios for pretraining
    mask_ratios = [0.25, 0.50, 0.75]
    
    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()
        logger.info(f'Starting epoch {epoch}/{config.epochs}')
        
        # Process each data directory
        for data_dir in data_dirs:
            logger.info(f'Processing data directory: {data_dir}')
            
            # Process each chunk from the data directory
            for chunk_idx, (partition_adata, mapped_partition_id, original_partition_id) in enumerate(
                process_data_streaming(data_dir, chunk_size=config.chunk_size)
            ):
                logger.info(f'Processing chunk {chunk_idx + 1}')
                
                # Clear memory before processing new chunk
                torch.cuda.empty_cache()
                gc.collect()
                
                # Prepare data for this chunk
                all_counts = partition_adata.X.toarray() if issparse(partition_adata.X) else partition_adata.X
                chunk_batch_labels = np.array([mapped_partition_id] * len(partition_adata))
                
                # Split into train and validation with 99.7/0.3 ratio
                indices = np.arange(len(partition_adata))
                train_idx, valid_idx = train_test_split(
                    indices, 
                    test_size=0.003,
                    random_state=config.seed
                )
                
                # Convert indices to numpy arrays
                train_idx = np.array(train_idx)
                valid_idx = np.array(valid_idx)
                
                # Randomly select mask ratio for this chunk
                current_mask_ratio = np.random.choice(mask_ratios)
                logger.info(f'Using mask ratio: {current_mask_ratio}')
                
                # Tokenize this chunk
                tokenized_train = tokenize_in_chunks(
                    all_counts[train_idx], 
                    gene_ids, 
                    chunk_size=config.chunk_size
                )
                tokenized_valid = tokenize_in_chunks(
                    all_counts[valid_idx], 
                    gene_ids, 
                    chunk_size=config.chunk_size
                )
                
                # Clear memory after tokenization
                del all_counts
                torch.cuda.empty_cache()
                gc.collect()
                
                # Train on this chunk
                for train_chunk, valid_chunk in zip(tokenized_train, tokenized_valid):
                    # Prepare data with current mask ratio
                    train_data_pt, valid_data_pt = prepare_data(
                        train_chunk,
                        valid_chunk,
                        chunk_batch_labels[train_idx],
                        chunk_batch_labels[valid_idx],
                        config,
                        epoch,
                        mask_ratio=current_mask_ratio
                    )
                    
                    # Save model-specific keys while keeping original keys
                    train_data_pt.update({
                        "src": train_data_pt["gene_ids"],
                        "values": train_data_pt["values"],
                        "src_key_padding_mask": train_data_pt["gene_ids"].eq(vocab[config.pad_token])
                    })
                    
                    valid_data_pt.update({
                        "src": valid_data_pt["gene_ids"],
                        "values": valid_data_pt["values"],
                        "src_key_padding_mask": valid_data_pt["gene_ids"].eq(vocab[config.pad_token])
                    })
                    
                    train_loader = prepare_dataloader(
                        train_data_pt,
                        batch_size=config.batch_size,
                        shuffle=False,
                        intra_domain_shuffle=True,
                        drop_last=False
                    )
                    
                    valid_loader = prepare_dataloader(
                        valid_data_pt,
                        batch_size=config.batch_size,
                        shuffle=False,
                        intra_domain_shuffle=False,
                        drop_last=False
                    )
                    
                    if config.do_train:
                        train(
                            model,
                            train_loader,
                            vocab,
                            criterion,
                            criterion_dab,
                            criterion_cls,
                            scaler,
                            optimizer,
                            scheduler,
                            device,
                            config,
                            logger,
                            epoch,
                            mask_ratio=current_mask_ratio
                        )
                    
                    val_loss = evaluate(
                        model,
                        valid_loader,
                        vocab,
                        criterion,
                        criterion_dab,
                        criterion_cls,
                        device,
                        config,
                        epoch,
                        use_mod=config.use_mod,
                        mask_ratio=current_mask_ratio
                    )
                    
                    # Clear memory after training and evaluation
                    del train_data_pt
                    del valid_data_pt
                    del train_loader
                    del valid_loader
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Clear memory after processing chunk
                del tokenized_train
                del tokenized_valid
                torch.cuda.empty_cache()
                gc.collect()
        
        elapsed = time.time() - epoch_start_time
        logger.info('-' * 89)
        logger.info(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | valid loss {val_loss:5.4f}')
        logger.info('-' * 89)
        
        # Save model after each epoch
        model_path = save_dir / f'model_custom_embeddings_epoch{epoch}.pt'
        torch.save(model.state_dict(), model_path)
        logger.info(f'Saved model for epoch {epoch} to {model_path}')
        
        # Save best model if this is the best validation loss so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            logger.info(f'Best model with score {best_val_loss:5.4f}')
            
            # Save best model
            best_model_path = save_dir / f'model_custom_embeddings_best.pt'
            torch.save(best_model.state_dict(), best_model_path)
            logger.info(f'Saved best model to {model_path}')
        
        scheduler.step()
        logger.info(f'Learning rate: {scheduler.get_last_lr()[0]:.6f}')
        
        # Clear memory after epoch
        torch.cuda.empty_cache()
        gc.collect()

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    vocab,
    criterion_gep_gepc,
    criterion_dab,
    criterion_cls,
    device,
    config,
    epoch,
    use_mod=False,
    mask_ratio=None
) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    import wandb

    model.eval()
    total_loss = 0.0
    # total_error = 0.0
    total_dab = 0.0
    total_num = 0
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            if config.task == "annotation":
                celltype_labels = batch_data["celltype_labels"].to(device)
            if config.task == "multiomic" and use_mod:
                mod_types = batch_data["mod_types"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[config.pad_token])

            with torch.cuda.amp.autocast(enabled=config.amp):
                # Prepare model arguments
                model_args = {
                    "src": input_gene_ids,
                    "values": input_values,
                    "src_key_padding_mask": src_key_padding_mask,
                    "batch_labels": batch_labels if config.use_batch_labels or config.DSBN else None,
                    "CLS": config.CLS,
                    "MVC": config.GEPC,
                    "ECS": config.ESC,
                }
                
                # Only add mod_types if use_mod is True
                if use_mod:
                    model_args["mod_types"] = mod_types
                    
                output_dict = model(**model_args)

                if config.task == "annotation":
                    output_values = output_dict["cls_output"]
                    loss = criterion_cls(output_values, celltype_labels)

                elif config.task in ["integration", "multiomic"]:
                    output_values = output_dict["mlm_output"]
                    masked_positions = input_values.eq(config.mask_value)
                    loss = criterion_gep_gepc(
                        output_values, target_values, masked_positions
                    )

                if config.DAR:
                    loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

            total_loss += loss.item() * len(input_gene_ids)

            if config.DAR:
                total_dab += (
                    loss_dab.item() * len(input_gene_ids) if config.DAR else 0.0
                )
            else:
                total_dab = 0

            total_num += len(input_gene_ids)

    wandb.log(
        {
            "valid/loss": (total_loss + config.dab_weight * total_dab) / total_num,
            "epoch": epoch,
        },
    )

    return total_loss / total_num

# Load and process data
# data_folders = ['blood', 'brain', 'heart', 'intestine', 'kidney', 'lung', 'others', 'pancreas']
data_folders = ['pancreas']
data_dirs = [Path(f'data/processed_all/{folder}') for folder in data_folders]

# Get gene names from the first partition of the first directory
first_partition = next(process_data_streaming(data_dirs[0], chunk_size=config.chunk_size))[0]
gene_ids = first_partition.var_names.tolist()

# Prepare vocabulary and special tokens
logger.info('Preparing vocabulary...')
vocab = {gene: i for i, gene in enumerate(gene_ids)}
vocab[config.pad_token] = len(vocab)
vocab['<cls>'] = len(vocab)
vocab['<eoc>'] = len(vocab)
vocab_size = len(vocab)  # Store the final vocabulary size after adding special tokens
logger.info(f'Vocabulary size: {vocab_size}')

# Load custom embeddings
logger.info('Loading custom embeddings...')
custom_embeddings = load_custom_embeddings('embeddings_mouse.npz', gene_ids, target_dim=config.layer_size)
logger.info(f'Loaded custom embeddings with shape: {custom_embeddings.shape}')

# Initialize model with custom embeddings
logger.info('Initializing model with custom embeddings...')
model = TransformerModel(
    ntoken=vocab_size,  # Use the final vocabulary size
    d_model=config.layer_size,
    nhead=config.nhead,
    d_hid=config.layer_size,
    nlayers=config.nlayers,
    vocab=vocab,
    dropout=config.dropout,
    pad_token=config.pad_token,
    pad_value=config.pad_value,
    do_mvc=config.GEPC,
    do_dab=True,
    use_batch_labels=config.use_batch_labels,
    num_batch_labels=len(data_folders),  # One batch label per tissue type
    domain_spec_batchnorm=config.domain_spec_batchnorm,
    n_input_bins=config.n_bins,
    ecs_threshold=config.ecs_thres,
    explicit_zero_prob=config.explicit_zero_prob,
    use_fast_transformer=config.fast_transformer,
    pre_norm=config.pre_norm,
    custom_embeddings=custom_embeddings
)
model.to(device)
logger.info(f'Model initialized with {sum(p.numel() for p in model.parameters())} parameters')

# Initialize training components
logger.info('Initializing training components...')
criterion = masked_mse_loss
criterion_dab = nn.CrossEntropyLoss()
criterion_cls = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=config.lr, eps=1e-4 if config.amp else 1e-8
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.schedule_ratio)
scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

# Start training
logger.info('Starting training with custom embeddings...')
train_in_chunks(model, data_dirs, config, device) 