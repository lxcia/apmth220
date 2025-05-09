#!/usr/bin/env python3
import os
import sys
import logging
import torch
import numpy as np
import wandb
from pathlib import Path
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.model import TransformerModel
from scgpt.loss import masked_relative_error
from scgpt.utils import process_data_streaming
from scgpt.trainer import train_epoch, evaluate, prepare_data, prepare_dataloader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        self.seed = 42
        self.do_train = True
        self.mask_ratio = 0.4
        self.epochs = 30
        self.n_bins = 51
        self.GEPC = True
        self.ecs_thres = 0.8
        self.dab_weight = 1.0
        self.lr = 1e-4
        self.batch_size = 32
        self.layer_size = 128
        self.nlayers = 4
        self.nhead = 4
        self.dropout = 0.2
        self.schedule_ratio = 0.9
        self.save_eval_interval = 5
        self.log_interval = 100
        self.fast_transformer = True
        self.pre_norm = False
        self.amp = True
        self.explicit_zero_prob = True
        self.use_batch_labels = True
        self.domain_spec_batchnorm = 'dsbn'
        self.input_emb_style = 'continuous'
        self.cell_emb_style = 'cls'
        self.mvc_decoder_style = 'inner product'
        self.pad_token = '<pad>'
        self.pad_value = -2
        self.mask_value = -1
        self.include_zero_gene = True
        self.max_seq_len = 1536
        self.task = 'integration'
        self.chunk_size = 50000

def main():
    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize wandb
    logger.info("Initializing wandb...")
    wandb.init(
        project="scGPT",
        config=vars(config),
        reinit=True
    )
    logger.info(f"wandb initialized with config: {wandb.config}")
    
    # Load and process data
    logger.info("Loading data from partitions...")
    data_dir = Path("data/processed/blood")
    all_partitions = process_data_streaming(data_dir, chunk_size=config.chunk_size)
    
    if not all_partitions:
        raise ValueError("No partitions loaded. Check data directory and file format.")
    
    # Get gene names from first partition (they should be the same across all partitions)
    gene_ids = all_partitions[0].var_names.tolist()
    logger.info(f"Vocabulary size: {len(gene_ids)}")
    
    # Prepare vocabulary and special tokens
    vocab = {gene: i for i, gene in enumerate(gene_ids)}
    vocab[config.pad_token] = len(vocab)
    vocab['<cls>'] = len(vocab)
    vocab['<eoc>'] = len(vocab)
    
    # Initialize model
    logger.info("Initializing model...")
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=config.layer_size,
        nhead=config.nhead,
        d_hid=config.layer_size,
        nlayers=config.nlayers,
        dropout=config.dropout,
        pad_token=vocab[config.pad_token],
        pad_value=config.pad_value,
        do_token_emb=True,
        vocab_size=len(vocab),
        n_input_bins=config.n_bins,
        cell_emb_style=config.cell_emb_style,
        mvc_decoder_style=config.mvc_decoder_style,
        ecs_threshold=config.ecs_thres,
        explicit_zero_prob=config.explicit_zero_prob,
        use_fast_transformer=config.fast_transformer,
        pre_norm=config.pre_norm,
        do_continuous=True,
        n_bins=config.n_bins,
        input_emb_style=config.input_emb_style,
        use_batch_labels=config.use_batch_labels,
        domain_spec_batchnorm=config.domain_spec_batchnorm,
        use_fast_transformer_masked=config.fast_transformer,
        fast_transformer_backend='flash',
        pre_norm=config.pre_norm,
        amp=config.amp,
        device=device
    )
    model = model.to(device)
    
    # Initialize training components
    criterion_gep_gepc = masked_relative_error
    criterion_dab = torch.nn.CrossEntropyLoss()
    criterion_cls = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.schedule_ratio)
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(1, config.epochs + 1):
        # Prepare data for this epoch
        train_data, valid_data = prepare_data(
            all_partitions,
            gene_ids,
            vocab,
            config,
            epoch
        )
        
        # Create data loaders
        train_loader = prepare_dataloader(
            train_data,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4
        )
        valid_loader = prepare_dataloader(
            valid_data,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Train for one epoch
        train_epoch(
            model,
            train_loader,
            vocab,
            criterion_gep_gepc,
            criterion_dab,
            criterion_cls,
            scaler,
            optimizer,
            scheduler,
            device,
            config,
            logger,
            epoch
        )
        
        # Evaluate
        val_loss = evaluate(
            model,
            valid_loader,
            vocab,
            criterion_gep_gepc,
            criterion_dab,
            criterion_cls,
            device,
            config,
            epoch
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'save/best_model.pt')
            logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
        
        # Log metrics
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "valid/loss": val_loss
        })

if __name__ == "__main__":
    config = Config()
    main() 