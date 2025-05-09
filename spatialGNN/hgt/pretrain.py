'''
PRETRAIN NODE EMBEDDING MODEL
This script contains the main function for pretraining the node embedding model.
'''

# standard imports
import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle

# import PyTorch and DGL
import torch
import dgl

# import PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Timer

# path manipulation
from pathlib import Path

# import project configuration file
import sys
sys.path.append('../..')
import project_config

# custom imports
from hyperparameters import parse_args, get_hyperparameters
from dataloaders import load_graph, partition_graph, create_dataloaders
from models import HGT
import wandb
from utils import check_data_leakage

# Add this constant at the top of the file after imports
# lum's path
GRAPH_PATH_TEMPLATE = '/n/data1/hms/dbmi/zitnik/lab/users/lucia1215/spatialGNN/data/FULL_HETERO_GRAPH_BIDIRECTED_{sparsity}sparsity.bin'
# kel's path
# GRAPH_PATH_TEMPLATE = '/n/home08/kel331/spatialGNN/data/FULL_HETERO_GRAPH_BIDIRECTED_{sparsity}sparsity.bin'
# GRAPH_PATH_TEMPLATE = '/n/home08/kel331/spatialGNN/data/FULL_HETERO_GRAPH_{sparsity}sparsity.bin'
GRAPH_PATH = None

# check if CUDA is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("DEBUG device:", device)


# PRE-TRAINING FUNCTION
def pretrain(hparams):
    try:
        # Update graph path with sparsity
        global GRAPH_PATH
        GRAPH_PATH = GRAPH_PATH_TEMPLATE.format(sparsity=hparams['sparsity'])
        
        # Get the current file's directory and construct path to data folder
        current_dir = Path(__file__).parent
        data_dir = current_dir / 'data'
        
        # Add data_dir to hparams
        hparams['data_dir'] = str(data_dir)
        
        # Print the data directory path for debugging
        print(f"Using data directory: {hparams['data_dir']}")
        
        # Verify that the required file exists
        # lum's path
        mapping_file = Path("/n/data1/hms/dbmi/zitnik/lab/users/lucia1215/spatialGNN/data")
        # kel's path
        # mapping_file = Path("/n/home08/kel331/spatialGNN/data")
        if not mapping_file.exists():
            raise FileNotFoundError(f"Required mapping file not found: {mapping_file}")
        
        # set seed
        pl.seed_everything(hparams['seed'], workers = True)

        # Load the graph
        neuroKG = dgl.load_graphs(GRAPH_PATH)[0][0]
        
        # Print graph information
        print("Graph edge types:", neuroKG.canonical_etypes)
        print("Number of edges per type:", {etype: neuroKG.number_of_edges(etype) 
                                          for etype in neuroKG.canonical_etypes})
        
        # Continue with partitioning
        train_neuroKG, val_neuroKG, test_neuroKG = partition_graph(neuroKG, hparams)

        # get dataloaders
        train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
            neuroKG, train_neuroKG, val_neuroKG, test_neuroKG,
            sampler_fanout = hparams['sampler_fanout'], fixed_k = hparams['fixed_k'], negative_k = hparams['negative_k'],
            train_batch_size = hparams['train_batch_size'], val_batch_size = hparams['val_batch_size'],
            test_batch_size = hparams['test_batch_size'], num_workers = hparams['num_workers']
        )

        # enable CPU affinity
        train_dataloader.enable_cpu_affinity()
        val_dataloader.enable_cpu_affinity()
        test_dataloader.enable_cpu_affinity()

        # instantiate logger

        print("Step 1: Initializing wandb...")
        wandb.init(
            entity="luciam-harvard-medical-school",
            project="spatialgnn-pretraining",
            config=hparams,
            settings=wandb.Settings(start_method="fork"),
            name=hparams["run_name"]
        )
        print("Wandb initialized successfully")

        print("Step 2: Creating WandbLogger...")
        wandb_logger = WandbLogger(
            name='test',
            project='spatialgnn-pretraining',
            entity='luciam-harvard-medical-school',
            save_dir=hparams['wandb_save_dir'],
            id='0',
            resume="allow"
        )
        print("WandbLogger created successfully")

        print("Step 3: Initializing model...")
        model = HGT(
            num_nodes=neuroKG.num_nodes(),
            num_ntypes=len(neuroKG.ntypes),
            num_etypes=len(neuroKG.canonical_etypes),
            hparams=hparams
        )
        model.data_dir = hparams['data_dir']
        model.graph = neuroKG
        
        print("Model initialized successfully")
        print(f"Model structure: {model}")

        print("Step 4: Setting up callbacks...")
        checkpoint_callback = ModelCheckpoint(
            monitor='val/auroc',
            dirpath=Path(hparams['save_dir']) / 'checkpoints',
            filename=f"{hparams['run_id']}_{{epoch}}-{{step}}",
            save_top_k=1,
            mode='max'
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        timer = Timer(duration="02:00:00:00")
        print("Callbacks created successfully")

        print("Step 5: Initializing trainer...")
        # Add this section for saving embeddings after first epoch
        class SaveEmbeddingsCallback(pl.Callback):
            def on_train_epoch_end(self, trainer, pl_module):
                if trainer.current_epoch == 0:  # After first epoch
                    print("Saving embeddings after first epoch...")
                    embeddings_dir = hparams['save_dir'] + '/embeddings'
                    os.makedirs(embeddings_dir, exist_ok=True)
                    
                    # Get embeddings
                    with torch.no_grad():
                        n_id = torch.arange(pl_module.emb.weight.shape[0])
                        embeddings = pl_module.emb(n_id)
                    
                    # Save with special filename indicating first epoch
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename_base = f"embeddings_{hparams['run_id']}_{hparams['sparsity']}_epoch1_{timestamp}"
                    
                    # Save as pickle
                    embeddings_filename = os.path.join(embeddings_dir, f"{filename_base}.pkl")
                    with open(embeddings_filename, 'wb') as f:
                        pickle.dump({
                            'embeddings': embeddings,
                            'epoch': 1,
                            'run_id': hparams['run_id'],
                            'sparsity': hparams['sparsity']
                        }, f)
                    
                    # Save as numpy array
                    np_filename = os.path.join(embeddings_dir, f"{filename_base}.npy")
                    np.save(np_filename, embeddings.cpu().numpy())
                    
                    print(f"Saved first epoch embeddings to {embeddings_filename} and {np_filename}")

        # Add the callback to your trainer
        early_save_callback = SaveEmbeddingsCallback()
        trainer = pl.Trainer(
            devices=1 if torch.cuda.is_available() else 0,
            accelerator="gpu",
            logger=wandb_logger,
            max_epochs=hparams['max_epochs'],
            # COMMENT THIS OUT TO SAVE EMBEDDINGS AFTER FIRST EPOCH
            callbacks=[checkpoint_callback, lr_monitor, timer],
            # callbacks=[checkpoint_callback, lr_monitor, timer, early_save_callback],  # Add the new callback here
            gradient_clip_val=hparams['grad_clip'],
            profiler=hparams['profiler'],
            log_every_n_steps=hparams['log_every_n_steps'],
            val_check_interval=1,
            deterministic=True,
        )
        print("Trainer initialized successfully")

        print("Step 6: Starting model training...")
        trainer.fit(model, train_dataloader, val_dataloader)
        print("Training completed successfully")

        # After creating dataloaders
        leakage_results = check_data_leakage(train_neuroKG, val_neuroKG, test_neuroKG)
        print("Data leakage check results:")
        for k, v in leakage_results.items():
            print(f"{k}: {v}")


    except Exception as e:
        print(f"\nERROR OCCURRED:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        print(f"\nFull traceback:")
        print(traceback.format_exc())
        
        # Log error to wandb if possible
        try:
            if wandb.run is not None:
                wandb.log({"error": str(e)})
                wandb.finish(exit_code=1)
        except:
            pass
        
        raise

    finally:
        # Ensure wandb closes properly
        if wandb.run is not None:
            wandb.finish()


@torch.no_grad()
def save_embeddings(hparams):
    print("Saving embeddings...")
    
    # Create embeddings directory in current working directory if it doesn't exist
    embeddings_dir = hparams['save_dir'] + '/embeddings'
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # set seed
    pl.seed_everything(hparams['seed'], workers = True)

    # load DrugKG knowledge graph
    # drugKG = load_graph(hparams)
    # Replace the load_graph call with direct DGL loading
    drugKG = dgl.load_graphs(GRAPH_PATH)[0][0]

    # instantiate model
    print("Loading model from checkpoint " + hparams['best_ckpt'])
    model = HGT.load_from_checkpoint(
        checkpoint_path = str(Path(hparams['save_dir']) / 'checkpoints' / hparams['best_ckpt']), 
        num_nodes = drugKG.num_nodes(), num_ntypes = len(drugKG.ntypes),
        num_etypes = len(drugKG.canonical_etypes), hparams = hparams
    )

    # # generate embeddings
    # dataloader = dgl.dataloading.DataLoader(drugKG, batch_size = 1)
    # trainer = pl.Trainer(gpus=0, gradient_clip_val=hparams['grad_clip'])
    # embeddings = trainer.predict(model, dataloaders=dataloader) 

    # generate embeddings
    n_id = torch.arange(model.emb.weight.shape[0])
    embeddings = model.emb(n_id)

    # Update the saving logic to include run_id and sparsity
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename_base = f"embeddings_{hparams['run_id']}_{hparams['sparsity']}_{timestamp}"
    
    # Save as pickle
    embeddings_filename = os.path.join(embeddings_dir, f"{filename_base}.pkl")
    with open(embeddings_filename, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            # ... rest of your embeddings dict ...
        }, f)
    
    # Save as numpy array
    np_filename = os.path.join(embeddings_dir, f"{filename_base}.npy")
    np.save(np_filename, embeddings)
    
    print(f"Saved embeddings to {embeddings_filename} and {np_filename}")


if __name__ == "__main__":
    
    # get hyperparameters
    args = parse_args()
    hparams = get_hyperparameters(args)

    # after training is complete, save node embeddings
    if hparams['save_embeddings']:
        # save node embeddings from a trained model
        save_embeddings(hparams)
        pass
    else:
        # train model
        pretrain(hparams)