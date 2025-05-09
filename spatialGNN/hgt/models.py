'''
HETEROGENEOUS GRAPH TRANSFROMER
We define a heterogeneous graph transformer model to learn node embeddings on the knowledge graph.
'''

# standard imports
import numpy as np
import pandas as pd

# import PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# import DGL
import dgl
from dgl.nn.pytorch.conv import HGTConv

# import PyTorch Lightning
import pytorch_lightning as pl
import wandb

# path manipulation
from pathlib import Path

# import project config file
import sys
sys.path.append('../..')
import project_config

# custom imports
from utils import calculate_metrics
from analyze_embeddings import analyze_gene_embeddings

# check if CUDA is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# BILINEAR DECODER CLASS
class BilinearDecoder(pl.LightningModule): # overrides nn.Module

    # INITIALIZATION
    def __init__(self, num_etypes, embedding_dim):
        '''
        This function initializes a bilinear decoder.

        Args:
            num_etypes (int): Number of edge types.
            embedding_dim (int): Dimension of embedding (i.e., output dimension * number of attention heads).
        '''
        super().__init__()

        # edge-type specific learnable weights
        self.relation_weights = nn.Parameter(torch.Tensor(num_etypes, embedding_dim))

        # initialize weights
        nn.init.xavier_uniform_(self.relation_weights, gain = nn.init.calculate_gain('leaky_relu'))
    

    # ADD EDGE TYPE INDEX
    def add_edge_type_index(self, edge_graph):
        '''
        This function adds an integer edge type label to each edge in the graph. This is required for the decoder.
        Specifically, the edge type label is used to subset the right row of the relation weight matrix.
        
        Args:
            edge_graph (dgl.DGLGraph): Positive or negative edge graph.
        '''

        # iterate over the canonical edge types
        for edge_index, edge_type in enumerate(edge_graph.canonical_etypes):
        
            # get number of edges of that type
            num_edges = edge_graph.num_edges(edge_type)

            # add integer label to edge
            edge_graph.edges[edge_type].data['edge_type_index'] = torch.tensor(
                [edge_index] * num_edges,
                dtype=torch.int64,
                device=self.device
            )
    
    
    # DECODER
    def decode(self, edges):
        '''
        This is a user-defined function over the edges to generate the score for each edge.
        See https://docs.dgl.ai/en/0.9.x/generated/dgl.DGLGraph.apply_edges.html.
        '''
        
        # Convert edge_type_index to Long type
        edge_type_index = edges.data['edge_type_index'][0].long()
        
        # get source embeddings
        src_embeddings = edges.src['node_embedding']
        dst_embeddings = edges.dst['node_embedding']

        # apply activation function
        src_embeddings = F.leaky_relu(src_embeddings)
        dst_embeddings = F.leaky_relu(dst_embeddings)

        # get relation weight for specific edge type
        rel_weights = self.relation_weights[edge_type_index]

        # compute weighted dot product
        score = torch.sum(src_embeddings * rel_weights * dst_embeddings, dim=1)

        return {'score': score}


    # COMPUTE SCORE
    def compute_score(self, edge_graph):
        '''
        This function computes the score for positive or negative edges using dgl.DGLGraph.apply_edges.
        
        Args:
            edge_graph (dgl.DGLGraph): Positive or negative edge graph.
        '''

        with edge_graph.local_scope():

            # get edge types with > 0 number of edges in the positive graph
            nonzero_edge_types = [etype for etype in edge_graph.canonical_etypes if edge_graph.num_edges(etype) != 0]

            # compute score for positive graph
            for etype in nonzero_edge_types:
                edge_graph.apply_edges(self.decode, etype = etype)
            
            # return scores
            return edge_graph.edata['score']
    
    
    # FORWARD PASS
    def forward(self, subgraph, edge_graph, node_embeddings):
        '''
        This function performs a forward pass of the bilinear decoder.

        Args:
            subgraph (dgl.DGLHeteroGraph): Subgraph.
            edge_graph (dgl.DGLHeteroGraph): Either positive or negative graph that describes edges.
            node_embeddings (torch.Tensor): Node embeddings.
        '''
        # First ensure node_index exists in edge_graph
        for ntype in edge_graph.ntypes:
            if 'node_index' not in edge_graph.nodes[ntype].data:
                # Create node indices if they don't exist
                edge_graph.nodes[ntype].data['node_index'] = torch.arange(
                    edge_graph.num_nodes(ntype), 
                    device=edge_graph.device
                )

        # get subgraph node IDs
        subgraph_nodes = subgraph.ndata['node_index']

        # iterate over node types to assign node embeddings to edge graph
        for ntype in edge_graph.ntypes:
            # get edge graph node IDs
            edge_graph_nodes = edge_graph.nodes[ntype].data['node_index']
            
            # Create a mapping from subgraph nodes to their positions
            node_to_pos = {int(node.item()): pos for pos, node in enumerate(subgraph_nodes)}
            
            # Get positions of edge_graph nodes in subgraph
            try:
                edge_graph_positions = torch.tensor([
                    node_to_pos[int(node.item())]
                    for node in edge_graph_nodes
                    if int(node.item()) in node_to_pos
                ], device=edge_graph.device)
            except Exception as e:
                print(f"Error occurred while computing edge_graph_positions: {str(e)}")
                print("edge_graph_positions could not be computed.")
                # Optionally, print the edge_graph_nodes and node_to_pos for debugging
                print("edge_graph_nodes:", edge_graph_nodes)
                print("node_to_pos:", node_to_pos)
                raise  # Re-raise the exception to propagate the error
            
            # Get embeddings for these positions
            edge_graph.nodes[ntype].data['node_embedding'] = node_embeddings[edge_graph_positions]

        # add edge indices to edge graph
        self.add_edge_type_index(edge_graph)

        # compute scores over edges given by edge graph
        scores = self.compute_score(edge_graph)

        # return scores
        return scores
    

# HETEROGENEOUS GRAPH TRANSFORMER
class HGT(pl.LightningModule):
    
    # INITIALIZATION
    # Changes:
    # - num_feat is now 64
    # - hidden_dim is now 32
    # - output_dim is now 16
    def __init__(self, num_nodes, num_ntypes, num_etypes, num_feat = 64, num_heads = 4,
                 hidden_dim = 32, output_dim = 16, num_layers = 2,
                 dropout_prob = 0.5, pred_threshold = 0.5,
                 lr = 0.0001, wd = 0.0, lr_factor = 0.01, lr_patience = 100, lr_threshold = 1e-4,
                 lr_threshold_mode = 'rel', lr_cooldown = 0, min_lr = 1e-8, eps = 1e-8,
                 hparams = None, data_dir = None):
        '''
        This function initializes the model and defines the model hyperparameters and architecture.

        Args:
            num_nodes (int): Number of nodes in the graph.
            num_ntypes (int): Number of node types in the graph.
            num_etypes (int): Number of edge types in the graph.
            num_feat (int): Number of input features (i.e., hidden embedding dimension).
            num_heads (int): Number of attention heads.
            hidden_dim (int): Number of hidden units in the second to last HGT layer.
            output_dim (int): Number of output units.
            num_layers (int): Number of HGT layers.
            dropout_prob (float): Dropout probability.
            pred_threshold (float): Prediction threshold to compute metrics.
            lr (float): Learning rate.
            wd (float): Weight decay.
            lr_factor (float): Factor by which to reduce learning rate.
            lr_patience (int): Number of epochs with no improvement after which learning rate will be reduced.
            lr_threshold (float): Threshold for measuring the new optimum, to only focus on significant changes.
            lr_threshold_mode (str): One of ['rel', 'abs'].
            lr_cooldown (int): Number of epochs to wait before resuming normal operation after lr reduction.
            min_lr (float): A lower bound on the learning rate of all param groups or each group respectively.
            eps (float): Term added to the denominator to improve numerical stability.
            hparams (dict): Dictionary of model hyperparameters. Will override all other arguments if not None.
            data_dir (str): Directory containing mapping dictionaries.
        '''

        super().__init__()

        # if hparams_dict is None, construct dictionary from arguments
        if hparams is None:
            hparams = locals()

        # save model hyperparameters
        self.save_hyperparameters(hparams)
        self.num_feat = hparams['num_feat']
        self.num_heads = hparams['num_heads']
        self.hidden_dim = hparams['hidden_dim']
        self.output_dim = hparams['output_dim']
        self.num_layers = hparams['num_layers']
        self.dropout_prob = hparams['dropout_prob']
        self.pred_threshold = hparams['pred_threshold']

        # learning rate parameters
        self.lr = hparams['lr']
        self.wd = hparams['wd']
        self.lr_factor = hparams['lr_factor']
        self.lr_patience = hparams['lr_patience']
        self.lr_threshold = hparams['lr_threshold']
        self.lr_threshold_mode = hparams['lr_threshold_mode']
        self.lr_cooldown = hparams['lr_cooldown']
        self.min_lr = hparams['min_lr']
        self.eps = hparams['eps']

        # loss function weighting
        self.loss_weight = hparams['loss_weight']
        self.loss_weight2 = hparams['loss_weight2']

        # calculate sizes of hidden dimensions
        self.h_dim_1 = hidden_dim * 2
        self.h_dim_2 = hidden_dim

        # define node embeddings
        self.emb = nn.Embedding(num_nodes, num_feat)

        # layer 1
        self.conv1 = HGTConv(in_size = num_feat, head_size = self.h_dim_1, num_heads = num_heads,
                                num_ntypes = num_ntypes, num_etypes = num_etypes, dropout = 0.2, use_norm = True)
        
        # layer normalization 1
        self.norm1 = nn.LayerNorm(self.h_dim_1 * num_heads)

        if self.num_layers == 2:
        
            # layer 2
            self.conv2 = HGTConv(in_size = self.h_dim_1 * num_heads, head_size = output_dim, num_heads = num_heads,
                                    num_ntypes = num_ntypes, num_etypes = num_etypes, dropout = 0.2, use_norm = True)
            
        elif self.num_layers == 3:
        
            # layer 2
            self.conv2 = HGTConv(in_size = self.h_dim_1 * num_heads, head_size = self.h_dim_2, num_heads = num_heads,
                                    num_ntypes = num_ntypes, num_etypes = num_etypes, dropout = 0.2, use_norm = True)

            # layer normalization 2
            self.norm2 = nn.LayerNorm(self.h_dim_2 * num_heads)

            # layer 3
            self.conv3 = HGTConv(in_size = self.h_dim_2 * num_heads, head_size = output_dim, num_heads = num_heads,
                                    num_ntypes = num_ntypes, num_etypes = num_etypes, dropout = 0.2, use_norm = True)
            
        else:

            # raise error
            raise ValueError('Number of layers must be 2 or 3.')

        # define decoder
        self.decoder = BilinearDecoder(num_etypes, output_dim * num_heads)
        
        self.data_dir = data_dir
        self.graph = None
        
    
    
    # FORWARD PASS
    def forward(self, subgraph):
        '''
        This function performs a forward pass of the model. Note that the subgraph must be converted to from a 
        heterogeneous graph to homogeneous graph for efficiency.

        Args:
            subgraph (dgl.DGLGraph): Subgraph containing the nodes and edges for the current batch.
        '''

        # get global indices
        global_node_indices = subgraph.ndata['node_index']

        # get node embeddings from the first MFG layer
        x = self.emb(global_node_indices)

        # pass node embedding through first two layers
        x = self.conv1(subgraph, x, subgraph.ndata[dgl.NTYPE], subgraph.edata[dgl.ETYPE])
        x = self.norm1(x)
        x = F.leaky_relu(x)
        x = self.conv2(subgraph, x, subgraph.ndata[dgl.NTYPE], subgraph.edata[dgl.ETYPE])

        # check if 3 layers
        if self.num_layers == 3:

            # pass node embedding through layer 3
            x = self.norm2(x)
            x = F.leaky_relu(x)
            x = self.conv3(subgraph, x, subgraph.ndata[dgl.NTYPE], subgraph.edata[dgl.ETYPE])

        # return node embeddings
        return x
    

    # STEP FUNCTION USED FOR TRAINING, VALIDATION, AND TESTING
    def _step(self, input_nodes, pos_graph, neg_graph, subgraph, mode):
        '''Defines the step that is run on each batch of data. PyTorch Lightning handles steps including:
            - Moving data to the correct device.
            - Epoch and batch iteration.
            - optimizer.step(), loss.backward(), optimizer.zero_grad() calls.
            - Calling of model.eval(), enabling/disabling grads during evaluation.
            - Logging of metrics.
        
        Args:
            input_nodes (torch.Tensor): Input nodes.
            pos_graph (dgl.DGLHeteroGraph): Positive graph.
            neg_graph (dgl.DGLHeteroGraph): Negative graph.
            subgraph (dgl.DGLHeteroGraph): Subgraph.
            mode (str): The mode of the step (train, val, test).
        '''

        # get batch size by summing number of nodes in each node type
        batch_size = sum([x.shape[0] for x in input_nodes.values()])

        # convert heterogeneous graph to homogeneous graph for efficiency
        # Add node indices before conversion
        for ntype in subgraph.ntypes:
            # Create tensor on the same device as the graph
            node_indices = torch.arange(subgraph.num_nodes(ntype), device=subgraph.device)
            subgraph.nodes[ntype].data['node_index'] = node_indices
        
        subgraph = dgl.to_homogeneous(subgraph, ndata=['node_index'])
        
        # # send to GPU
        # subgraph = subgraph.to(device)
        # pos_graph = pos_graph.to(device)
        # neg_graph = neg_graph.to(device)

        # get node embeddings
        node_embeddings = self.forward(subgraph)

        # # assert that node IDs and indices are identical between positive and negative graphs
        # pos_IDs = pos_graph.ndata['_ID']
        # neg_IDs = neg_graph.ndata['_ID']
        # assert all([torch.equal(value, neg_IDs[key]) for key, value in pos_IDs.items()])

        # set node index of negative graph for decoder
        neg_graph.ndata['node_index'] = pos_graph.ndata['node_index']

        # compute score from decoder
        pos_scores = self.decoder(subgraph, pos_graph, node_embeddings)
        neg_scores = self.decoder(subgraph, neg_graph, node_embeddings)

        # compute loss
        loss, metrics, edge_type_metrics = self.compute_loss(pos_scores, neg_scores)

        # return loss and metrics
        return loss, metrics, edge_type_metrics, batch_size
    

    # TRAINING STEP
    def training_step(self, batch, batch_idx):
        '''Defines the step that is run on each batch of training data.'''

        # get batch elements
        input_nodes, pos_graph, neg_graph, subgraph = batch

        # get loss and metrics
        loss, metrics, edge_type_metrics, batch_size = self._step(input_nodes, pos_graph, neg_graph, subgraph, mode = 'train')

        # log loss and metrics
        values = {"train/loss": loss.detach(),
                  "train/accuracy": metrics['accuracy'],
                  "train/ap": metrics['ap'],
                  "train/f1": metrics['f1'],
                  "train/auroc": metrics['auroc']}
        self.log_dict(values, batch_size = batch_size)

        # log edge-type specific metrics
        for edge_type, metric in edge_type_metrics.items():
            edge_type_label = [label.replace('/', '_') for label in edge_type]
            edge_type_label = '-'.join(edge_type_label)
            values = {f"edge_type_metrics/train/{edge_type_label}/accuracy": metric['accuracy'],
                      f"edge_type_metrics/train/{edge_type_label}/ap": metric['ap'],
                      f"edge_type_metrics/train/{edge_type_label}/f1": metric['f1'],
                      f"edge_type_metrics/train/{edge_type_label}/auroc": metric['auroc']}
            self.log_dict(values, batch_size = batch_size)

        return loss
    

    # VALIDATION STEP
    def validation_step(self, batch, batch_idx):
        '''Defines the step that is run on each batch of validation data.'''

        # get batch elements
        input_nodes, pos_graph, neg_graph, subgraph = batch

        # get loss and metrics
        loss, metrics, edge_type_metrics, batch_size = self._step(input_nodes, pos_graph, neg_graph, subgraph, mode = 'val')

        # log loss and metrics
        values = {"val/loss": loss.detach(),
                  "val/accuracy": metrics['accuracy'],
                  "val/ap": metrics['ap'],
                  "val/f1": metrics['f1'],
                  "val/auroc": metrics['auroc']}
        self.log_dict(values, batch_size = batch_size)

        # log edge-type specific metrics
        for edge_type, metric in edge_type_metrics.items():
            edge_type_label = [label.replace('/', '_') for label in edge_type]
            edge_type_label = '-'.join(edge_type_label)
            values = {f"edge_type_metrics/val/{edge_type_label}/accuracy": metric['accuracy'],
                      f"edge_type_metrics/val/{edge_type_label}/ap": metric['ap'],
                      f"edge_type_metrics/val/{edge_type_label}/f1": metric['f1'],
                      f"edge_type_metrics/val/{edge_type_label}/auroc": metric['auroc']}
            self.log_dict(values, batch_size = batch_size)

        return loss


    # TEST STEP
    def test_step(self, batch, batch_idx):
        '''Defines the step that is run on each batch of test data.'''

        # get batch elements
        input_nodes, pos_graph, neg_graph, subgraph = batch

        # get loss and metrics
        loss, metrics, edge_type_metrics, batch_size = self._step(input_nodes, pos_graph, neg_graph, subgraph, mode = 'test')

        # log loss and metrics
        values = {"test/loss": loss.detach(),
                  "test/accuracy": metrics['accuracy'],
                  "test/ap": metrics['ap'],
                  "test/f1": metrics['f1'],
                  "test/auroc": metrics['auroc']}
        self.log_dict(values, batch_size = batch_size)

        # log edge-type specific metrics
        for edge_type, metric in edge_type_metrics.items():
            edge_type_label = [label.replace('/', '_') for label in edge_type]
            edge_type_label = '-'.join(edge_type_label)
            values = {f"edge_type_metrics/test/{edge_type_label}/accuracy": metric['accuracy'],
                      f"edge_type_metrics/test/{edge_type_label}/ap": metric['ap'],
                      f"edge_type_metrics/test/{edge_type_label}/f1": metric['f1'],
                      f"edge_type_metrics/test/{edge_type_label}/auroc": metric['auroc']}
            self.log_dict(values, batch_size = batch_size)

    
    # LOSS FUNCTION
    def compute_loss(self, pos_scores, neg_scores):
        '''
        This function computes the loss and metrics for the current batch.
        '''
        # Store distributions for analysis
        pos_dist = {}
        neg_dist = {}
        
        # Initialize lists to store all predictions and targets
        all_preds = []
        all_targets = []
        
        # Track scores by edge type
        for edge_type in pos_scores.keys():
            pos_dist[edge_type] = pos_scores[edge_type].detach().cpu().numpy()
            neg_dist[edge_type] = neg_scores[edge_type].detach().cpu().numpy()
            
            # Get predictions for this edge type
            pos_pred_type = torch.sigmoid(pos_scores[edge_type])
            neg_pred_type = torch.sigmoid(neg_scores[edge_type])
            
            # Add to combined predictions and targets
            all_preds.extend([pos_pred_type, neg_pred_type])
            all_targets.extend([torch.ones_like(pos_pred_type), torch.zeros_like(neg_pred_type)])
            
            # Log distributions to wandb
            if self.logger:
                self.logger.experiment.log({
                    f"score_dist/pos_{edge_type}": wandb.Histogram(pos_dist[edge_type]),
                    f"score_dist/neg_{edge_type}": wandb.Histogram(neg_dist[edge_type])
                })

        # Combine all predictions and targets
        pred = torch.cat(all_preds)
        target = torch.cat(all_targets)
        
        # Define edge type weights
        edge_weights = {
            ('gene', 'ispartof', 'spatialcontext'): self.loss_weight,  # Higher weight for ispartof edges
            ('spatialcontext', 'iscontextof', 'gene'): self.loss_weight2,
            ('gene', 'interactswith', 'gene'): 1.0,
            ('spatialcontext', 'issimilarto', 'spatialcontext'): 1.0,
            # ... other edge types ...
        }
        
        # Initialize weighted loss
        total_loss = 0
        total_weights = 0
        
        # Compute weighted loss for each edge type
        for edge_type in pos_scores.keys():
            print("edge type debug: ", edge_type)
            # Get predictions and targets for this edge type
            pos_pred_type = torch.sigmoid(pos_scores[edge_type])
            neg_pred_type = torch.sigmoid(neg_scores[edge_type])
            
            pos_target = torch.ones(pos_pred_type.shape[0], device=self.device)
            neg_target = torch.zeros(neg_pred_type.shape[0], device=self.device)
            
            # Combine predictions and targets
            pred_type = torch.cat((pos_pred_type, neg_pred_type))
            target_type = torch.cat((pos_target, neg_target))
            
            # Get weight for this edge type (default to 1.0 if not specified)
            weight = edge_weights.get(edge_type, 1.0)
            
            # Compute weighted loss for this edge type
            type_loss = F.binary_cross_entropy(pred_type, target_type, reduction="mean")
            total_loss += weight * type_loss
            total_weights += weight
        
        # Normalize loss by total weights
        loss = total_loss / total_weights
        
        # Move prediction and target to CPU
        pred = pred.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        
        # Calculate overall metrics
        metrics = calculate_metrics(pred, target, self.pred_threshold)
        
        # Calculate edge type-specific metrics
        edge_type_metrics = {}
        for edge_type in pos_scores.keys():
            pos_preds = torch.sigmoid(pos_scores[edge_type]).cpu().detach().numpy()
            neg_preds = torch.sigmoid(neg_scores[edge_type]).cpu().detach().numpy()
            
            type_preds = np.concatenate([pos_preds, neg_preds])
            type_targets = np.concatenate([np.ones_like(pos_preds), np.zeros_like(neg_preds)])
            
            edge_type_metrics[edge_type] = calculate_metrics(type_preds, type_targets, self.pred_threshold)
            
            # Log edge-type specific metrics
            if self.logger:
                self.logger.experiment.log({
                    f"edge_metrics/{edge_type}/pos_mean": np.mean(pos_preds),
                    f"edge_metrics/{edge_type}/neg_mean": np.mean(neg_preds),
                    f"edge_metrics/{edge_type}/pos_std": np.std(pos_preds),
                    f"edge_metrics/{edge_type}/neg_std": np.std(neg_preds),
                })

        return loss, metrics, edge_type_metrics
    

    # OPTIMIZER AND SCHEDULER
    def configure_optimizers(self):
        '''
        This function is called by PyTorch Lightning to get the optimizer and scheduler.
        We reduce the learning rate by a factor of lr_factor if the validation loss does not improve for lr_patience epochs.

        Args:
            None

        Returns:
            dict: Dictionary containing the optimizer and scheduler.
        '''
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay = self.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode = 'min', factor = self.lr_factor, patience = self.lr_patience,
            threshold = self.lr_threshold, threshold_mode = self.lr_threshold_mode,
            cooldown = self.lr_cooldown, min_lr = self.min_lr, eps = self.eps
        )
        
        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    'name': 'curr_lr'
                    },
                }

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Called at the end of each training batch"""
        try:
            if self.global_step == 0 or (self.global_step % 10 == 0):
                if self.graph is not None:
                    # Get processed embeddings through model forward pass
                    self.eval()  # Set model to eval mode temporarily
                    with torch.no_grad():
                        # Move graph to same device as model
                        device = next(self.parameters()).device
                        graph = self.graph.to(device)
                        
                        # Add node indices before conversion
                        for ntype in graph.ntypes:
                            graph.nodes[ntype].data['node_index'] = torch.arange(graph.num_nodes(ntype), device=device)
                        
                        # Convert to homogeneous and get embeddings
                        homo_graph = dgl.to_homogeneous(graph, ndata=['node_index'])
                        all_embeddings = self.forward(homo_graph)
                        
                        # Extract gene embeddings
                        gene_mask = homo_graph.ndata[dgl.NTYPE] == graph.ntypes.index('gene')
                        embeddings = all_embeddings[gene_mask]
                    self.train()  # Set model back to train mode
                    
                    # Calculate and print embedding statistics
                    max_val = torch.max(embeddings).item()
                    min_val = torch.min(embeddings).item()
                    mean_val = torch.mean(embeddings).item()
                    std_val = torch.std(embeddings).item()
                    
                    # Get top 5 maximum values and their indices
                    top_5_values, top_5_indices = torch.topk(embeddings.view(-1), 5)
                    
                    # Convert to numpy for easier printing
                    top_5_values = top_5_values.cpu().numpy()
                    top_5_indices = top_5_indices.cpu().numpy()
                    
                    print(f"\nEmbedding Statistics at step {self.global_step}:")
                    print(f"Max value: {max_val:.4f}")
                    print(f"Min value: {min_val:.4f}")
                    print(f"Mean value: {mean_val:.4f}")
                    print(f"Std value: {std_val:.4f}")
                    print(f"Top 5 values: {top_5_values}")
                    print(f"Indices of top 5 values: {top_5_indices}")
                    
                    # Create histogram of embedding values
                    plt.figure(figsize=(10, 6))
                    embeddings_flat = embeddings.cpu().numpy().flatten()
                    plt.hist(embeddings_flat, bins=50, density=True, alpha=0.7)
                    plt.axvline(mean_val, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.2f}')
                    plt.axvline(mean_val + std_val, color='g', linestyle='dashed', linewidth=1, label=f'Mean ± Std')
                    plt.axvline(mean_val - std_val, color='g', linestyle='dashed', linewidth=1)
                    plt.title(f'Distribution of Embedding Values at Step {self.global_step}')
                    plt.xlabel('Embedding Value')
                    plt.ylabel('Density')
                    plt.legend()
                    hist_fig = plt.gcf()
                    
                    # Log statistics and figures to wandb
                    self.logger.experiment.log({
                        "embeddings/max_value": max_val,
                        "embeddings/min_value": min_val,
                        "embeddings/mean_value": mean_val,
                        "embeddings/std_value": std_val,
                        "embeddings/top_5_mean": np.mean(top_5_values),
                        "embeddings/distribution": wandb.Image(hist_fig),
                        "global_step": self.global_step
                    })
                    plt.close(hist_fig)  # Clean up histogram figure
                    
                    # Create and log UMAP visualization
                    if self.global_step % 100 == 0:
                        umap_fig = analyze_gene_embeddings(self, self.graph, self.data_dir, hparams=self.hparams, save_final_embeddings=True)
                        if umap_fig is not None:
                            self.logger.experiment.log({
                                "gene_embeddings_umap": wandb.Image(umap_fig),
                                "global_step": self.global_step
                            })
                            print(f"Logged visualizations at global step {self.global_step}")
                            plt.close(umap_fig)  # Clean up UMAP figure
                        else:
                            print(f"Warning: Failed to create UMAP visualization at global step {self.global_step}")
                        
        except Exception as e:
            print(f"Error in on_train_batch_end at step {self.global_step}: {str(e)}")
            import traceback
            print(f"Full traceback:\n{traceback.format_exc()}")

    def on_train_end(self):
        """Called when training ends"""
        try:
            print("Training completed. Saving final embeddings...")
            if self.graph is not None:
                analyze_gene_embeddings(self, self.graph, self.data_dir, hparams=self.hparams, save_final_embeddings=True)
                print("Final embeddings saved successfully")
        except Exception as e:
            print(f"Error saving final embeddings: {str(e)}")
            import traceback
            print(f"Full traceback:\n{traceback.format_exc()}")