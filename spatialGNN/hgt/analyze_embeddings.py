'''
ANALYZE GENE EMBEDDINGS
'''
import pickle
import torch
import dgl
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.manifold import TSNE
from datetime import datetime

def analyze_gene_embeddings(model, graph, data_dir, hparams, save_final_embeddings=False):
    try:
        # Load the mapping files
        mapping_file = "/n/home08/kel331/spatialGNN/data/map_ints_to_genes.pkl"
        context_mapping_file = "/n/home08/kel331/spatialGNN/data/map_ints_to_contexts.pkl"
        
        if not os.path.exists(mapping_file) or not os.path.exists(context_mapping_file):
            print(f"Warning: Required mapping files not found")
            return None
            
        with open(mapping_file, 'rb') as f:
            map_ints_to_genes = pickle.load(f)
        with open(context_mapping_file, 'rb') as f:
            map_ints_to_contexts = pickle.load(f)
        
        # Get embeddings
        model.eval()
        with torch.no_grad():
            # Move graph to same device as model
            device = next(model.parameters()).device
            graph = graph.to(device)
            
            # Add node indices before conversion
            for ntype in graph.ntypes:
                graph.nodes[ntype].data['node_index'] = torch.arange(graph.num_nodes(ntype), device=device)
            
            # Convert to homogeneous graph
            homo_graph = dgl.to_homogeneous(graph, ndata=['node_index'])
            
            # Get gene embeddings directly from the model's embedding layer if it exists
            if hasattr(model, 'node_embeddings') and hasattr(model.node_embeddings, 'weight'):
                all_embeddings = model.node_embeddings.weight
            elif hasattr(model, 'emb') and hasattr(model.emb, 'weight'):
                all_embeddings = model.emb.weight
            else:
                # Fallback to current method
                all_embeddings = model.forward(homo_graph)
            
            # Extract gene embeddings
            gene_type_id = graph.ntypes.index('gene')
            gene_mask = homo_graph.ndata[dgl.NTYPE] == gene_type_id
            gene_embeddings = all_embeddings[gene_mask]
            gene_indices = homo_graph.ndata['node_index'][gene_mask]
            
            # Move to CPU after operations are complete
            gene_embeddings = gene_embeddings.cpu().numpy()
            gene_indices = gene_indices.cpu().numpy()
        
        # Add quality checks
        print("\nEmbedding Statistics:")
        print(f"Shape: {gene_embeddings.shape}")
        print(f"Mean: {np.mean(gene_embeddings):.4f}")
        print(f"Std: {np.std(gene_embeddings):.4f}")
        print(f"Min: {np.min(gene_embeddings):.4f}")
        print(f"Max: {np.max(gene_embeddings):.4f}")
        
        # Check for NaN or Inf values
        if np.any(np.isnan(gene_embeddings)) or np.any(np.isinf(gene_embeddings)):
            print("WARNING: Embeddings contain NaN or Inf values!")
            
        # Check if embeddings are too uniform
        if np.std(gene_embeddings) < 1e-6:
            print("WARNING: Embeddings have very low variance!")
        
        # Get gene contexts from the graph
        gene_contexts = []
        for idx in gene_indices:
            # Get the contexts for this gene from the graph
            gene_id = int(idx)
            # Find edges of type 'ispartof' from this gene
            edges = graph.edges(etype='ispartof')
            src_idx = (edges[0] == gene_id).nonzero(as_tuple=True)[0]
            if len(src_idx) > 0:
                context_ids = edges[1][src_idx].cpu().numpy()
                contexts = [str(map_ints_to_contexts[int(c)]) for c in context_ids]
                gene_contexts.append(', '.join(contexts))
            else:
                gene_contexts.append('Unknown')
        
        # Create DataFrame with embeddings and metadata
        embedding_data = []
        for idx, emb, contexts in zip(gene_indices, gene_embeddings, gene_contexts):
            gene_name = map_ints_to_genes[int(idx)]
            embedding_data.append({
                'gene': gene_name,
                'embedding': emb,
                'context': contexts
            })
        
        df = pd.DataFrame(embedding_data)
        
        # Load or create selected contexts for visualization
        contexts_cache_file = f"{data_dir}/selected_contexts_sets.pkl"
        if os.path.exists(contexts_cache_file):
            with open(contexts_cache_file, 'rb') as f:
                selected_contexts_sets = pickle.load(f)
            print("Loading previously selected context sets")
        else:
            # Select 3 sets of 20 random contexts
            all_contexts = df['context'].unique()
            selected_contexts_sets = []
            remaining_contexts = all_contexts.copy()
            
            for i in range(3):  # Create 3 sets
                if len(remaining_contexts) >= 20:
                    selected_contexts = np.random.choice(remaining_contexts, 20, replace=False)
                    selected_contexts_sets.append(selected_contexts)
                    # Remove selected contexts from remaining pool
                    remaining_contexts = remaining_contexts[~np.isin(remaining_contexts, selected_contexts)]
                else:
                    print(f"Warning: Not enough contexts for set {i+1}")
                    selected_contexts_sets.append(remaining_contexts)
                
            # Save the selected contexts
            with open(contexts_cache_file, 'wb') as f:
                pickle.dump(selected_contexts_sets, f)
            print("Selected and saved new random context sets")

        # Create visualization parameters for both UMAP and t-SNE
        viz_params = {
            'umap': [
                {'n_neighbors': 10, 'min_dist': 0.05, 'title': 'UMAP (Local Structure)'},
                {'n_neighbors': 30, 'min_dist': 0.3, 'title': 'UMAP (Balanced)'},
                {'n_neighbors': 100, 'min_dist': 0.6, 'title': 'UMAP (Global Structure)'}
            ],
            'tsne': [
                {'perplexity': 5, 'title': 't-SNE (Local Structure)'},
                {'perplexity': 30, 'title': 't-SNE (Balanced)'},
                {'perplexity': 50, 'title': 't-SNE (Global Structure)'}
            ]
        }

        # Create a figure with 3 rows and 6 columns (3 UMAP + 3 t-SNE)
        fig, axes = plt.subplots(3, 6, figsize=(36, 24))
        
        for row, selected_contexts in enumerate(selected_contexts_sets):
            # Filter DataFrame for current context set
            df_subset = df[df['context'].isin(selected_contexts)].copy()
            print(f"\nContext Set {row + 1}:")
            print(f"Number of genes: {len(df_subset)}")
            print("Contexts:", ', '.join(selected_contexts))
            
            # Stack embeddings once for both UMAP and t-SNE
            embeddings_array = np.stack(df_subset['embedding'].values)
            
            # Create color palette for this set of contexts
            n_colors = len(selected_contexts)
            palette = sns.color_palette("bright", n_colors=n_colors)
            color_dict = dict(zip(selected_contexts, palette))
            
            # UMAP visualizations (first 3 columns)
            for col, params in enumerate(viz_params['umap']):
                reducer = umap.UMAP(
                    n_neighbors=params['n_neighbors'],
                    min_dist=params['min_dist'],
                    n_components=2,
                    metric='euclidean',
                    random_state=42
                )
                
                embeddings_2d = reducer.fit_transform(embeddings_array)
                
                # Add coordinates to dataframe
                df_subset['Dim1'] = embeddings_2d[:, 0]
                df_subset['Dim2'] = embeddings_2d[:, 1]
                
                # Create scatter plot
                sns.scatterplot(
                    data=df_subset,
                    x='Dim1',
                    y='Dim2',
                    hue='context',
                    palette=color_dict,
                    alpha=0.5,
                    s=10,
                    ax=axes[row, col]
                )
                
                axes[row, col].set_title(f"{params['title']}")
                if col != 2:
                    axes[row, col].legend([],[], frameon=False)
                
            # t-SNE visualizations (last 3 columns)
            for col, params in enumerate(viz_params['tsne']):
                tsne = TSNE(
                    n_components=2,
                    perplexity=params['perplexity'],
                    random_state=42,
                    n_iter=1000
                )
                
                embeddings_2d = tsne.fit_transform(embeddings_array)
                
                # Add coordinates to dataframe
                df_subset['Dim1'] = embeddings_2d[:, 0]
                df_subset['Dim2'] = embeddings_2d[:, 1]
                
                # Create scatter plot
                sns.scatterplot(
                    data=df_subset,
                    x='Dim1',
                    y='Dim2',
                    hue='context',
                    palette=color_dict,
                    alpha=0.5,
                    s=10,
                    ax=axes[row, col+3]  # offset by 3 for t-SNE columns
                )
                
                axes[row, col+3].set_title(f"{params['title']}")
                if col != 2:
                    axes[row, col+3].legend([],[], frameon=False)
                
            # Add legends only to the rightmost plots
            axes[row, 2].legend(bbox_to_anchor=(1.05, 1), 
                              loc='upper left', 
                              borderaxespad=0, 
                              fontsize=8,
                              title="Contexts")
            axes[row, 5].legend(bbox_to_anchor=(1.05, 1), 
                              loc='upper left', 
                              borderaxespad=0, 
                              fontsize=8,
                              title="Contexts")

        # Create separate UMAP and t-SNE plots for all embeddings
        all_embeddings_array = np.stack(df['embedding'].values)
        
        # UMAP for all embeddings
        umap_reducer = umap.UMAP(
            n_neighbors=30,  # You can adjust these parameters
            min_dist=0.1,
            n_components=2,
            metric='euclidean',
            random_state=42
        )
        all_umap_embeddings_2d = umap_reducer.fit_transform(all_embeddings_array)
        
        # Add coordinates to dataframe
        df['All_UMAP_Dim1'] = all_umap_embeddings_2d[:, 0]
        df['All_UMAP_Dim2'] = all_umap_embeddings_2d[:, 1]
        
        # Create scatter plot for all UMAP embeddings
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=df,
            x='All_UMAP_Dim1',
            y='All_UMAP_Dim2',
            hue='context',
            palette='bright',
            alpha=0.5,
            s=10
        )
        plt.title("UMAP of All Embeddings")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=8, title="Contexts")
        plt.tight_layout()
        plt.savefig(f"{data_dir}/umap_all_embeddings.png")
        plt.close()

        # t-SNE for all embeddings
        tsne = TSNE(
            n_components=2,
            perplexity=30,  # You can adjust these parameters
            random_state=42,
            n_iter=1000
        )
        all_tsne_embeddings_2d = tsne.fit_transform(all_embeddings_array)
        
        # Add coordinates to dataframe
        df['All_tSNE_Dim1'] = all_tsne_embeddings_2d[:, 0]
        df['All_tSNE_Dim2'] = all_tsne_embeddings_2d[:, 1]
        
        # Create scatter plot for all t-SNE embeddings
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=df,
            x='All_tSNE_Dim1',
            y='All_tSNE_Dim2',
            hue='context',
            palette='bright',
            alpha=0.5,
            s=10
        )
        plt.title("t-SNE of All Embeddings")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=8, title="Contexts")
        plt.tight_layout()
        plt.savefig(f"{data_dir}/tsne_all_embeddings.png")
        plt.close()

        # Log plots to wandb
        import wandb
        wandb.log({
            "UMAP of All Embeddings": wandb.Image(f"{data_dir}/umap_all_embeddings.png"),
            "t-SNE of All Embeddings": wandb.Image(f"{data_dir}/tsne_all_embeddings.png")
        })

        # After creating the DataFrame with embeddings and metadata
        if save_final_embeddings:
            # Create embeddings directory in current working directory if it doesn't exist
            embeddings_dir = hparams["save_dir"] + '/embeddings'
            os.makedirs(embeddings_dir, exist_ok=True)
            
            # Create embeddings dictionary with all necessary information
            embeddings_dict = {
                'embeddings': gene_embeddings,
                'gene_indices': gene_indices,
                'gene_names': [map_ints_to_genes[int(idx)] for idx in gene_indices],
                'contexts': gene_contexts,
                'mapping': {
                    'genes': map_ints_to_genes,
                    'contexts': map_ints_to_contexts
                }
            }
            
            # Generate filename with current timestamp
            run = hparams["run_name"]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            embeddings_filename = os.path.join(embeddings_dir, f"embeddings_{hparams['run_id']}_{{epoch}}-{{step}}_{timestamp}")
            
            # Save embeddings
            with open(embeddings_filename, 'wb') as f:
                pickle.dump(embeddings_dict, f)
            
            print(f"Saved final embeddings to {embeddings_filename}")
            
            # Also save as numpy array for convenience
            np_filename = os.path.join(embeddings_dir, f"embeddings_{hparams['run_id']}_{{epoch}}-{{step}}_array_{timestamp}.npy")
            np.save(np_filename, gene_embeddings)
            print(f"Saved embeddings array to {np_filename}")

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"Error in analyze_gene_embeddings: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return None
