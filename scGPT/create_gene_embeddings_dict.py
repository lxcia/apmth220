import pickle
import numpy as np
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def create_gene_embeddings_dict(pickle_path: str, output_path: str):
    """Create a dictionary with gene names and embeddings in the format expected by load_custom_embeddings.
    The output will be a dictionary with two keys:
    - 'embeddings': numpy array of shape (num_genes, embedding_dim)
    - 'gene_names': numpy array of gene names
    """
    try:
        logger.info(f'Loading pickle file: {pickle_path}')
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract the components
        gene_names = data['gene_names']
        embeddings = data['embeddings']
        
        # Create the dictionary in the exact format needed
        embeddings_dict = {
            'embeddings': embeddings,
            'gene_names': np.array(gene_names)
        }
        
        # Save the dictionary in .npy format
        logger.info(f'Saving to {output_path}')
        # Use np.savez instead of np.save to handle the dictionary properly
        np.savez(output_path, **embeddings_dict)
        
        logger.info(f'Successfully created dictionary with {len(gene_names)} genes')
        logger.info(f'Embeddings shape: {embeddings.shape}')
        logger.info(f'Gene names shape: {len(gene_names)}')
        
    except Exception as e:
        logger.error(f'Error creating gene embeddings dictionary: {e}')
        raise

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create gene to embedding mapping')
    parser.add_argument('pickle_path', help='Path to the input pickle file')
    parser.add_argument('output_path', help='Path to save the output mapping (should end in .npz)')
    args = parser.parse_args()
    
    create_gene_embeddings_dict(args.pickle_path, args.output_path) 