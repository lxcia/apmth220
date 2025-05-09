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

def convert_embeddings(input_path: str, output_path: str, gene_names_path: str = None):
    """Convert embeddings from numpy array to dictionary format.
    
    Args:
        input_path: Path to the input embeddings file (numpy array)
        output_path: Path to save the output dictionary format
        gene_names_path: Optional path to a file containing gene names
    """
    try:
        # Load the current embeddings
        logger.info(f'Loading embeddings from {input_path}')
        embeddings = np.load(input_path, allow_pickle=True)
        logger.info(f'Loaded embeddings with shape: {embeddings.shape}')
        
        # Get gene names
        if gene_names_path:
            # Load gene names from file if provided
            logger.info(f'Loading gene names from {gene_names_path}')
            with open(gene_names_path, 'r') as f:
                gene_names = [line.strip() for line in f]
        else:
            # If no gene names file provided, create generic names
            logger.info('No gene names file provided, creating generic names')
            gene_names = [f'gene_{i}' for i in range(embeddings.shape[0])]
        
        # Verify the number of gene names matches the number of embeddings
        if len(gene_names) != embeddings.shape[0]:
            raise ValueError(f"Number of gene names ({len(gene_names)}) does not match number of embeddings ({embeddings.shape[0]})")
        
        # Create dictionary format
        embeddings_dict = {
            'embeddings': embeddings,
            'gene_names': gene_names
        }
        
        # Save the dictionary
        logger.info(f'Saving dictionary format to {output_path}')
        np.save(output_path, embeddings_dict, allow_pickle=True)
        logger.info('Conversion complete!')
        
    except Exception as e:
        logger.error(f'Error during conversion: {e}')
        raise

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert embeddings to dictionary format')
    parser.add_argument('input_path', help='Path to input embeddings file')
    parser.add_argument('output_path', help='Path to save output dictionary format')
    parser.add_argument('--gene_names', help='Path to file containing gene names (one per line)')
    args = parser.parse_args()
    
    convert_embeddings(args.input_path, args.output_path, args.gene_names) 