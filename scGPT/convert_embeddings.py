import pickle
import numpy as np
import torch
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

def convert_embeddings(pkl_path: str, npy_path: str):
    """Convert embeddings from .pkl to .npy format"""
    try:
        # Load embeddings from pickle file
        logger.info(f'Loading embeddings from {pkl_path}')
        with open(pkl_path, 'rb') as f:
            embeddings = pickle.load(f)
        
        # Inspect the loaded data
        logger.info(f'Type of loaded data: {type(embeddings)}')
        if isinstance(embeddings, dict):
            logger.info(f'Dictionary keys: {embeddings.keys()}')
            # If it's a dictionary, try to find the embeddings
            for key, value in embeddings.items():
                logger.info(f'Key: {key}, Type: {type(value)}')
                if isinstance(value, (np.ndarray, torch.Tensor)):
                    logger.info(f'Shape: {value.shape if hasattr(value, "shape") else "No shape"}')
                    embeddings = value
                    break
        
        # Convert to numpy array if it's a torch tensor
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        # Ensure we have a numpy array
        if not isinstance(embeddings, np.ndarray):
            logger.error(f'Expected numpy array or torch tensor, got {type(embeddings)}')
            raise ValueError('Loaded data is not in the expected format')
        
        # Save as numpy array
        logger.info(f'Saving embeddings to {npy_path}')
        logger.info(f'Final embeddings shape: {embeddings.shape}')
        logger.info(f'Final embeddings dtype: {embeddings.dtype}')
        np.save(npy_path, embeddings, allow_pickle=True)
        
        # Verify the conversion
        loaded_embeddings = np.load(npy_path, allow_pickle=True)
        logger.info(f'Successfully converted embeddings. Shape: {loaded_embeddings.shape}')
        logger.info(f'Data type: {loaded_embeddings.dtype}')
        
    except Exception as e:
        logger.error(f'Failed to convert embeddings: {e}')
        raise

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python convert_embeddings.py <input_pkl_file> <output_npy_file>")
        sys.exit(1)
    
    pkl_path = sys.argv[1]
    npy_path = sys.argv[2]
    convert_embeddings(pkl_path, npy_path) 