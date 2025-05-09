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

def examine_pickle(file_path: str):
    """Examine and print metadata and a sample of 5 genes from a pickle file."""
    try:
        logger.info(f'Loading pickle file: {file_path}')
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        logger.info('File Structure:')
        logger.info('=' * 50)
        
        if isinstance(data, dict):
            logger.info('Dictionary with the following keys:')
            for key in data.keys():
                value = data[key]
                if isinstance(value, np.ndarray):
                    logger.info(f'  {key}: numpy.ndarray, shape={value.shape}, dtype={value.dtype}')
                    if len(value) > 0:
                        logger.info(f'    Sample (first 5): {value[:5]}')
                elif isinstance(value, list):
                    logger.info(f'  {key}: list, length={len(value)}')
                    if len(value) > 0:
                        logger.info(f'    Sample (first 5): {value[:5]}')
                elif isinstance(value, dict):
                    logger.info(f'  {key}: dict, {len(value)} items')
                    if len(value) > 0:
                        items = list(value.items())[:5]
                        logger.info(f'    Sample (first 5): {dict(items)}')
                else:
                    logger.info(f'  {key}: {type(value).__name__}')
                logger.info('-' * 30)
        else:
            logger.info(f'Type: {type(data).__name__}')
            if isinstance(data, np.ndarray):
                logger.info(f'Shape: {data.shape}')
                logger.info(f'Dtype: {data.dtype}')
                if len(data) > 0:
                    logger.info(f'Sample (first 5): {data[:5]}')
            elif isinstance(data, list):
                logger.info(f'Length: {len(data)}')
                if len(data) > 0:
                    logger.info(f'Sample (first 5): {data[:5]}')
        
        logger.info('=' * 50)
        
    except Exception as e:
        logger.error(f'Error examining pickle file: {e}')
        raise

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Examine metadata and sample of a pickle file')
    parser.add_argument('file_path', help='Path to the pickle file')
    args = parser.parse_args()
    
    examine_pickle(args.file_path) 