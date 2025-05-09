import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the embeddings file
embeddings_data = np.load('embeddings_mouse.npy', allow_pickle=True)
print(f'Data type: {type(embeddings_data)}')
print(f'Shape: {embeddings_data.shape if hasattr(embeddings_data, "shape") else "No shape"}')
print(f'First few elements: {embeddings_data[:5] if hasattr(embeddings_data, "__getitem__") else "Not indexable"}')