import pickle
import numpy as np

def load_embeddings(pickle_file, numpy_file):
    """
    Load embeddings from pickle and numpy files.

    Args:
        pickle_file (str): Path to the pickle file containing the embeddings dictionary.
        numpy_file (str): Path to the numpy file containing the embeddings array.

    Returns:
        dict: A dictionary containing embeddings, gene names, contexts, and mappings.
        np.ndarray: A numpy array of the embeddings.
    """
    # Load the complete embeddings dictionary
    with open(pickle_file, 'rb') as f:
        embeddings_dict = pickle.load(f)

    # Load just the numpy array
    embeddings_array = np.load(numpy_file)

    return embeddings_dict, embeddings_array

def explore_embeddings(embeddings_dict):
    """
    Explore the contents of the embeddings dictionary.

    Args:
        embeddings_dict (dict): The dictionary containing embeddings and metadata.
    """
    # Access different components
    embeddings = embeddings_dict['embeddings']
    gene_names = embeddings_dict['gene_names']
    contexts = embeddings_dict['contexts']

    print("Number of embeddings:", len(embeddings))
    print("Sample gene names:", gene_names[:5])
    print("Sample contexts:", contexts[:5])

# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    pickle_file = 'embeddings_YYYYMMDD_HHMMSS.pkl'
    numpy_file = 'embeddings_array_YYYYMMDD_HHMMSS.npy'

    embeddings_dict, embeddings_array = load_embeddings(pickle_file, numpy_file)
    explore_embeddings(embeddings_dict) 