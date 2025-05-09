import pickle
import os

def explore_mapping_files(data_dir):
    # List of mapping files
    mapping_files = [
        'map_contexts_to_ints.pkl',
        'map_genes_to_ints.pkl', 
        'map_ints_to_contexts.pkl',
        'map_ints_to_genes.pkl'
    ]
    
    for filename in mapping_files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"\nFile not found: {filename}")
            continue
            
        with open(filepath, 'rb') as f:
            mapping = pickle.load(f)
            
        print(f"\nExploring {filename}:")
        print(f"Type: {type(mapping)}")
        print(f"Size: {len(mapping)} items")
        
        # Show first and last 10 items
        print("\nFirst 10 examples:")
        if isinstance(mapping, dict):
            for i, (key, value) in enumerate(list(mapping.items())[:10]):
                print(f"  {key} -> {value}")
        else:
            for i, item in enumerate(list(mapping)[:10]):
                print(f"  {item}")
                
        print("\nLast 10 examples:")
        if isinstance(mapping, dict):
            for i, (key, value) in enumerate(list(mapping.items())[-10:]):
                print(f"  {key} -> {value}")
        else:
            for i, item in enumerate(list(mapping)[-10:]):
                print(f"  {item}")
                
        # Show data types of keys and values if it's a dictionary
        if isinstance(mapping, dict):
            key_type = type(next(iter(mapping)))
            val_type = type(next(iter(mapping.values())))
            print(f"\nKey type: {key_type}")
            print(f"Value type: {val_type}")

if __name__ == "__main__":
    # Replace this with your actual data directory
    data_dir = "data"  # or the full path to your data directory
    explore_mapping_files(data_dir)