'''
PROJECT CONFIGURATION FILE
This file contains the configuration variables for the project. The variables are used 
in the other scripts to define the paths to the data and results directories. The variables 
are also used to set the random seed for reproducibility.
'''

# Import libraries
import os
from pathlib import Path
from dotenv import load_dotenv

# Check if on O2 or not
home_variable = os.getenv('HOME')
on_o2 = (home_variable == "/home/kel331")
on_kempner = (home_variable == "/n/home13/anoori")

# Define base project directory based on whether on O2 or not
if on_o2:
    PROJECT_DIR = Path('/n/data1/hms/dbmi/zitnik/lab/users/kel331/spatialGNN/hgt')
elif on_kempner:
    PROJECT_DIR = Path('/n/holylabs/LABS/mzitnik_lab/Users/kel331/spatialGNN/hgt')
else:
    PROJECT_DIR = Path('/Users/an583/Documents/Zitnik_Lab/NeuroKG/neuroKG')

# Define project configuration variables
DATA_DIR = PROJECT_DIR / 'Data'
RESULTS_DIR = PROJECT_DIR / 'Results'
SEED = 42

# Load secrets from the .env file
secrets_file = PROJECT_DIR / 'secrets.env'
load_dotenv(dotenv_path=secrets_file)

# Create the secrets dictionary
secrets = {
    'AZURE_OPENAI_ENDPOINT': os.getenv('AZURE_OPENAI_ENDPOINT'),
    'AZURE_OPENAI_API_KEY': os.getenv('AZURE_OPENAI_API_KEY')
}

# Define model code directories
GALAXY_DIR = PROJECT_DIR / 'Code' / 'GALAXY'
CIPHER_DIR = PROJECT_DIR / 'Code' / 'CIPHER'

# Define data directories for DrugKG
DRUGKG_DIR = DATA_DIR / 'DrugKG' / '2_harmonize_KG'
DRUKG_SPLIT_DIR = DRUGKG_DIR / 'disease_splits' / 'split_edges'

# Define data directories for NeuroKG
NEUROKG_DIR = DATA_DIR / 'NeuroKG' / '3_harmonize_KG'

# CZ CELLxGENE Census dataset variables
CELLXGENE_DATASET = Path('/n/data1/hms/dbmi/zitnik/lab/datasets/2023-05-CELLxGENE')
CELLXGENE_DIR = DATA_DIR / 'CELLxGENE'