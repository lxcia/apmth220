import pandas as pd
import sys
import os
import argparse

from cellphonedb.src.core.methods import cpdb_statistical_analysis_method

parser = argparse.ArgumentParser(description="Preparing input files for CellPhoneDB.")
parser.add_argument("-counts", type=str, default="/n/data1/hms/dbmi/zitnik/lab/users/kel331/virtualcellsim/PINNACLE/data/networks/ranked_test.h5ad", help="Data as h5ad file.")
args = parser.parse_args()

cpdb_file_path = '/n/data1/hms/dbmi/zitnik/lab/users/kel331/virtualcellsim/PINNACLE/data/networks/cpdb/v5.0.0/cellphonedb.zip'
meta_file_path = '/n/data1/hms/dbmi/zitnik/lab/users/kel331/virtualcellsim/PINNACLE/data/networks/cpdb/meta_CellPhoneDB.txt'
counts_file_path = args.counts
out_path = '/n/data1/hms/dbmi/zitnik/lab/users/kel331/virtualcellsim/PINNACLE/data/networks/cpdb'

cpdb_results = cpdb_statistical_analysis_method.call(
    cpdb_file_path = cpdb_file_path,                 # mandatory: CellphoneDB database zip file.
    meta_file_path = meta_file_path,                 # mandatory: tsv file defining barcodes to cell label.
    counts_file_path = counts_file_path,             # mandatory: normalized count matrix - a path to the counts file, or an in-memory AnnData object
    counts_data = 'hgnc_symbol',                     # defines the gene annotation in counts matrix.
    output_path = out_path,                          # Path to save results.
    output_suffix = None,                            # Replaces the timestamp in the output files by a user defined string in the  (default: None).
    pvalue = 0.01,
    threshold = 0.1
    )



