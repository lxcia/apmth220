#!/bin/bash

#SBATCH -c 16
#SBATCH -t 24:00
#SBATCH -p highmem
#SBATCH  -o logs/%x.out
#SBATCH --mem 740G
#SBATCH -J convert
 
module load gcc/9.2.0
module load cuda/11.7

conda init bash
source ~/.bashrc
conda activate sc2

# python 0.constructPPI.py -rank True -annotation "cell_type" -rank_pval_filename "../data/networks/ranked_test"  -celltype_ppi_filename "../data/networks/ppi_test"
python convert_to_geneid.py