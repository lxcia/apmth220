#!/bin/bash

#SBATCH -c 16
#SBATCH -t 1-00:00
#SBATCH -p medium
#SBATCH  -o logs/%x.out
#SBATCH --mem 240G
#SBATCH -J test_0.5
 
module load gcc/9.2.0
module load cuda/11.7

conda init bash
source ~/.bashrc
conda activate pinnacle

# python 0.constructPPI.py -rank True -annotation "cell_type" -rank_pval_filename "../data/networks/ranked_test"  -celltype_ppi_filename "../data/networks/ppi_test"
python 0.constructPPI.py -annotation "cell_type" -rank_pval_filename "../data/networks/ranked_test"  -celltype_ppi_filename "../data/networks/ppi_test"