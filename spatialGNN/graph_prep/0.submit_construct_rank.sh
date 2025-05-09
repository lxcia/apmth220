#!/bin/bash

#SBATCH -c 16
#SBATCH -t 14-00:00
#SBATCH -p long
#SBATCH  -o logs/%x.out
#SBATCH --mem 240G
#SBATCH -J 0.ranking.nohighmem_14days
 
module load gcc/9.2.0
module load cuda/11.7

conda init bash
source ~/.bashrc
conda activate pinnacle

python 0.constructPPI.py -rank True -annotation "orig_cell_type" -rank_pval_filename "../data/networks/ranked_CELLxGENE"  -celltype_ppi_filename "../data/networks/ppi_CELLxGENE"