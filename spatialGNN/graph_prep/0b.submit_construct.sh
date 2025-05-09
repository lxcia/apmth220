#!/bin/bash

#SBATCH -c 16
#SBATCH -t 12:00:00
#SBATCH -p short
#SBATCH  -o logs/%x.out
#SBATCH --mem 240G
#SBATCH -J 0.construct.maxpvalue0,05_2
 
module load gcc/9.2.0
module load cuda/11.7

conda init bash
source ~/.bashrc
conda activate pinnacle

python 0.constructPPI.py -annotation "orig_cell_type" -rank_pval_filename "../data/networks/ranked_CELLxGENE"  -celltype_ppi_filename "../data/networks/ppi_CELLxGENE_maxpvalue0,05" -num_cells_cutoff 10 -max_pval 0.05