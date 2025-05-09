#!/bin/bash

#SBATCH -c 16
#SBATCH -t 12:00:00
#SBATCH -p short
#SBATCH  -o logs/%x.out
#SBATCH --mem 240G
#SBATCH -J 5.-cutoff4

module load gcc/9.2.0
module load cuda/11.7

conda init bash
source ~/.bashrc
conda activate pinnacle


python3 5.constructMG.py -celltype_ppi "/n/data1/hms/dbmi/zitnik/lab/users/kel331/virtualcellsim/XL-PINNACLE/data/networks/ppi_CELLxGENE_maxpvalue0,05_maxpval=0.05.csv" \
                         -annotation "orig_cell_type" \
                         -cci_edgelist "/n/data1/hms/dbmi/zitnik/lab/users/kel331/virtualcellsim/XL-PINNACLE/data/networks/cci_edgelist_cellxgene_cutoff4.txt" \
                         -mg_edgelist "/n/data1/hms/dbmi/zitnik/lab/users/kel331/virtualcellsim/XL-PINNACLE/data/networks/mg_edgelist_cellxgene_cutoff4.txt" \
                         -counts "/n/data1/hms/dbmi/zitnik/lab/users/kel331/virtualcellsim/XL-PINNACLE/data/networks/ranked_CELLxGENE.h5ad"
