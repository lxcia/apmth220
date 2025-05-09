#!/bin/bash

#SBATCH -c 16
#SBATCH -t 1:00:00
#SBATCH -p short
#SBATCH  -o logs/%x.out
#SBATCH --mem 240G
#SBATCH -J 4.runcci.pvalue0.0001

module load gcc/9.2.0
module load cuda/11.7

conda init bash
source ~/.bashrc
conda activate pinnacle


python3 4.constructCCI.py -cpdb_output "/n/data1/hms/dbmi/zitnik/lab/users/kel331/virtualcellsim/XL-PINNACLE/data/networks/cpdb" -cci_edgelist "/n/data1/hms/dbmi/zitnik/lab/users/kel331/virtualcellsim/XL-PINNACLE/data/networks/cci_edgelist_cellxgene_pvalue0.0001.txt" -pvalue 0.0001
