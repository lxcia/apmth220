#!/bin/bash

#SBATCH -c 16
#SBATCH -t 4-00:00
#SBATCH -p highmem
#SBATCH  -o logs/%x.out
#SBATCH --mem 740G
#SBATCH -J 3.runcellphonedb_0.01pvalue

module load gcc/9.2.0
module load cuda/11.7

conda init bash
source ~/.bashrc
conda activate cpdb


python3 3.run_cellphonedb.py -counts /n/data1/hms/dbmi/zitnik/lab/users/kel331/virtualcellsim/XL-PINNACLE/data/networks/ranked_CELLxGENE.h5ad
