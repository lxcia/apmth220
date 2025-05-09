#!/bin/bash

#SBATCH -c 16
#SBATCH -t 1-00:00
#SBATCH -p priority
#SBATCH  -o logs/%x.out
#SBATCH --mem 240G
#SBATCH -J test_2
 
module load gcc/9.2.0
module load cuda/11.7

conda init bash
source ~/.bashrc
conda activate pinnacle

python 2.prepCellPhoneDB.py -data "/n/data1/hms/dbmi/zitnik/lab/users/kel331/virtualcellsim/PINNACLE/data/raw/cellxgene_0_999999.h5ad" -output_meta_f "/n/data1/hms/dbmi/zitnik/lab/users/kel331/virtualcellsim/PINNACLE/data/networks/cpdb/"