#!/bin/bash

#SBATCH -c 16
#SBATCH -t 5-00:00
#SBATCH -p highmem
#SBATCH  -o logs/%x.out
#SBATCH --mem 740G
#SBATCH -J 2.prep
 
module load gcc/9.2.0
module load cuda/11.7

conda init bash
source ~/.bashrc
conda activate pinnacle

python 2.prepCellPhoneDB.py -data "/n/data1/hms/dbmi/zitnik/lab/users/kel331/virtualcellsim/XL-PINNACLE/data/preprocessed/downsampled2_adata_normalized_log.h5ad" -output_meta_f "/n/data1/hms/dbmi/zitnik/lab/users/kel331/virtualcellsim/XL-PINNACLE/data/networks/cpdb/"