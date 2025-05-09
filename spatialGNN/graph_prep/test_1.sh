#!/bin/bash

#SBATCH -c 16
#SBATCH -t 1-00:00
#SBATCH -p highmem
#SBATCH  -o logs/%x.out
#SBATCH --mem 340G
#SBATCH -J test_1
 
module load gcc/9.2.0
module load cuda/11.7

conda init bash
source ~/.bashrc
conda activate pinnacle

python 1.evaluatePPI.py -celltype_ppi "/n/data1/hms/dbmi/zitnik/lab/users/kel331/virtualcellsim/PINNACLE/data/networks/ppi_test_maxpval=1.csv"