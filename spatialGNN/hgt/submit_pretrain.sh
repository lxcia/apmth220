#!/bin/bash

#SBATCH -c 4
#SBATCH -t 10:00:00
#SBATCH -p priority
#SBATCH  -o slurm_logs/%x.out
#SBATCH --mem 10G
#SBATCH -J test_hgt_pretrain_cpu
# #SBATCH --gres=gpu:1

module load gcc/9.2.0
module load cuda/11.7

conda init bash
source ~/.bashrc
conda activate spatialgnn

python3 pretrain.py \
        --sparsity "10%" \
        --run_name "kevintest_12.1_10%sparsity"

