#!/bin/bash

#SBATCH -c 8
#SBATCH -t 10:00:00
#SBATCH --account=kempner_mzitnik_lab
#SBATCH -p kempner_h100
#SBATCH -o slurm_logs/%x.out
#SBATCH --mem 100G
#SBATCH --gres=gpu:1
#SBATCH -J test_hgt_pretrain_bidirected_10%sparsity_gpu_weights_1_100000

module load gcc/14.2.0-fasrc01
module load cuda/12.0.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01
module load nvhpc/23.7-fasrc01 

conda init bash
source ~/.bashrc
conda activate spatialgnn

python3 pretrain.py \
        --sparsity "10%" \
        --run_name "bidirected_10%sparsity_gpu_weights_1_100000" \
        --run_id "bidirected_10%sparsity_gpu_weights_1_100000" \
        --loss_weight=1 \
        --loss_weight2=100000 \
        --output_dim=64

