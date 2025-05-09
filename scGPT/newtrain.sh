#!/bin/bash
#SBATCH -c 8
#SBATCH --time=8:00:00
#SBATCH --mem=240G
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --output=./train_logs/4-12/5-23.log
#SBATCH --error=./train_logs/4-12/5-23.err

# Load CUDA module
module load gcc/6.2.0 cuda/10.2 miniconda3/4.10.3

# Debug information
echo "Current PATH: $PATH"
echo "Current PYTHONPATH: $PYTHONPATH"
echo "Current CONDA_PREFIX: $CONDA_PREFIX"

# Initialize conda
source /n/app/miniconda3/4.10.3/etc/profile.d/conda.sh
conda init bash

# Activate conda environment
conda activate scgpt_train

# Verify environment
echo "After activation, CONDA_PREFIX: $CONDA_PREFIX"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Pip list:"
/n/data1/hms/dbmi/zitnik/lab/users/lucia1215/conda_envs/scgpt_train/bin/pip list | grep torch

# Set up environment
export PYTHONPATH=$PYTHONPATH:$(pwd)
export WANDB_MODE=online  # or offline if you don't want to use wandb

# Set CUDA environment variables
export CUDA_HOME=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Set memory management settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

# Run the training script using the Python from the conda environment
# DEBUG MODE: Running for only 1 epoch
# Original command: /n/data1/hms/dbmi/zitnik/lab/users/lucia1215/conda_envs/scgpt_train/bin/python newtrain.py
/n/data1/hms/dbmi/zitnik/lab/users/lucia1215/conda_envs/scgpt_train/bin/python newtrain.py --epochs 1 --batch_size 16 --chunk_size 25000


