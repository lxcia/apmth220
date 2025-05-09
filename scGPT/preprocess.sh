#!/bin/bash
#SBATCH -c 8
#SBATCH --time=8:00:00
#SBATCH --mem=240G
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --output=./preprocess_logs/preprocess-others-pancreas-9-49-%j.log
#SBATCH --error=./preprocess_logs/preprocess-others-pancreas-9-49-%j.err

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
conda activate preprocess

# Verify environment
echo "After activation, CONDA_PREFIX: $CONDA_PREFIX"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Set up environment
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Set memory management settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

# Create logs directory if it doesn't exist
mkdir -p ./preprocess_logs

# Run the preprocessing script
/n/data1/hms/dbmi/zitnik/lab/users/lucia1215/conda_envs/preprocess/bin/python preprocess_data.py 