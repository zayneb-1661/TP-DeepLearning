#!/bin/bash
#SBATCH --job-name=tp5-captioning
#SBATCH --account=csc_8607
#SBATCH --partition=gpu
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd ~/TP-DeepLearning/TP5

source ~/miniforge3/etc/profile.d/conda.sh
conda activate deeplearning

echo "Job : $SLURM_JOB_ID"
echo "Noeud : $SLURM_NODELIST"

nvidia-smi

python -u train.py \
    --epochs 15 \
    --batch-size 32 \
    --workers 4 \
    --lr 0.001
