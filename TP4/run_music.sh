#!/bin/bash
#SBATCH --job-name=music-lstm
#SBATCH --account=csc_8607
#SBATCH --partition=gpu
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate deeplearning

echo "Job $SLURM_JOB_ID sur $SLURM_NODELIST"
nvidia-smi

python -u train_music.py \
    --action train \
    --rnn-type lstm \
    --hidden-size 1024 \
    --subset-size 50000 \
    --epochs 30 \
    --batch-size 64 \
    --workers 2 \
    --max-grad-norm 1.0 \
    --generate-after \
    --output-length 300
