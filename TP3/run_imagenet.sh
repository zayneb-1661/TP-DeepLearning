#!/bin/bash
#SBATCH --job-name=alexnet-full
#SBATCH --account=csc_8607
#SBATCH --partition=gpu
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/alexnet_full_%j.out

source ~/miniforge3/etc/profile.d/conda.sh
conda activate deeplearning

python -u train_imagenet.py \
    --model alexnet \
    --subset 0 \
    --epochs 50 \
    --batch-size 64 \
    --workers 8 \
    --patience 5

