#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --output=/home/henryj/deepfake/logs/%x/%x_%j.out
#SBATCH --error=/home/henryj/deepfake/logs/%x/%x_%j.err
#SBATCH --account=pi-wltrimbl
#SBATCH --time=0-00:10:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=henryj@rcc.uchicago.edu

ml load cuda/12.2
ml load ffmpeg
source ~/.bashrc
conda info --envs | head -5
conda activate /home/henryj/deepfake/venv
conda info --envs | head -5

python /home/henryj/deepfake/training_data/scripts/inference.py
