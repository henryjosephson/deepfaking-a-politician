#!/bin/bash
#SBATCH --job-name=transcribe_wavs_to_metadata
#SBATCH --output=./logs/%x/%x_%j.out
#SBATCH --error=./logs/%x/%x_%j.err
#SBATCH --account=pi-wltrimbl
#SBATCH --time=0-00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=henryj@uchicago.edu

ml load cuda
ml load ffmpeg
source ~/.bashrc
conda info --envs | head -5
conda activate /home/henryj/deepfake/venv
conda info --envs | head -5

python /home/henryj/deepfake/training_data/scripts/transcribe_wavs_to_metadata.py
