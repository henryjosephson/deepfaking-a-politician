#!/bin/bash
#SBATCH --job-name=transcribe_wavs_to_metadata
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --account=pi-wltrimbl
#SBATCH --0:05:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=henryj@uchicago.edu

module load python
module load pytorch
module load anaconda
module load ffmpeg

pip install -U openai-whisper

python ./training_data/scripts/transcribe_wavs_to_metadata.py
