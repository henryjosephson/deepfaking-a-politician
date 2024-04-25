#!/usr/local/bin/python3

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

import os
import whisper

print("loading model")

model = whisper.load_model("small.en")
model.to(device)

print("started transcribing")

DATASET_PATH = "/home/henryj/deepfake/training_data/MyTTSDataset/"

for wavfile in os.listdir(path=DATASET_PATH + "wavs/"):
    result = model.transcribe(DATASET_PATH + "wavs/" + wavfile)
    with open(DATASET_PATH + "metadata.csv", 'a') as f:
        f.write(DATASET_PATH + "wavs/" + wavfile + "|" + result["text"] + "\n")
    print("transcribed " + wavfile)

print("done!")
