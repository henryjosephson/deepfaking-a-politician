#!/usr/local/bin/python3

import os
from pydub import AudioSegment
import matplotlib.pyplot as plt

lengths = []

for wavfile in os.listdir(path="training_data/MyTTSDataset/wavs/"):
    sound = AudioSegment.from_wav("training_data/MyTTSDataset/wavs/" + wavfile)
    lengths.append(len(sound))

print(sorted(x / 1000 * 60 for x in lengths)) # lengths in milliseconds

#plt.hist(lengths, bins=40)
#plt.yticks(range(0, max(lengths)), [x / 1000 * 60 for x in range(0, max(lengths))][::50000])
#plt.show()