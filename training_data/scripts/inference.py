import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Add here the xtts_config path
CONFIG_PATH = (
    "/home/henryj/deepfake/training_data/scripts/run/training/"
    + "GPT_XTTS_v2.0_LJSpeech_FT-May-07-2024_10+34AM-49c9af8/config.json"
)
# Add here the vocab file that you have used to train the model
TOKENIZER_PATH = (
    "/home/henryj/deepfake/training_data/scripts/run/training/"
    + "XTTS_v2.0_original_model_files/vocab.json"
)
# Add here the checkpoint that you want to do inference with
XTTS_CHECKPOINT = (
    "/home/henryj/deepfake/training_data/scripts/run/training/"
    + "GPT_XTTS_v2.0_LJSpeech_FT-May-07-2024_10+34AM-49c9af8/"#best_model.pth"
)
# Add here the speaker reference
SPEAKER_REFERENCE = (
    "/home/henryj/deepfake/training_data/AlexBoresVoice0.wav"
)

# output wav path
OUTPUT_WAV_PATH = "/home/henryj/deepfake/outputs/xtts-ft.wav"

print("Loading model...")
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=XTTS_CHECKPOINT, vocab_path=TOKENIZER_PATH, use_deepspeed=False)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[SPEAKER_REFERENCE])

print("Inference...")
out = model.inference(
    "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
    "en",
    gpt_cond_latent,
    speaker_embedding,
    temperature=0.7, # Add custom parameters here
)
torchaudio.save(OUTPUT_WAV_PATH, torch.tensor(out["wav"]).unsqueeze(0), 24000)