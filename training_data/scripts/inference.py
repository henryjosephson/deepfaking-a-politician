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
OUTPUT_WAV_PATH = "/home/henryj/deepfake/outputs/xtts-ft"

print("Loading model...")
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=XTTS_CHECKPOINT, vocab_path=TOKENIZER_PATH, use_deepspeed=False)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[SPEAKER_REFERENCE])

ottos_sentences = [
    "Failure doesn’t mean you are a failure, it just means you haven’t succeeded yet.",
    "We will compare this recording against the audio you uploaded in the previous step to verify it’s your voice.",
    "A liquidity trap is caused when people hold cash because they expect an adverse event such as deflation, insufficient aggregate demand, or war.",
    "Stirner suggested that communism was tainted with the same idealism as Christianity and infused with superstitious ideas like morality and justice.",
    "Malicious users could download deepfake software on their personal computers and avoid any degree of oversight.",
    "With tenure, Suzie would have all the more leisure for yachting, but her publications are no good."
    "Are those shy Eurasian footwear, cowboy chaps, or jolly earthmoving headgear?",
    "The beige hue on the waters of the loch impressed all, including the French queen, before she heard that symphony again, just as young Arthur wanted.",
    "Shaw, those twelve beige hooks are joined if I patch a young, gooey mouth.",
    "Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.",
]

print("Inference...")

for s in ottos_sentences:
    out = model.inference(
        s,
        "en",
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.7, # Add custom parameters here
    )
    torchaudio.save(OUTPUT_WAV_PATH + s.split(' ')[0] + '.wav', torch.tensor(out["wav"]).unsqueeze(0), 24000)
