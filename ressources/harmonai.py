# https://github.com/Woolverine94/biniou
# Harmonai.py
import gradio as gr
import os
from diffusers import DiffusionPipeline
import scipy.io.wavfile
import time
import torch
import random
from ressources.common import *

device_harmonai = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path_harmonai = "./models/harmonai/"
os.makedirs(model_path_harmonai, exist_ok=True)

model_list_harmonai = []

for filename in os.listdir(model_path_harmonai):
    f = os.path.join(model_path_harmonai, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors') or filename.endswith('.bin')):
        model_list_harmonai.append(f)

model_list_harmonai_builtin = [
    "harmonai/glitch-440k",
    "harmonai/honk-140k",
    "harmonai/jmann-small-190k",
    "harmonai/jmann-large-580k",
    "harmonai/maestro-150k",
    "harmonai/unlocked-250k",
]

for k in range(len(model_list_harmonai_builtin)):
    model_list_harmonai.append(model_list_harmonai_builtin[k])

@metrics_decoration
def music_harmonai(
    length_harmonai, 
    model_harmonai, 
    steps_harmonai, 
    seed_harmonai, 
    batch_size_harmonai, 
    batch_repeat_harmonai, 
    progress_harmonai=gr.Progress(track_tqdm=True)
    ):

    print(">>>[Harmonai 🔊 ]: starting module")

    if model_harmonai[0:9] == "./models/" :
        pipe_harmonai = DiffusionPipeline.from_single_file(model_harmonai, torch_dtype=torch.float32)
    else : 
        pipe_harmonai = DiffusionPipeline.from_pretrained(
            model_harmonai, 
            cache_dir=model_path_harmonai, 
            torch_dtype=torch.float32, 
            resume_download=True,
            local_files_only=True if offline_test() else None
            )
    pipe_harmonai = pipe_harmonai.to(device_harmonai)

    if seed_harmonai == 0:
        random_seed = torch.randint(0, 10000000000, (1,))
        generator = torch.manual_seed(random_seed)
    else:
        generator = torch.manual_seed(seed_harmonai)

    for i in range (batch_repeat_harmonai):
        audios = pipe_harmonai(
            audio_length_in_s=length_harmonai,
            num_inference_steps=steps_harmonai,
            generator=generator,
            batch_size=batch_size_harmonai,
        ).audios

        for j, audio in enumerate(audios):
            timestamp = time.time()
            savename = f"outputs/{timestamp}_{j}.wav"
            scipy.io.wavfile.write(savename, pipe_harmonai.unet.config.sample_rate, audio.transpose())

    print(f">>>[Harmonai 🔊 ]: generated {batch_repeat_harmonai} batch(es) of {batch_size_harmonai}")
    reporting_harmonai = f">>>[Harmonai 🔊 ]: "+\
        f"Settings : Model={model_harmonai} | "+\
        f"Steps={steps_harmonai} | "+\
        f"Duration={length_harmonai} sec. | "# +\
#        f"Seed List="+ ', '.join([f"{final_seed[m]}" for m in range(len(final_seed))])
    print(reporting_harmonai) 
    
    del pipe_harmonai, generator, audios
    clean_ram()

    print(f">>>[Harmonai 🔊 ]: leaving module")
    return savename
