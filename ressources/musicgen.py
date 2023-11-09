# https://github.com/Woolverine94/biniou
# Musicgen.py
import os
import gradio as gr
import torch
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import time
import random
from ressources.common import *

device_musicgen = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path_musicgen = "./models/Audiocraft/"
os.makedirs(model_path_musicgen, exist_ok=True)

modellist_musicgen = [
    "facebook/musicgen-small",
    "facebook/musicgen-medium",
    "facebook/musicgen-large",
    "facebook/musicgen-melody",    
]

# Bouton Cancel
stop_musicgen = False

def initiate_stop_musicgen() :
    global stop_musicgen
    stop_musicgen = True

def check_musicgen(generated_tokens, total_tokens) : 
    global stop_musicgen
    if stop_musicgen == False :
        return
    elif stop_musicgen == True :
        print(">>>[MusicGen 🎶 ]: generation canceled by user")
        stop_musicgen = False
        try:
            del ressources.musicgen.pipe_musicgen
        except NameError as e:
            raise Exception("Interrupting ...")
    return

@metrics_decoration
def music_musicgen(
    prompt_musicgen, 
    model_musicgen, 
    duration_musicgen, 
    num_batch_musicgen, 
    temperature_musicgen, 
    top_k_musicgen, 
    top_p_musicgen, 
    use_sampling_musicgen, 
    cfg_coef_musicgen, 
    progress_musicgen=gr.Progress(track_tqdm=True)
    ):

    print(">>>[MusicGen 🎶 ]: starting module")

    pipe_musicgen = MusicGen.get_pretrained(model_musicgen, device=device_musicgen)
    pipe_musicgen.set_generation_params(
        duration=duration_musicgen, 
        use_sampling=use_sampling_musicgen, 
        temperature=temperature_musicgen, 
        top_k=top_k_musicgen, 
        top_p=top_p_musicgen, 
        cfg_coef=cfg_coef_musicgen
    )
    pipe_musicgen.set_custom_progress_callback(check_musicgen)
    prompt_musicgen_final = [f"{prompt_musicgen}"]
    for i in range (num_batch_musicgen):
        wav = pipe_musicgen.generate(prompt_musicgen_final, progress=True)
        for idx, one_wav in enumerate(wav):
            timestamp = time.time()
            savename = f"outputs/{timestamp}_{idx}"
            savename_final = savename+ ".wav" 
            audio_write(savename, one_wav.cpu(), pipe_musicgen.sample_rate, strategy="loudness", loudness_compressor=True)

    print(f">>>[MusicGen 🎶 ]: generated {num_batch_musicgen} batch(es) of 1")
    reporting_musicgen = f">>>[MusicGen 🎶 ]: "+\
        f"Settings : Model={model_musicgen} | "+\
        f"Duration={duration_musicgen} | "+\
        f"CFG scale={cfg_coef_musicgen} | "+\
        f"Use sampling={use_sampling_musicgen} | "+\
        f"Temperature={temperature_musicgen} | "+\
        f"Top_k={top_k_musicgen} | "+\
        f"Top_p={top_p_musicgen} | "+\
        f"Prompt={prompt_musicgen}"
    print(reporting_musicgen)
            
    del pipe_musicgen
    clean_ram()      

    print(f">>>[MusicGen 🎶 ]: leaving module")
    return savename_final           
