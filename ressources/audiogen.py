# https://github.com/Woolverine94/biniou
# Audiogen.py
import os
import gradio as gr
import torch
import torchaudio
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
import time
import random
from ressources.common import *

device_audiogen = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path_audiogen = "./models/Audiocraft/"
os.makedirs(model_path_audiogen, exist_ok=True)

modellist_audiogen = [
    "facebook/audiogen-medium",  
]

# Bouton Cancel
stop_audiogen = False

def initiate_stop_audiogen() :
    global stop_audiogen
    stop_audiogen = True

def check_audiogen(generated_tokens, total_tokens) : 
    global stop_audiogen
    if stop_audiogen == False :
        return
    elif stop_audiogen == True :
        stop_audiogen = False
        try:
            del ressources.audiogen.pipe_audiogen
        except NameError as e:
            raise Exception("Interrupting ...")
    return

@metrics_decoration
def music_audiogen(
    prompt_audiogen, 
    model_audiogen, 
    duration_audiogen, 
    num_batch_audiogen, 
    temperature_audiogen, 
    top_k_audiogen, 
    top_p_audiogen, 
    use_sampling_audiogen, 
    cfg_coef_audiogen, 
    progress_audiogen=gr.Progress(track_tqdm=True)
    ):

    pipe_audiogen = AudioGen.get_pretrained(model_audiogen, device=device_audiogen)
    pipe_audiogen.set_generation_params(
        duration=duration_audiogen, 
        use_sampling=use_sampling_audiogen, 
        temperature=temperature_audiogen, 
        top_k=top_k_audiogen, 
        top_p=top_p_audiogen, 
        cfg_coef=cfg_coef_audiogen
    )
    
    pipe_audiogen.set_custom_progress_callback(check_audiogen)
    prompt_audiogen_final = [f"{prompt_audiogen}"] 
    
    for i in range (num_batch_audiogen):
        wav = pipe_audiogen.generate(prompt_audiogen_final, progress=True)
        for idx, one_wav in enumerate(wav):
            timestamp = time.time()
            savename = f"outputs/{timestamp}_{idx}"
            savename_final = savename+ ".wav" 
            audio_write(savename, one_wav.cpu(), pipe_audiogen.sample_rate, strategy="loudness", loudness_compressor=True)

    del pipe_audiogen
    clean_ram()
            
    return savename_final
