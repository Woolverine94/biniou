# https://github.com/Woolverine94/biniou
# Musicgen.py
import os
import gradio as gr
import torch
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import random
from ressources.common import *

device_label_musicgen_mel, model_arch = detect_device()
device_musicgen_mel = torch.device(device_label_musicgen_mel)

model_path_musicgen_mel = "./models/Audiocraft/"
os.makedirs(model_path_musicgen_mel, exist_ok=True)

modellist_musicgen_mel = [
    "facebook/musicgen-stereo-melody",
    "facebook/musicgen-melody",
    "facebook/musicgen-stereo-melody-large",
    "facebook/musicgen-melody-large",
]

# Bouton Cancel
stop_musicgen_mel = False

def initiate_stop_musicgen_mel() :
    global stop_musicgen_mel
    stop_musicgen_mel = True

def check_musicgen_mel(generated_tokens, total_tokens) : 
    global stop_musicgen_mel
    if stop_musicgen_mel == False :
        return
    elif stop_musicgen_mel == True :
        print(">>>[MusicGen Melody 🎶 ]: generation canceled by user")
        stop_musicgen_mel = False
        try:
            del ressources.musicgen.pipe_musicgen_mel
        except NameError as e:
            raise Exception("Interrupting ...")
    return

@metrics_decoration
def music_musicgen_mel(
    prompt_musicgen_mel, 
    model_musicgen_mel, 
    duration_musicgen_mel, 
    num_batch_musicgen_mel, 
    temperature_musicgen_mel, 
    top_k_musicgen_mel, 
    top_p_musicgen_mel, 
    use_sampling_musicgen_mel, 
    cfg_coef_musicgen_mel, 
    source_audio_musicgen_mel,
    source_type_musicgen_mel,
    progress_musicgen_mel=gr.Progress(track_tqdm=True)
    ):

    print(">>>[MusicGen Melody 🎶 ]: starting module")

    pipe_musicgen_mel = MusicGen.get_pretrained(model_musicgen_mel, device=device_musicgen_mel)
    pipe_musicgen_mel.set_generation_params(
        duration=duration_musicgen_mel, 
        use_sampling=use_sampling_musicgen_mel, 
        temperature=temperature_musicgen_mel, 
        top_k=top_k_musicgen_mel, 
        top_p=top_p_musicgen_mel, 
        cfg_coef=cfg_coef_musicgen_mel
    )
    melody_musicgen_mel, sr_musicgen_mel = torchaudio.load(source_audio_musicgen_mel)
    pipe_musicgen_mel.set_custom_progress_callback(check_musicgen_mel)
    prompt_musicgen_mel_final = [f"{prompt_musicgen_mel}"]
    savename_array= []
    for i in range (num_batch_musicgen_mel):
        wav = pipe_musicgen_mel.generate_with_chroma(
            prompt_musicgen_mel_final, 
            melody_musicgen_mel[None].expand(-1, -1, -1), 
            sr_musicgen_mel, 
            progress=True,
        )
        for idx, one_wav in enumerate(wav):
            savename, savename_final = name_idx_audio(idx)
            audio_write(
                savename, 
                one_wav.cpu(), 
                pipe_musicgen_mel.sample_rate, 
                strategy="loudness", 
                loudness_compressor=True
            )
            savename_array.append(savename_final)

    print(f">>>[MusicGen Melody 🎶 ]: generated {num_batch_musicgen_mel} batch(es) of 1")
    reporting_musicgen_mel = f">>>[MusicGen Melody 🎶 ]: "+\
        f"Settings : Model={model_musicgen_mel} | "+\
        f"Duration={duration_musicgen_mel} | "+\
        f"CFG scale={cfg_coef_musicgen_mel} | "+\
        f"Use sampling={use_sampling_musicgen_mel} | "+\
        f"Temperature={temperature_musicgen_mel} | "+\
        f"Top_k={top_k_musicgen_mel} | "+\
        f"Top_p={top_p_musicgen_mel} | "+\
        f"Source audio type={source_type_musicgen_mel} | "+\
        f"Prompt={prompt_musicgen_mel}"
    print(reporting_musicgen_mel)

    metadata_writer_wav(reporting_musicgen_mel, savename_array)

    del pipe_musicgen_mel
    clean_ram()

    print(f">>>[MusicGen Melody 🎶 ]: leaving module")
    return savename_final
