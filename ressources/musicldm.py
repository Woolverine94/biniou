# https://github.com/Woolverine94/biniou
# musicldm.py
import os
import gradio as gr
from diffusers import MusicLDMPipeline
import torch
import scipy
import random
from ressources.common import *
from ressources.scheduler import *

device_label_musicldm, model_arch = detect_device()
device_musicldm = torch.device(device_label_musicldm)

model_path_musicldm = "./models/MusicLDM/"
os.makedirs(model_path_musicldm, exist_ok=True)

model_list_musicldm = [
    "ucsd-reach/musicldm",
]

# Bouton Cancel
stop_musicldm = False

def initiate_stop_musicldm() :
    global stop_musicldm
    stop_musicldm = True

def check_musicldm(step, timestep, latents) : 
    global stop_musicldm
    if stop_musicldm == False :
        return
    elif stop_musicldm == True :
        print(">>>[MusicLDM 🎶 ]: generation canceled by user")
        stop_musicldm = False
        try:
            del ressources.musicldm.pipe_musicldm
        except NameError as e:
            raise Exception("Interrupting ...")
    return

@metrics_decoration
def music_musicldm(
    modelid_musicldm, 
    sampler_musicldm, 
    prompt_musicldm, 
    negative_prompt_musicldm, 
    num_audio_per_prompt_musicldm, 
    num_prompt_musicldm, 
    guidance_scale_musicldm, 
    num_inference_step_musicldm, 
    audio_length_musicldm,
    seed_musicldm,    
    progress_musicldm=gr.Progress(track_tqdm=True)
    ):

    print(">>>[MusicLDM 🎶 ]: starting module")

    pipe_musicldm = MusicLDMPipeline.from_pretrained(
        modelid_musicldm, 
        cache_dir=model_path_musicldm, 
        torch_dtype=model_arch,
        use_safetensors=True, 
        resume_download=True,
        local_files_only=True if offline_test() else None
    )

    pipe_musicldm = get_scheduler(pipe=pipe_musicldm, scheduler=sampler_musicldm)
    pipe_musicldm.enable_attention_slicing("max")

    if device_label_musicldm == "cuda" :
        pipe_musicldm.enable_sequential_cpu_offload()
    else : 
        pipe_musicldm = pipe_musicldm.to(device_musicldm)
    pipe_musicldm.enable_vae_slicing()
    
    if seed_musicldm == 0:
        random_seed = random.randrange(0, 10000000000, 1)
        final_seed = random_seed
    else:
        final_seed = seed_musicldm
    generator = []
    for k in range(num_prompt_musicldm):
        generator.append([torch.Generator(device_musicldm).manual_seed(final_seed + (k*num_audio_per_prompt_musicldm) + l ) for l in range(num_audio_per_prompt_musicldm)])
    
    prompt_musicldm = str(prompt_musicldm)
    negative_prompt_musicldm = str(negative_prompt_musicldm)
    if prompt_musicldm == "None":
        prompt_musicldm = ""
    if negative_prompt_musicldm == "None":
        negative_prompt_musicldm = ""

    final_audio=[]
    final_seed = []
    for i in range (num_prompt_musicldm):
        audio = pipe_musicldm(
            prompt=prompt_musicldm, 
            negative_prompt=negative_prompt_musicldm, 
            num_waveforms_per_prompt=num_audio_per_prompt_musicldm, 
            guidance_scale=guidance_scale_musicldm, 
            num_inference_steps=num_inference_step_musicldm, 
            audio_length_in_s=audio_length_musicldm, 
            generator=generator[i], 
            callback=check_musicldm,              
        ).audios
        
        for j in range(len(audio)):
            seed_id = random_seed + i*num_audio_per_prompt_musicldm + j if (seed_musicldm == 0) else seed_musicldm + i*num_audio_per_prompt_musicldm + j
            savename = f"outputs/{seed_id}_{timestamper()}.wav"
            scipy.io.wavfile.write(savename, rate=16000, data=audio[j])
            final_audio.append(savename) 
            final_seed.append(seed_id)

    print(f">>>[MusicLDM 🎶 ]: generated {num_prompt_musicldm} batch(es) of {num_audio_per_prompt_musicldm} audio")
    reporting_musicldm = f">>>[musicldm 🎶 ]: "+\
        f"Settings : Model={modelid_musicldm} | "+\
        f"Sampler={sampler_musicldm} | "+\
        f"Steps={num_inference_step_musicldm} | "+\
        f"CFG scale={guidance_scale_musicldm} | "+\
        f"Duration={audio_length_musicldm} | "+\
        f"Prompt={prompt_musicldm} | "+\
        f"Negative prompt={negative_prompt_musicldm} | "+\
        f"Seed List="+ ', '.join([f"{final_seed[m]}" for m in range(len(final_seed))])

    print(reporting_musicldm)
            
    del pipe_musicldm
    clean_ram()      

    print(f">>>[MusicLDM 🎶 ]: leaving module")
    return savename
