# https://github.com/Woolverine94/biniou
# txt2vid_ms.py
import gradio as gr
import os
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from compel import Compel
import torch
import time
import random
import shutil
from ressources.scheduler import *
from ressources.common import *
import tomesd

device_label_txt2vid_ms, model_arch = detect_device()
device_txt2vid_ms = torch.device(device_label_txt2vid_ms)

# Gestion des modèles
model_path_txt2vid_ms = "./models/Modelscope/"
os.makedirs(model_path_txt2vid_ms, exist_ok=True)

model_list_txt2vid_ms = [
    "cerspense/zeroscope_v2_576w",
    "camenduru/potat1",
    "damo-vilab/text-to-video-ms-1.7b",
]

# Bouton Cancel
stop_txt2vid_ms = False

def initiate_stop_txt2vid_ms() :
    global stop_txt2vid_ms
    stop_txt2vid_ms = True

def check_txt2vid_ms(step, timestep, latents) : 
    global stop_txt2vid_ms
    if stop_txt2vid_ms == False :
        return
    elif stop_txt2vid_ms == True :
        print(">>>[Modelscope 📼 ]: generation canceled by user")
        stop_txt2vid_ms = False
        try:
            del ressources.txt2vid_ms.pipe_txt2vid_ms
        except NameError as e:
            raise Exception("Interrupting ...")
    return

@metrics_decoration
def video_txt2vid_ms(
    modelid_txt2vid_ms, 
    sampler_txt2vid_ms, 
    prompt_txt2vid_ms, 
    negative_prompt_txt2vid_ms, 
    num_frames_txt2vid_ms, 
    num_prompt_txt2vid_ms, 
    guidance_scale_txt2vid_ms, 
    num_inference_step_txt2vid_ms, 
    height_txt2vid_ms, 
    width_txt2vid_ms, 
    seed_txt2vid_ms, 
    use_gfpgan_txt2vid_ms,
#    tkme_txt2vid_ms,
    progress_txt2vid_ms=gr.Progress(track_tqdm=True)
    ):

    print(">>>[Modelscope 📼 ]: starting module")

    pipe_txt2vid_ms = DiffusionPipeline.from_pretrained(
        modelid_txt2vid_ms, 
        cache_dir=model_path_txt2vid_ms, 
        torch_dtype=model_arch, 
        resume_download=True,
        local_files_only=True if offline_test() else None         
    )
    pipe_txt2vid_ms = get_scheduler(pipe=pipe_txt2vid_ms, scheduler=sampler_txt2vid_ms)
    pipe_txt2vid_ms.unet.enable_forward_chunking(chunk_size=1, dim=1)
    if device_label_txt2vid_ms == "cuda" :
        pipe_txt2vid_ms.enable_sequential_cpu_offload()
    else : 
        pipe_txt2vid_ms = pipe_txt2vid_ms.to(device_txt2vid_ms)
    pipe_txt2vid_ms.enable_vae_slicing()
    
    if seed_txt2vid_ms == 0:
        random_seed = random.randrange(0, 10000000000, 1)
        final_seed = random_seed
    else:
        final_seed = seed_txt2vid_ms
    generator = []
    for k in range(num_prompt_txt2vid_ms):
        generator.append(torch.Generator(device_txt2vid_ms).manual_seed(final_seed + k))
        
    prompt_txt2vid_ms = str(prompt_txt2vid_ms)
    negative_prompt_txt2vid_ms = str(negative_prompt_txt2vid_ms)
    if prompt_txt2vid_ms == "None":
        prompt_txt2vid_ms= ""
    if negative_prompt_txt2vid_ms == "None":
        negative_prompt_txt2vid_ms = ""

    compel = Compel(tokenizer=pipe_txt2vid_ms.tokenizer, text_encoder=pipe_txt2vid_ms.text_encoder, truncate_long_prompts=False)
    conditioning = compel.build_conditioning_tensor(prompt_txt2vid_ms)
    neg_conditioning = compel.build_conditioning_tensor(negative_prompt_txt2vid_ms)
    [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])    

    final_seed = []
    for i in range (num_prompt_txt2vid_ms):
        video_frames = pipe_txt2vid_ms(
            prompt_embeds=conditioning,            
            negative_prompt_embeds=neg_conditioning,
            height=height_txt2vid_ms,
            width=width_txt2vid_ms,
            num_inference_steps=num_inference_step_txt2vid_ms,
            guidance_scale=guidance_scale_txt2vid_ms,
            num_frames=num_frames_txt2vid_ms,
            generator = generator[i],
            callback=check_txt2vid_ms, 
        ).frames

        video_path = export_to_video(video_frames)
        timestamp = time.time()
        seed_id = random_seed + i if (seed_txt2vid_ms == 0) else seed_txt2vid_ms + i
        savename = f"outputs/{seed_id}_{timestamp}.mp4"
        shutil.move(video_path, savename)
        final_seed.append(seed_id)
                    
    print(f">>>[Modelscope 📼 ]: generated {num_prompt_txt2vid_ms} batch(es) of 1")
    reporting_txt2vid_ms = f">>>[Modelscope 📼 ]: "+\
        f"Settings : Model={modelid_txt2vid_ms} | "+\
        f"Sampler={sampler_txt2vid_ms} | "+\
        f"Steps={num_inference_step_txt2vid_ms} | "+\
        f"CFG scale={guidance_scale_txt2vid_ms} | "+\
        f"Video length={num_frames_txt2vid_ms} frames | "+\
        f"Size={width_txt2vid_ms}x{height_txt2vid_ms} | "+\
        f"GFPGAN={use_gfpgan_txt2vid_ms} | "+\
        f"Prompt={prompt_txt2vid_ms} | "+\
        f"Negative prompt={negative_prompt_txt2vid_ms} |"+\
        f"Seed List="+ ', '.join([f"{final_seed[m]}" for m in range(len(final_seed))])
    print(reporting_txt2vid_ms) 

    del pipe_txt2vid_ms, generator, video_frames
    clean_ram()        

    print(f">>>[Modelscope 📼 ]: leaving module")
    return savename
