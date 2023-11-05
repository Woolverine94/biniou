# https://github.com/Woolverine94/biniou
# txt2vid_ze.py
import gradio as gr
import os
import imageio
from diffusers import TextToVideoZeroPipeline
import numpy as np
import torch
import time
import random
from ressources.scheduler import *
from ressources.common import *
from ressources.gfpgan import *
import tomesd

device_txt2vid_ze = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path_txt2vid_ze = "./models/Stable_Diffusion/"
os.makedirs(model_path_txt2vid_ze, exist_ok=True)

model_list_txt2vid_ze = [
    "SG161222/Realistic_Vision_V3.0_VAE",
#    "ckpt/anything-v4.5-vae-swapped",
    "runwayml/stable-diffusion-v1-5",
    "nitrosocke/Ghibli-Diffusion",
]

# Bouton Cancel
stop_txt2vid_ze = False

def initiate_stop_txt2vid_ze() :
    global stop_txt2vid_ze
    stop_txt2vid_ze = True

def check_txt2vid_ze(step, timestep, latents) : 
    global stop_txt2vid_ze
    if stop_txt2vid_ze == False :
        return
    elif stop_txt2vid_ze == True :
        stop_txt2vid_ze = False
        try:
            del ressources.txt2vid_ze.pipe_txt2vid_ze
        except NameError as e:
            raise Exception("Interrupting ...")
    return

@metrics_decoration
def video_txt2vid_ze(
    modelid_txt2vid_ze, 
    num_inference_step_txt2vid_ze, 
    sampler_txt2vid_ze, 
    guidance_scale_txt2vid_ze, 
    seed_txt2vid_ze, 
    num_frames_txt2vid_ze, 
    num_fps_txt2vid_ze, 
    height_txt2vid_ze, 
    width_txt2vid_ze, 
    num_videos_per_prompt_txt2vid_ze, 
    num_prompt_txt2vid_ze, 
    motion_field_strength_x_txt2vid_ze, 
    motion_field_strength_y_txt2vid_ze, 
    timestep_t0_txt2vid_ze :int, 
    timestep_t1_txt2vid_ze :int, 
    prompt_txt2vid_ze, 
    negative_prompt_txt2vid_ze, 
    nsfw_filter, 
    num_chunks_txt2vid_ze :int, 
    use_gfpgan_txt2vid_ze,
    tkme_txt2vid_ze,
    progress_txt2vid_ze=gr.Progress(track_tqdm=True)
    ):
    
    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_txt2vid_ze, device_txt2vid_ze, nsfw_filter)

    pipe_txt2vid_ze = TextToVideoZeroPipeline.from_pretrained(
        modelid_txt2vid_ze, 
        cache_dir=model_path_txt2vid_ze, 
        torch_dtype=torch.float32, 
        use_safetensors=True, 
        safety_checker=nsfw_filter_final, 
        feature_extractor=feat_ex, 
        resume_download=True,
        local_files_only=True if offline_test() else None        
    )
    
    pipe_txt2vid_ze = get_scheduler(pipe=pipe_txt2vid_ze, scheduler=sampler_txt2vid_ze)
    pipe_txt2vid_ze = pipe_txt2vid_ze.to(device_txt2vid_ze)
    tomesd.apply_patch(pipe_txt2vid_ze, ratio=tkme_txt2vid_ze)
    
    if seed_txt2vid_ze == 0:
        random_seed = torch.randint(0, 10000000000, (1,))
        generator = torch.manual_seed(random_seed)
    else:
        generator = torch.manual_seed(seed_txt2vid_ze)

    for j in range (num_prompt_txt2vid_ze):
        if num_chunks_txt2vid_ze != 1 :
            result = []
            chunk_ids = np.arange(0, num_frames_txt2vid_ze, num_chunks_txt2vid_ze)
            generator = torch.Generator(device=device_txt2vid_ze)
            for i in range(len(chunk_ids)):
                print(f"Processing chunk {i + 1} / {len(chunk_ids)}")
                ch_start = chunk_ids[i]
                ch_end = num_frames_txt2vid_ze if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
                if i == 0 :
                    frame_ids = [0] + list(range(ch_start, ch_end))
                else :
                    frame_ids = [ch_start -1] + list(range(ch_start, ch_end))
                generator = generator.manual_seed(seed_txt2vid_ze)
                output = pipe_txt2vid_ze(
                    prompt=prompt_txt2vid_ze,
                    negative_prompt=negative_prompt_txt2vid_ze,
                    height=height_txt2vid_ze,
                    width=width_txt2vid_ze,
                    num_inference_steps=num_inference_step_txt2vid_ze,
                    guidance_scale=guidance_scale_txt2vid_ze,
                    frame_ids=frame_ids,
                    video_length=len(frame_ids), 
                    num_videos_per_prompt=num_videos_per_prompt_txt2vid_ze,
                    motion_field_strength_x=motion_field_strength_x_txt2vid_ze,
                    motion_field_strength_y=motion_field_strength_y_txt2vid_ze,
                    t0=timestep_t0_txt2vid_ze,
                    t1=timestep_t1_txt2vid_ze,
                    generator = generator,
                    callback = check_txt2vid_ze,
                )
                result.append(output.images[1:])
            result = np.concatenate(result)
        else :
            result = pipe_txt2vid_ze(
                prompt=prompt_txt2vid_ze,
                negative_prompt=negative_prompt_txt2vid_ze,
                height=height_txt2vid_ze,
                width=width_txt2vid_ze,
                num_inference_steps=num_inference_step_txt2vid_ze,
                guidance_scale=guidance_scale_txt2vid_ze,
                video_length=num_frames_txt2vid_ze,
                num_videos_per_prompt=num_videos_per_prompt_txt2vid_ze,
                motion_field_strength_x=motion_field_strength_x_txt2vid_ze,
                motion_field_strength_y=motion_field_strength_y_txt2vid_ze,
                t0=timestep_t0_txt2vid_ze,
                t1=timestep_t1_txt2vid_ze,
                generator = generator,
                callback = check_txt2vid_ze,
            ).images

        result = [(r * 255).astype("uint8") for r in result]

        for j in range(len(result)):
            if use_gfpgan_txt2vid_ze == True :
                result[j] = image_gfpgan_mini(result[j])

        timestamp = time.time()
        savename = f"outputs/{timestamp}.mp4"
        imageio.mimsave(savename, result, fps=num_fps_txt2vid_ze)
        
    del nsfw_filter_final, feat_ex, pipe_txt2vid_ze, generator, result
    clean_ram()
    
    return savename

