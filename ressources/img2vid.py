# https://github.com/Woolverine94/biniou
# img2vid.py
import gradio as gr
import os
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
import numpy as np
import torch
import random
from ressources.scheduler import *
from ressources.common import *
from ressources.gfpgan import *
import tomesd

device_label_img2vid, model_arch = detect_device()
device_img2vid = torch.device(device_label_img2vid)

# Block to remove :
if os.path.exists("./models/Stable_Diffusion_Video/") :
    os.rename("./models/Stable_Diffusion_Video/", "./models/Stable_Video_Diffusion/")
# End of block

model_path_img2vid = "./models/Stable_Video_Diffusion/"
model_path_safetychecker_img2vid = "./models/Stable_Diffusion/"
os.makedirs(model_path_img2vid, exist_ok=True)

model_list_img2vid = [
    "stabilityai/stable-video-diffusion-img2vid",
    "stabilityai/stable-video-diffusion-img2vid-xt",
]

# Bouton Cancel
stop_img2vid = False

def initiate_stop_img2vid() :
    global stop_img2vid
    stop_img2vid = True

def check_img2vid(pipe, step_index, timestep, callback_kwargs):
    global stop_img2vid
    if stop_img2vid == False :
        return callback_kwargs
    elif stop_img2vid == True :
        print(">>>[Text2Video-Zero 📼 ]: generation canceled by user")
        stop_img2vid = False
        try:
            del ressources.img2vid.pipe_img2vid
        except NameError as e:
            raise Exception("Interrupting ...")
    return

@metrics_decoration
def video_img2vid(
    modelid_img2vid,
    num_inference_steps_img2vid,
    sampler_img2vid,
    min_guidance_scale_img2vid,
    max_guidance_scale_img2vid,
    seed_img2vid,
    num_frames_img2vid,
    num_fps_img2vid,
    decode_chunk_size_img2vid,
    width_img2vid,
    height_img2vid,
    num_prompt_img2vid,
    num_videos_per_prompt_img2vid,
    motion_bucket_id_img2vid,
    noise_aug_strength_img2vid,
    nsfw_filter,
    img_img2vid,
    use_gfpgan_img2vid,
    tkme_img2vid,
    progress_img2vid=gr.Progress(track_tqdm=True)
    ):
    
    print(">>>[Stable Video Diffusion 📼 ]: starting module")

    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_safetychecker_img2vid, device_img2vid, nsfw_filter)

    pipe_img2vid = StableVideoDiffusionPipeline.from_pretrained(
        modelid_img2vid,
        cache_dir=model_path_img2vid,
        torch_dtype=model_arch,
        use_safetensors=True,
        safety_checker=nsfw_filter_final,
        feature_extractor=feat_ex,
        resume_download=True,
        local_files_only=True if offline_test() else None
    )
    
    pipe_img2vid = get_scheduler(pipe=pipe_img2vid, scheduler=sampler_img2vid)
#    tomesd.apply_patch(pipe_img2vid, ratio=tkme_img2vid)
    if device_label_img2vid == "cuda" :
#        pipe_img2vid.enable_sequential_cpu_offload()
        pipe_img2vid.enable_model_cpu_offload()
    else :
        pipe_img2vid = pipe_img2vid.to(device_img2vid)
    
    if seed_img2vid == 0:
        random_seed = random.randrange(0, 10000000000, 1)
        final_seed = random_seed
    else:
        final_seed = seed_img2vid
    generator = []
    for k in range(num_prompt_img2vid):
        generator.append(torch.Generator(device_img2vid).manual_seed(final_seed + k))

    image_input = PIL.Image.open(img_img2vid)
    image_input = image_input.convert("RGB")

    final_seed = []
    for i in range (num_prompt_img2vid):
        result = pipe_img2vid(
            image=image_input,
            height=height_img2vid,
            width=width_img2vid,
            num_inference_steps=num_inference_steps_img2vid,
            min_guidance_scale=min_guidance_scale_img2vid,
            max_guidance_scale=max_guidance_scale_img2vid,
            num_frames=num_frames_img2vid,
            fps=num_fps_img2vid,
            decode_chunk_size=decode_chunk_size_img2vid,
            num_videos_per_prompt=num_videos_per_prompt_img2vid,
            motion_bucket_id=motion_bucket_id_img2vid,
            noise_aug_strength=noise_aug_strength_img2vid,
            generator = generator[i],
            callback_on_step_end=check_img2vid,
            callback_on_step_end_tensor_inputs=['latents'],
        ).frames[0]

        timestamp = time.time()
        seed_id = random_seed + i*num_videos_per_prompt_img2vid if (seed_img2vid == 0) else seed_img2vid + i*num_videos_per_prompt_img2vid
        savename = f"outputs/{seed_id}_{timestamper()}.mp4"
        export_to_video(result, savename, fps=num_fps_img2vid)
        final_seed.append(seed_id)

    print(f">>>[Stable Video Diffusion 📼 ]: generated {num_prompt_img2vid} batch(es) of {num_videos_per_prompt_img2vid}")
    reporting_img2vid = f">>>[Stable Video Diffusion 📼 ]: "+\
        f"Settings : Model={modelid_img2vid} | "+\
        f"Sampler={sampler_img2vid} | "+\
        f"Steps={num_inference_steps_img2vid} | "+\
        f"Min guidance scale={min_guidance_scale_img2vid} | "+\
        f"Max guidance scale={max_guidance_scale_img2vid} | "+\
        f"Video length={num_frames_img2vid} frames | "+\
        f"FPS={num_fps_img2vid} frames | "+\
        f"Chunck size={decode_chunk_size_img2vid} | "+\
        f"Size={width_img2vid}x{height_img2vid} | "+\
        f"Motion bucket id={motion_bucket_id_img2vid} | "+\
        f"Noise strength={noise_aug_strength_img2vid} | "+\
        f"GFPGAN={use_gfpgan_img2vid} | "+\
        f"nsfw_filter={bool(int(nsfw_filter))} | "+\
        f"Seed List="+ ', '.join([f"{final_seed[m]}" for m in range(len(final_seed))])
    print(reporting_img2vid)

    del nsfw_filter_final, feat_ex, pipe_img2vid, generator, result
    clean_ram()

    print(f">>>[Stable Video Diffusion 📼 ]: leaving module")
    return savename
