# https://github.com/Woolverine94/biniou
# vid2vid_ze.py
import gradio as gr
import os
import PIL
import torch
import imageio
import ffmpeg
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor
import time
import random
from ressources.scheduler import *
from ressources.common import *
from ressources.gfpgan import *
import tomesd

device_vid2vid_ze = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Gestion des modèles -> pas concerné (safetensors refusé)
model_path_vid2vid_ze = "./models/pix2pix/"
model_path_safety_checker = "./models/Stable_Diffusion/"
os.makedirs(model_path_vid2vid_ze, exist_ok=True)

model_list_vid2vid_ze = []

for filename in os.listdir(model_path_vid2vid_ze):
    f = os.path.join(model_path_vid2vid_ze, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
        model_list_vid2vid_ze.append(f)

model_list_vid2vid_ze_builtin = [
    "timbrooks/instruct-pix2pix",
]

for k in range(len(model_list_vid2vid_ze_builtin)):
    model_list_vid2vid_ze.append(model_list_vid2vid_ze_builtin[k])

# Bouton Cancel
stop_vid2vid_ze = False

def initiate_stop_vid2vid_ze() :
    global stop_vid2vid_ze
    stop_vid2vid_ze = True

def check_vid2vid_ze(step, timestep, latents) : 
    global stop_vid2vid_ze
    if stop_vid2vid_ze == False :
        return
    elif stop_vid2vid_ze == True :
        stop_vid2vid_ze = False
        try:
            del ressources.vid2vid_ze.pipe_vid2vid_ze
        except NameError as e:
            raise Exception("Interrupting ...")
    return

def image_vid2vid_ze(
    modelid_vid2vid_ze,
    sampler_vid2vid_ze,
    vid_vid2vid_ze,
    prompt_vid2vid_ze,
    negative_prompt_vid2vid_ze,
    num_images_per_prompt_vid2vid_ze,
    num_prompt_vid2vid_ze,
    guidance_scale_vid2vid_ze,
    image_guidance_scale_vid2vid_ze,
    num_inference_step_vid2vid_ze,
    height_vid2vid_ze,
    width_vid2vid_ze,
    seed_vid2vid_ze,
    num_frames_vid2vid_ze,
    num_fps_txt2vid_ze,
    use_gfpgan_vid2vid_ze,
    nsfw_filter,
    tkme_vid2vid_ze,
    progress_vid2vid_ze=gr.Progress(track_tqdm=True)
    ):

    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_safety_checker, device_vid2vid_ze, nsfw_filter)
    
    probe = ffmpeg.probe(vid_vid2vid_ze)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    num_frames_total = int(video_info['nb_frames'])
    if num_frames_total<num_frames_vid2vid_ze :
        num_frames_vid2vid_ze = num_frames_total
    reader = imageio.get_reader(vid_vid2vid_ze, "ffmpeg")
    video = [Image.fromarray(reader.get_data(i)) for i in range(num_frames_vid2vid_ze)]
    
    pipe_vid2vid_ze= StableDiffusionInstructPix2PixPipeline.from_pretrained(
        modelid_vid2vid_ze, 
        cache_dir=model_path_vid2vid_ze, 
        torch_dtype=torch.float32, 
        use_safetensors=True, 
        safety_checker=nsfw_filter_final, 
        feature_extractor=feat_ex,
        resume_download=True,
        local_files_only=True if offline_test() else None
    )
    
    pipe_vid2vid_ze = get_scheduler(pipe=pipe_vid2vid_ze, scheduler=sampler_vid2vid_ze)
    pipe_vid2vid_ze = pipe_vid2vid_ze.to(device_vid2vid_ze)
    pipe_vid2vid_ze.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=3))
    tomesd.apply_patch(pipe_vid2vid_ze, ratio=tkme_vid2vid_ze)
    
    if seed_vid2vid_ze == 0:
        random_seed = torch.randint(0, 10000000000, (1,))
        generator = torch.manual_seed(random_seed)
    else:
        generator = torch.manual_seed(seed_vid2vid_ze)
        
    prompt_vid2vid_ze = str(prompt_vid2vid_ze)
    negative_prompt_vid2vid_ze = str(negative_prompt_vid2vid_ze)
    if prompt_vid2vid_ze == "None":
        prompt_vid2vid_ze = ""
    if negative_prompt_vid2vid_ze == "None":
        negative_prompt_vid2vid_ze = ""

    final_image = []
    
    for i in range (num_prompt_vid2vid_ze):
        image = pipe_vid2vid_ze(
            image=video,
            prompt=[prompt_vid2vid_ze] * len(video),
            negative_prompt=[negative_prompt_vid2vid_ze] * len(video),
            num_images_per_prompt=num_images_per_prompt_vid2vid_ze,
            guidance_scale=guidance_scale_vid2vid_ze,
            image_guidance_scale=image_guidance_scale_vid2vid_ze,
            num_inference_steps=num_inference_step_vid2vid_ze,
            generator = generator,
            callback = check_vid2vid_ze,             
        ).images

        for j in range(len(image)):
            if use_gfpgan_vid2vid_ze == True :
                image[j] = image_gfpgan_mini(image[j])             
            final_image.append(image[j])
            
    timestamp = time.time()
    savename = f"outputs/{timestamp}.mp4"
    final_video = imageio.mimsave(savename, final_image, fps=num_fps_txt2vid_ze)            
            
    del nsfw_filter_final, feat_ex, pipe_vid2vid_ze, generator, image
    clean_ram()            

    return savename, savename
