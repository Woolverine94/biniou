# https://github.com/Woolverine94/biniou
# img2var.py
import gradio as gr
import os
import PIL
import torch
from diffusers import StableDiffusionImageVariationPipeline
import time
import random
from ressources.scheduler import *
from ressources.common import *
from ressources.gfpgan import *
import tomesd

device_img2var = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Gestion des modèles
model_path_img2var = "./models/Stable_Diffusion/"
os.makedirs(model_path_img2var, exist_ok=True)

model_list_img2var = [
    "lambdalabs/sd-image-variations-diffusers",
]

# Bouton Cancel
stop_img2var = False

def initiate_stop_img2var() :
    global stop_img2var
    stop_img2var = True

def check_img2var(step, timestep, latents) : 
    global stop_img2var
    if stop_img2var == False :
        return
    elif stop_img2var == True :
        stop_img2var = False
        try:
            del ressources.img2var.pipe_img2var
        except NameError as e:
            raise Exception("Interrupting ...")
    return

def image_img2var(
    modelid_img2var, 
    sampler_img2var, 
    img_img2var, 
    num_images_per_prompt_img2var, 
    num_prompt_img2var, 
    guidance_scale_img2var, 
    num_inference_step_img2var, 
    height_img2var, 
    width_img2var, 
    seed_img2var, 
    use_gfpgan_img2var, 
    nsfw_filter, 
    tkme_img2var,    
    progress_img2var=gr.Progress(track_tqdm=True)
    ):

    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_img2var, device_img2var, nsfw_filter)

    pipe_img2var = StableDiffusionImageVariationPipeline.from_pretrained(
        modelid_img2var,
        revision="v2.0", 
        cache_dir=model_path_img2var, 
        torch_dtype=torch.float32, 
        safety_checker=nsfw_filter_final,
        resume_download=True,
        local_files_only=True if offline_test() else None                
        )

    pipe_img2var = get_scheduler(pipe=pipe_img2var, scheduler=sampler_img2var)
    pipe_img2var = pipe_img2var.to(device_img2var)
    pipe_img2var.enable_attention_slicing("max")  
    tomesd.apply_patch(pipe_img2var, ratio=tkme_img2var)
    
    if seed_img2var == 0:
        random_seed = torch.randint(0, 10000000000, (1,))
        generator = torch.manual_seed(random_seed)
    else:
        generator = torch.manual_seed(seed_img2var)

    dim_size = correct_size(width_img2var, height_img2var, 512)
    image_input = PIL.Image.open(img_img2var)
    image_input = image_input.convert("RGB")
    image_input = image_input.resize((dim_size[0], dim_size[1]))
    
    final_image = []
    
    for i in range (num_prompt_img2var):
        image = pipe_img2var(        
            image=image_input,
            num_images_per_prompt=num_images_per_prompt_img2var,
            guidance_scale=guidance_scale_img2var,
            num_inference_steps=num_inference_step_img2var,
            width=dim_size[0],
            height=dim_size[1],             
            generator = generator,
            callback = check_img2var,                
        ).images

        for j in range(len(image)):
            timestamp = time.time()
            savename = f"outputs/{timestamp}.png"
            if use_gfpgan_img2var == True :
                image[j] = image_gfpgan_mini(image[j])             
            image[j].save(savename)
            final_image.append(image[j])

    del nsfw_filter_final, feat_ex, pipe_img2var, generator, image_input, image
    clean_ram()
   
    return final_image, final_image   

