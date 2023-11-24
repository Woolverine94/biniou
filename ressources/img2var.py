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

device_label_img2var, model_arch = detect_device()
device_img2var = torch.device(device_label_img2var)

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
        print(">>>[Image variation 🖼️ ]: generation canceled by user")
        stop_img2var = False
        try:
            del ressources.img2var.pipe_img2var
        except NameError as e:
            raise Exception("Interrupting ...")
    return

@metrics_decoration
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

    print(">>>[Image variation 🖼️ ]: starting module")

    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_img2var, device_img2var, nsfw_filter)

    pipe_img2var = StableDiffusionImageVariationPipeline.from_pretrained(
        modelid_img2var,
        revision="v2.0", 
        cache_dir=model_path_img2var, 
        torch_dtype=model_arch,
        safety_checker=nsfw_filter_final,
        resume_download=True,
        local_files_only=True if offline_test() else None                
        )

    pipe_img2var = get_scheduler(pipe=pipe_img2var, scheduler=sampler_img2var)
    pipe_img2var.enable_attention_slicing("max")  
    tomesd.apply_patch(pipe_img2var, ratio=tkme_img2var)
    if device_label_img2var == "cuda" :
        pipe_img2var.enable_sequential_cpu_offload()
    else : 
        pipe_img2var = pipe_img2var.to(device_img2var)
    
    if seed_img2var == 0:
        random_seed = random.randrange(0, 10000000000, 1)
        final_seed = random_seed
    else:
        final_seed = seed_img2var
    generator = []
    for k in range(num_prompt_img2var):
        generator.append([torch.Generator(device_img2var).manual_seed(final_seed + (k*num_images_per_prompt_img2var) + l ) for l in range(num_images_per_prompt_img2var)])

    dim_size = correct_size(width_img2var, height_img2var, 512)
    image_input = PIL.Image.open(img_img2var)
    image_input = image_input.convert("RGB")
    image_input = image_input.resize((dim_size[0], dim_size[1]))
    
    final_image = []
    final_seed = []    
    for i in range (num_prompt_img2var):
        image = pipe_img2var(        
            image=image_input,
            num_images_per_prompt=num_images_per_prompt_img2var,
            guidance_scale=guidance_scale_img2var,
            num_inference_steps=num_inference_step_img2var,
            width=dim_size[0],
            height=dim_size[1],             
            generator = generator[i], 
            callback=check_img2var, 
#            callback_on_step_end=check_img2var, 
#            callback_on_step_end_tensor_inputs=['latents'], 
        ).images

        for j in range(len(image)):
            timestamp = time.time()
            seed_id = random_seed + i*num_images_per_prompt_img2var + j if (seed_img2var == 0) else seed_img2var + i*num_images_per_prompt_img2var + j
            savename = f"outputs/{seed_id}_{timestamp}.png"
            if use_gfpgan_img2var == True :
                image[j] = image_gfpgan_mini(image[j])
            image[j].save(savename)
            final_image.append(savename)
            final_seed.append(seed_id)
            
    print(f">>>[Image variation 🖼️ ]: generated {num_prompt_img2var} batch(es) of {num_images_per_prompt_img2var}")
    reporting_img2var = f">>>[Image variation 🖼️ ]: "+\
        f"Settings : Model={modelid_img2var} | "+\
        f"Sampler={sampler_img2var} | "+\
        f"Steps={num_inference_step_img2var} | "+\
        f"CFG scale={guidance_scale_img2var} | "+\
        f"Size={dim_size[0]}x{dim_size[1]} | "+\
        f"GFPGAN={use_gfpgan_img2var} | "+\
        f"Token merging={tkme_img2var} | "+\
        f"nsfw_filter={bool(int(nsfw_filter))} | "+\
        f"Seed List="+ ', '.join([f"{final_seed[m]}" for m in range(len(final_seed))])
    print(reporting_img2var)             

    del nsfw_filter_final, feat_ex, pipe_img2var, generator, image_input, image
    clean_ram()
 
    print(f">>>[Image variation 🖼️ ]: leaving module")  
    return final_image, final_image 
