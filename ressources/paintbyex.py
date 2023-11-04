# https://github.com/Woolverine94/biniou
# paintbyex.py
import gradio as gr
import os
import PIL
import torch
from diffusers import PaintByExamplePipeline
import time
import random
from ressources.scheduler import *
from ressources.common import *
from ressources.gfpgan import *
import tomesd

device_paintbyex = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Gestion des modèles
model_path_paintbyex = "./models/Paint_by_example/"
model_path_safety_checker = "./models/Stable_Diffusion/"
os.makedirs(model_path_paintbyex, exist_ok=True)
os.makedirs(model_path_safety_checker, exist_ok=True)
model_list_paintbyex = []

for filename in os.listdir(model_path_paintbyex):
    f = os.path.join(model_path_paintbyex, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
        model_list_paintbyex.append(f)

model_list_paintbyex_builtin = [
    "Fantasy-Studio/Paint-by-Example",
]

for k in range(len(model_list_paintbyex_builtin)):
    model_list_paintbyex.append(model_list_paintbyex_builtin[k])

# Bouton Cancel
stop_paintbyex = False

def initiate_stop_paintbyex() :
    global stop_paintbyex
    stop_paintbyex = True

def check_paintbyex(step, timestep, latents) : 
    global stop_paintbyex
    if stop_paintbyex == False :
        return
    elif stop_paintbyex == True :
        stop_paintbyex = False
        try:
            del ressources.paintbyex.pipe_paintbyex
        except NameError as e:
            raise Exception("Interrupting ...")
    return

def image_paintbyex(
    modelid_paintbyex, 
    sampler_paintbyex, 
    img_paintbyex, 
    rotation_img_paintbyex, 
    example_img_paintbyex, 
    num_images_per_prompt_paintbyex, 
    num_prompt_paintbyex, 
    guidance_scale_paintbyex,
    num_inference_step_paintbyex, 
    height_paintbyex, 
    width_paintbyex, 
    seed_paintbyex, 
    use_gfpgan_paintbyex, 
    nsfw_filter, 
    tkme_paintbyex,
    progress_paintbyex=gr.Progress(track_tqdm=True)
    ):
    
    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_safety_checker, device_paintbyex, nsfw_filter)
    
    if modelid_paintbyex[0:9] == "./models/" :
        pipe_paintbyex = PaintByExamplePipeline.from_single_file(
            modelid_paintbyex, 
            torch_dtype=torch.float32, 
#            use_safetensors=True, 
            safety_checker=nsfw_filter_final, 
#            feature_extractor=feat_ex, 
        )
    else :        
        pipe_paintbyex = PaintByExamplePipeline.from_pretrained(
            modelid_paintbyex, 
            cache_dir=model_path_paintbyex, 
            torch_dtype=torch.float32, 
#            use_safetensors=True, 
            safety_checker=nsfw_filter_final, 
#            feature_extractor=feat_ex, 
            resume_download=True,
            local_files_only=True if offline_test() else None
        )

    pipe_paintbyex = get_scheduler(pipe=pipe_paintbyex, scheduler=sampler_paintbyex)
    pipe_paintbyex = pipe_paintbyex.to(device_paintbyex)
    pipe_paintbyex.enable_attention_slicing("max")
    tomesd.apply_patch(pipe_paintbyex, ratio=tkme_paintbyex)
    
    if seed_paintbyex == 0:
        random_seed = random.randrange(0, 10000000000, 1)
        final_seed = random_seed
    else:
        final_seed = seed_paintbyex
    generator = []
    for k in range(num_prompt_paintbyex):
        generator.append([torch.Generator(device_paintbyex).manual_seed(final_seed + (k*num_images_per_prompt_paintbyex) + l ) for l in range(num_images_per_prompt_paintbyex)])

    angle_paintbyex = 360 - rotation_img_paintbyex   
    img_paintbyex["image"] = img_paintbyex["image"].rotate(angle_paintbyex, expand=True)
    dim_size = correct_size(width_paintbyex, height_paintbyex, 512)
    image_input = img_paintbyex["image"].convert("RGB")
    mask_image_input = img_paintbyex["mask"].convert("RGB")
    example_image_input = example_img_paintbyex.convert("RGB")    
    image_input = image_input.resize((dim_size[0],dim_size[1]))
    mask_image_input = mask_image_input.resize((dim_size[0],dim_size[1]))    
    savename_mask = f"outputs/mask.png"
    mask_image_input.save(savename_mask) 
   
    final_image = []
    
    for i in range (num_prompt_paintbyex):
        image = pipe_paintbyex(
            image=image_input,
            mask_image=mask_image_input, 
            example_image=example_image_input,
            num_images_per_prompt=num_images_per_prompt_paintbyex,
            guidance_scale=guidance_scale_paintbyex,
            width=dim_size[0],
            height=dim_size[1],
            num_inference_steps=num_inference_step_paintbyex,
            generator = generator[i],
            callback = check_paintbyex,              
        ).images

        for j in range(len(image)):
            timestamp = time.time()
            seed_id = random_seed + i*num_images_per_prompt_paintbyex + j if (seed_paintbyex == 0) else seed_paintbyex + i*num_images_per_prompt_paintbyex + j
            savename = f"outputs/{seed_id}_{timestamp}.png"
            if use_gfpgan_paintbyex == True :
                image[j] = image_gfpgan_mini(image[j])
            image[j].save(savename)
            final_image.append(savename)

    final_image.append(savename_mask)

    del nsfw_filter_final, feat_ex, pipe_paintbyex, generator, image_input, mask_image_input, example_image_input, image
    clean_ram()

    return final_image, final_image
