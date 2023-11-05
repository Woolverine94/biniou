# https://github.com/Woolverine94/biniou
# inpaint.py
import gradio as gr
import os
import PIL
import torch
from diffusers import StableDiffusionInpaintPipeline
from compel import Compel
import time
import random
from ressources.scheduler import *
from ressources.common import *
from ressources.gfpgan import *
import tomesd

device_inpaint = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Gestion des modèles
model_path_inpaint = "./models/inpaint/"
model_path_safety_checker = "./models/Stable_Diffusion/"
os.makedirs(model_path_inpaint, exist_ok=True)
os.makedirs(model_path_safety_checker, exist_ok=True)
model_list_inpaint = []

for filename in os.listdir(model_path_inpaint):
    f = os.path.join(model_path_inpaint, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
        model_list_inpaint.append(f)

model_list_inpaint_builtin = [
    "Uminosachi/realisticVisionV30_v30VAE-inpainting",
    "runwayml/stable-diffusion-inpainting",
]

for k in range(len(model_list_inpaint_builtin)):
    model_list_inpaint.append(model_list_inpaint_builtin[k])

# Bouton Cancel
stop_inpaint = False

def initiate_stop_inpaint() :
    global stop_inpaint
    stop_inpaint = True

def check_inpaint(step, timestep, latents) : 
    global stop_inpaint
    if stop_inpaint == False :
        return
    elif stop_inpaint == True :
        stop_inpaint = False
        try:
            del ressources.inpaint.pipe_inpaint
        except NameError as e:
            raise Exception("Interrupting ...")
    return

@metrics_decoration
def image_inpaint(
    modelid_inpaint, 
    sampler_inpaint, 
    img_inpaint, 
    rotation_img_inpaint, 
    prompt_inpaint, 
    negative_prompt_inpaint, 
    num_images_per_prompt_inpaint, 
    num_prompt_inpaint, 
    guidance_scale_inpaint,
    denoising_strength_inpaint, 
    num_inference_step_inpaint, 
    height_inpaint, 
    width_inpaint, 
    seed_inpaint, 
    use_gfpgan_inpaint, 
    nsfw_filter, 
    tkme_inpaint,
    progress_inpaint=gr.Progress(track_tqdm=True)
    ):
    
    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_safety_checker, device_inpaint, nsfw_filter)

    if modelid_inpaint[0:9] == "./models/" :
        pipe_inpaint = StableDiffusionInpaintPipeline.from_single_file(
            modelid_inpaint, 
            torch_dtype=torch.float32, 
            use_safetensors=True, 
            safety_checker=nsfw_filter_final, 
            feature_extractor=feat_ex
        )
    else :        
        pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            modelid_inpaint, 
            cache_dir=model_path_inpaint, 
            torch_dtype=torch.float32, 
            use_safetensors=True, 
            safety_checker=nsfw_filter_final, 
            feature_extractor=feat_ex,
            resume_download=True,
            local_files_only=True if offline_test() else None
        )

    pipe_inpaint = get_scheduler(pipe=pipe_inpaint, scheduler=sampler_inpaint)
    pipe_inpaint = pipe_inpaint.to(device_inpaint)
    pipe_inpaint.enable_attention_slicing("max")
    tomesd.apply_patch(pipe_inpaint, ratio=tkme_inpaint)
    
    if seed_inpaint == 0:
        random_seed = random.randrange(0, 10000000000, 1)
        final_seed = random_seed
    else:
        final_seed = seed_inpaint
    generator = []
    for k in range(num_prompt_inpaint):
        generator.append([torch.Generator(device_inpaint).manual_seed(final_seed + (k*num_images_per_prompt_inpaint) + l ) for l in range(num_images_per_prompt_inpaint)])

    angle_inpaint = 360 - rotation_img_inpaint   
    img_inpaint["image"] = img_inpaint["image"].rotate(angle_inpaint, expand=True)
    dim_size = correct_size(width_inpaint, height_inpaint, 512)
    image_input = img_inpaint["image"].convert("RGB")
    mask_image_input = img_inpaint["mask"].convert("RGB")
    image_input = image_input.resize((dim_size[0],dim_size[1]))
    mask_image_input = mask_image_input.resize((dim_size[0],dim_size[1]))    
    savename_mask = f"outputs/mask.png"
    mask_image_input.save(savename_mask)
    
    prompt_inpaint = str(prompt_inpaint)
    negative_prompt_inpaint = str(negative_prompt_inpaint)
    if prompt_inpaint == "None":
        prompt_inpaint = ""
    if negative_prompt_inpaint == "None":
        negative_prompt_inpaint = ""

    compel = Compel(tokenizer=pipe_inpaint.tokenizer, text_encoder=pipe_inpaint.text_encoder, truncate_long_prompts=False)
    conditioning = compel.build_conditioning_tensor(prompt_inpaint)
    neg_conditioning = compel.build_conditioning_tensor(negative_prompt_inpaint)
    [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])
    
    final_image = []
    
    for i in range (num_prompt_inpaint):
        image = pipe_inpaint(
            image=image_input,
            mask_image=mask_image_input,            
            prompt_embeds=conditioning,
            negative_prompt_embeds=neg_conditioning,
            num_images_per_prompt=num_images_per_prompt_inpaint,
            guidance_scale=guidance_scale_inpaint,
            strength=denoising_strength_inpaint,
            width=dim_size[0],
            height=dim_size[1],
            num_inference_steps=num_inference_step_inpaint,
            generator = generator[i],
            callback = check_inpaint,              
        ).images

        for j in range(len(image)):
            timestamp = time.time()
            seed_id = random_seed + i*num_images_per_prompt_inpaint + j if (seed_inpaint == 0) else seed_inpaint + i*num_images_per_prompt_inpaint + j
            savename = f"outputs/{seed_id}_{timestamp}.png"
            if use_gfpgan_inpaint == True :
                image[j] = image_gfpgan_mini(image[j])
            image[j].save(savename)
            final_image.append(savename)

    final_image.append(savename_mask)

    del nsfw_filter_final, feat_ex, pipe_inpaint, generator, image_input, mask_image_input, image
    clean_ram()

    return final_image, final_image
