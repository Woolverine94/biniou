# https://github.com/Woolverine94/biniou
# img2img.py
import gradio as gr
import os
import PIL
import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline
from compel import Compel
import time
import random
from ressources.scheduler import *
from ressources.common import *
from ressources.gfpgan import *
import tomesd

device_img2img = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Gestion des modèles
model_path_img2img = "./models/Stable_Diffusion/"
os.makedirs(model_path_img2img, exist_ok=True)

model_list_img2img = []

for filename in os.listdir(model_path_img2img):
    f = os.path.join(model_path_img2img, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
        model_list_img2img.append(f)

model_list_img2img_builtin = [
    "SG161222/Realistic_Vision_V3.0_VAE",
    "ckpt/anything-v4.5-vae-swapped",
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    "runwayml/stable-diffusion-v1-5",
    "nitrosocke/Ghibli-Diffusion",     
]

for k in range(len(model_list_img2img_builtin)):
    model_list_img2img.append(model_list_img2img_builtin[k])

# Bouton Cancel
stop_img2img = False

def initiate_stop_img2img() :
    global stop_img2img
    stop_img2img = True

def check_img2img(step, timestep, latents) : 
    global stop_img2img
    if stop_img2img == False :
        return
    elif stop_img2img == True :
        stop_img2img = False
        try:
            del ressources.img2img.pipe_img2img
        except NameError as e:
            raise Exception("Interrupting ...")
    return

def image_img2img(
    modelid_img2img, 
    sampler_img2img, 
    img_img2img, 
    prompt_img2img, 
    negative_prompt_img2img, 
    num_images_per_prompt_img2img, 
    num_prompt_img2img, 
    guidance_scale_img2img, 
    denoising_strength_img2img, 
    num_inference_step_img2img, 
    height_img2img, 
    width_img2img, 
    seed_img2img, 
    source_type_img2img, 
    use_gfpgan_img2img, 
    nsfw_filter, 
    tkme_img2img,    
    progress_img2img=gr.Progress(track_tqdm=True)
    ):

    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_img2img, device_img2img, nsfw_filter)

    if ('xl' or 'XL' or 'Xl' or 'xL') in modelid_img2img :
        is_xl_img2img: bool = True
    else :        
        is_xl_img2img: bool = False        

    if (is_xl_img2img == True) :
        if modelid_img2img[0:9] == "./models/" :
            pipe_img2img = StableDiffusionXLImg2ImgPipeline.from_single_file(
                modelid_img2img, 
                torch_dtype=torch.float32, 
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
            )
        else :        
            pipe_img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                modelid_img2img, 
                cache_dir=model_path_img2img, 
                torch_dtype=torch.float32, 
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=True if offline_test() else None                
            )
    else :
        if modelid_img2img[0:9] == "./models/" :
            pipe_img2img = StableDiffusionImg2ImgPipeline.from_single_file(
                modelid_img2img, 
                torch_dtype=torch.float32, 
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
            )
        else :        
            pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                modelid_img2img, 
                cache_dir=model_path_img2img, 
                torch_dtype=torch.float32, 
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=True if offline_test() else None                
            )

    pipe_img2img = get_scheduler(pipe=pipe_img2img, scheduler=sampler_img2img)
    pipe_img2img = pipe_img2img.to(device_img2img)
    pipe_img2img.enable_attention_slicing("max")  
    tomesd.apply_patch(pipe_img2img, ratio=tkme_img2img)
    
    if seed_img2img == 0:
        random_seed = torch.randint(0, 10000000000, (1,))
        generator = torch.manual_seed(random_seed)
    else:
        generator = torch.manual_seed(seed_img2img)

    if source_type_img2img == "sketch" :
        dim_size=[512, 512]
    else :        
        dim_size = correct_size(width_img2img, height_img2img, 512)
    image_input = PIL.Image.open(img_img2img)
    image_input = image_input.convert("RGB")
    image_input = image_input.resize((dim_size[0], dim_size[1]))

    prompt_img2img = str(prompt_img2img)
    negative_prompt_img2img = str(negative_prompt_img2img)
    if prompt_img2img == "None":
        prompt_img2img = ""
    if negative_prompt_img2img == "None":
        negative_prompt_img2img = ""

    if (is_xl_img2img == False) :
        compel = Compel(tokenizer=pipe_img2img.tokenizer, text_encoder=pipe_img2img.text_encoder, truncate_long_prompts=False)
        conditioning = compel.build_conditioning_tensor(prompt_img2img)
        neg_conditioning = compel.build_conditioning_tensor(negative_prompt_img2img)
        [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])
    
    final_image = []
    
    for i in range (num_prompt_img2img):
        if (is_xl_img2img == True) :
            image = pipe_img2img(        
                image=image_input,
                prompt=prompt_img2img,
                negative_prompt=negative_prompt_img2img,            
                num_images_per_prompt=num_images_per_prompt_img2img,
                guidance_scale=guidance_scale_img2img,
                strength=denoising_strength_img2img,
                num_inference_steps=num_inference_step_img2img,
                generator = generator,
                callback = check_img2img,                
            ).images
        else : 
            image = pipe_img2img(        
                image=image_input,
                prompt_embeds=conditioning,
                negative_prompt_embeds=neg_conditioning,
                num_images_per_prompt=num_images_per_prompt_img2img,
                guidance_scale=guidance_scale_img2img,
                strength=denoising_strength_img2img,
                num_inference_steps=num_inference_step_img2img,
                generator = generator,
                callback = check_img2img,                
            ).images        

        for j in range(len(image)):
            timestamp = time.time()
            savename = f"outputs/{timestamp}.png"
            if use_gfpgan_img2img == True :
                image[j] = image_gfpgan_mini(image[j])             
            image[j].save(savename)
            final_image.append(image[j])
            
    del nsfw_filter_final, feat_ex, pipe_img2img, generator, image_input, image
    clean_ram()
   
    return final_image, final_image
    

