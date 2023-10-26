# https://github.com/Woolverine94/biniou
# txt2img_lcm.py
import gradio as gr
import os
from diffusers import DiffusionPipeline
from compel import Compel, ReturnedEmbeddingsType
import torch
import time
import random
from ressources.scheduler import *
from ressources.gfpgan import *
import tomesd

device_txt2img_lcm = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Gestion des modèles
model_path_txt2img_lcm = "./models/LCM/"
model_path_txt2img_lcm_safetychecker = "./models/Stable_Diffusion/" 
os.makedirs(model_path_txt2img_lcm, exist_ok=True)
model_list_txt2img_lcm = []

for filename in os.listdir(model_path_txt2img_lcm):
    f = os.path.join(model_path_txt2img_lcm, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
        model_list_txt2img_lcm.append(f)

model_list_txt2img_lcm_builtin = [
    "SimianLuo/LCM_Dreamshaper_v7",
]

for k in range(len(model_list_txt2img_lcm_builtin)):
    model_list_txt2img_lcm.append(model_list_txt2img_lcm_builtin[k])

scheduler_list_txt2img_lcm = [
    "LCMScheduler",
]

# Bouton Cancel
stop_txt2img_lcm = False

def initiate_stop_txt2img_lcm() :
    global stop_txt2img_lcm
    stop_txt2img_lcm = True

def check_txt2img_lcm(step, timestep, latents) :
    global stop_txt2img_lcm
    if stop_txt2img_lcm == False :
#        result_preview = preview_image(step, timestep, latents, pipe_txt2img_lcm)
        return
    elif stop_txt2img_lcm == True :
        stop_txt2img_lcm = False
        try:
            del ressources.txt2img_lcm.pipe_txt2img_lcm
        except NameError as e:
            raise Exception("Interrupting ...")
    return

def image_txt2img_lcm(modelid_txt2img_lcm, 
    sampler_txt2img_lcm, 
    prompt_txt2img_lcm, 
    num_images_per_prompt_txt2img_lcm, 
    num_prompt_txt2img_lcm, 
    guidance_scale_txt2img_lcm, 
    lcm_origin_steps_txt2img_lcm, 
    num_inference_step_txt2img_lcm, 
    height_txt2img_lcm, 
    width_txt2img_lcm, 
    seed_txt2img_lcm, 
    use_gfpgan_txt2img_lcm, 
    nsfw_filter, 
    tkme_txt2img_lcm,
    progress_txt2img_lcm=gr.Progress(track_tqdm=True)
    ):
    
    global pipe_txt2img_lcm
    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_txt2img_lcm_safetychecker, device_txt2img_lcm, nsfw_filter)

    if modelid_txt2img_lcm[0:9] == "./models/" :
        pipe_txt2img_lcm = DiffusionPipeline.from_single_file(
            modelid_txt2img_lcm, 
            torch_dtype=torch.float32, 
            custom_pipeline="latent_consistency_txt2img", 
            custom_revision="main", 
#            revision="fb9c5d",
            use_safetensors=True, 
            safety_checker=nsfw_filter_final, 
            feature_extractor=feat_ex,
        )
    else :        
        pipe_txt2img_lcm = DiffusionPipeline.from_pretrained(
            modelid_txt2img_lcm, 
            cache_dir=model_path_txt2img_lcm, 
            torch_dtype=torch.float32, 
            custom_pipeline="latent_consistency_txt2img", 
            custom_revision="main", 
#            revision="fb9c5d", 
            use_safetensors=True, 
            safety_checker=nsfw_filter_final, 
            feature_extractor=feat_ex,
            resume_download=True,
            local_files_only=True if offline_test() else None
        )
    
#    pipe_txt2img_lcm = get_scheduler(pipe=pipe_txt2img_lcm, scheduler=sampler_txt2img_lcm)
    pipe_txt2img_lcm = pipe_txt2img_lcm.to(device_txt2img_lcm)
    pipe_txt2img_lcm.enable_attention_slicing("max")
    tomesd.apply_patch(pipe_txt2img_lcm, ratio=tkme_txt2img_lcm)
    
    if seed_txt2img_lcm == 0:
        random_seed = torch.randint(0, 10000000000, (1,))
        generator = torch.manual_seed(random_seed)
    else:
        generator = torch.manual_seed(seed_txt2img_lcm)

    prompt_txt2img_lcm = str(prompt_txt2img_lcm)
    if prompt_txt2img_lcm == "None":
        prompt_txt2img_lcm = ""

    compel = Compel(tokenizer=pipe_txt2img_lcm.tokenizer, text_encoder=pipe_txt2img_lcm.text_encoder, truncate_long_prompts=False)
    conditioning = compel.build_conditioning_tensor(prompt_txt2img_lcm)
   
    final_image = []
    for i in range (num_prompt_txt2img_lcm):
        image = pipe_txt2img_lcm(
            prompt_embeds=conditioning,
            height=height_txt2img_lcm,
            width=width_txt2img_lcm,
            num_images_per_prompt=num_images_per_prompt_txt2img_lcm,
            num_inference_steps=num_inference_step_txt2img_lcm,
            guidance_scale=guidance_scale_txt2img_lcm,
            lcm_origin_steps=lcm_origin_steps_txt2img_lcm,
        ).images

        for j in range(len(image)):
            timestamp = time.time()
            savename = f"outputs/{timestamp}.png"
            if use_gfpgan_txt2img_lcm == True :
                image[j] = image_gfpgan_mini(image[j])
            image[j].save(savename)
            final_image.append(image[j])
    
    del nsfw_filter_final, feat_ex, pipe_txt2img_lcm, generator, compel, conditioning, image
    clean_ram()
    
    return final_image, final_image
