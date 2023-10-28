# https://github.com/Woolverine94/biniou
# txt2img_mjm.py
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

device_txt2img_mjm = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Gestion des modèles
model_path_txt2img_mjm = "./models/Midjourney_mini/"
model_path_txt2img_mjm_safetychecker = "./models/Stable_Diffusion/" 
os.makedirs(model_path_txt2img_mjm, exist_ok=True)
model_list_txt2img_mjm = []

for filename in os.listdir(model_path_txt2img_mjm):
    f = os.path.join(model_path_txt2img_mjm, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
        model_list_txt2img_mjm.append(f)

model_list_txt2img_mjm_builtin = [
    "openskyml/midjourney-mini",
]

for k in range(len(model_list_txt2img_mjm_builtin)):
    model_list_txt2img_mjm.append(model_list_txt2img_mjm_builtin[k])

# Bouton Cancel
stop_txt2img_mjm = False

def initiate_stop_txt2img_mjm() :
    global stop_txt2img_mjm
    stop_txt2img_mjm = True

def check_txt2img_mjm(step, timestep, latents) :
    global stop_txt2img_mjm
    if stop_txt2img_mjm == False :
#        result_preview = preview_image(step, timestep, latents, pipe_txt2img_mjm)
        return
    elif stop_txt2img_mjm == True :
        stop_txt2img_mjm = False
        try:
            del ressources.txt2img_mjm.pipe_txt2img_mjm
        except NameError as e:
            raise Exception("Interrupting ...")
    return

def image_txt2img_mjm(
    modelid_txt2img_mjm,
    sampler_txt2img_mjm,
    prompt_txt2img_mjm,
    negative_prompt_txt2img_mjm,
    num_images_per_prompt_txt2img_mjm,
    num_prompt_txt2img_mjm,
    guidance_scale_txt2img_mjm,
    num_inference_step_txt2img_mjm,
    height_txt2img_mjm,
    width_txt2img_mjm,
    seed_txt2img_mjm,
    use_gfpgan_txt2img_mjm,
    nsfw_filter,
    tkme_txt2img_mjm,
    progress_txt2img_mjm=gr.Progress(track_tqdm=True)
    ):
    
    global pipe_txt2img_mjm
    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_txt2img_mjm_safetychecker, device_txt2img_mjm, nsfw_filter)

    if modelid_txt2img_mjm[0:9] == "./models/" :
        pipe_txt2img_mjm = DiffusionPipeline.from_single_file(
            modelid_txt2img_mjm, 
            torch_dtype=torch.float32, 
#            use_safetensors=True, 
            safety_checker=nsfw_filter_final, 
            feature_extractor=feat_ex,
        )
    else :        
        pipe_txt2img_mjm = DiffusionPipeline.from_pretrained(
            modelid_txt2img_mjm, 
            cache_dir=model_path_txt2img_mjm, 
            torch_dtype=torch.float32, 
#            use_safetensors=True, 
            safety_checker=nsfw_filter_final, 
            feature_extractor=feat_ex,
            resume_download=True,
            local_files_only=True if offline_test() else None
        )
    
    pipe_txt2img_mjm = get_scheduler(pipe=pipe_txt2img_mjm, scheduler=sampler_txt2img_mjm)
    pipe_txt2img_mjm = pipe_txt2img_mjm.to(device_txt2img_mjm)
    pipe_txt2img_mjm.enable_attention_slicing("max")
    tomesd.apply_patch(pipe_txt2img_mjm, ratio=tkme_txt2img_mjm)
    
    if seed_txt2img_mjm == 0:
        random_seed = torch.randint(0, 10000000000, (1,))
        generator = torch.manual_seed(random_seed)
    else:
        generator = torch.manual_seed(seed_txt2img_mjm)

    prompt_txt2img_mjm = str(prompt_txt2img_mjm)
    negative_prompt_txt2img_mjm = str(negative_prompt_txt2img_mjm) 
    if prompt_txt2img_mjm == "None":
        prompt_txt2img_mjm = ""
    if negative_prompt_txt2img_mjm == "None":
        negative_prompt_txt2img_mjm = ""
        
    compel = Compel(tokenizer=pipe_txt2img_mjm.tokenizer, text_encoder=pipe_txt2img_mjm.text_encoder, truncate_long_prompts=False)
    conditioning = compel.build_conditioning_tensor(prompt_txt2img_mjm)
    neg_conditioning = compel.build_conditioning_tensor(negative_prompt_txt2img_mjm)    
    [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])   
    
    final_image = []
    for i in range (num_prompt_txt2img_mjm):
        image = pipe_txt2img_mjm(
            prompt_embeds=conditioning,
            negative_prompt_embeds=neg_conditioning,
            height=height_txt2img_mjm,
            width=width_txt2img_mjm,
            num_images_per_prompt=num_images_per_prompt_txt2img_mjm,
            num_inference_steps=num_inference_step_txt2img_mjm,
            guidance_scale=guidance_scale_txt2img_mjm,
            generator = generator,
            callback = check_txt2img_mjm, 
        ).images

        for j in range(len(image)):
            timestamp = time.time()
            savename = f"outputs/{timestamp}.png"
            if use_gfpgan_txt2img_mjm == True :
                image[j] = image_gfpgan_mini(image[j])
            image[j].save(savename)
            final_image.append(image[j])
    
    del nsfw_filter_final, feat_ex, pipe_txt2img_mjm, generator, compel, conditioning, image
    clean_ram()
    
    return final_image, final_image
