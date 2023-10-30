# https://github.com/Woolverine94/biniou
# txt2img_sd.py
import gradio as gr
import os
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from compel import Compel, ReturnedEmbeddingsType
import torch
import time
import random
from ressources.scheduler import *
from ressources.gfpgan import *
import tomesd

device_txt2img_sd = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Gestion des modèles
model_path_txt2img_sd = "./models/Stable_Diffusion/"
os.makedirs(model_path_txt2img_sd, exist_ok=True)
model_list_txt2img_sd = []

for filename in os.listdir(model_path_txt2img_sd):
    f = os.path.join(model_path_txt2img_sd, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
        model_list_txt2img_sd.append(f)

model_list_txt2img_sd_builtin = [
    "SG161222/Realistic_Vision_V3.0_VAE",
    "segmind/SSD-1B",
#    "ckpt/anything-v4.5-vae-swapped",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "runwayml/stable-diffusion-v1-5",
    "nitrosocke/Ghibli-Diffusion", 
]

for k in range(len(model_list_txt2img_sd_builtin)):
    model_list_txt2img_sd.append(model_list_txt2img_sd_builtin[k])

# Bouton Cancel
stop_txt2img_sd = False

def initiate_stop_txt2img_sd() :
    global stop_txt2img_sd
    stop_txt2img_sd = True

def check_txt2img_sd(step, timestep, latents) :
    global stop_txt2img_sd
    if stop_txt2img_sd == False :
#        result_preview = preview_image(step, timestep, latents, pipe_txt2img_sd)
        return
    elif stop_txt2img_sd == True :
        stop_txt2img_sd = False
        try:
            del ressources.txt2img_sd.pipe_txt2img_sd
        except NameError as e:
            raise Exception("Interrupting ...")
    return

def image_txt2img_sd(modelid_txt2img_sd, 
    sampler_txt2img_sd, 
    prompt_txt2img_sd, 
    negative_prompt_txt2img_sd, 
    num_images_per_prompt_txt2img_sd, 
    num_prompt_txt2img_sd, 
    guidance_scale_txt2img_sd, 
    num_inference_step_txt2img_sd, 
    height_txt2img_sd, 
    width_txt2img_sd, 
    seed_txt2img_sd, 
    use_gfpgan_txt2img_sd, 
    nsfw_filter, 
    tkme_txt2img_sd,
    progress_txt2img_sd=gr.Progress(track_tqdm=True)
    ):
    
    global pipe_txt2img_sd
    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_txt2img_sd, device_txt2img_sd, nsfw_filter)

    if ('xl' or 'XL' or 'Xl' or 'xL') in modelid_txt2img_sd or (modelid_txt2img_sd == "segmind/SSD-1B") :
#    if ('xl' or 'XL' or 'Xl' or 'xL') in modelid_txt2img_sd :
        is_xl_txt2img_sd: bool = True
    else :        
        is_xl_txt2img_sd: bool = False
        
    if (is_xl_txt2img_sd == True) :
        if modelid_txt2img_sd[0:9] == "./models/" :
            pipe_txt2img_sd = StableDiffusionXLPipeline.from_single_file(
                modelid_txt2img_sd, 
                torch_dtype=torch.float32, 
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
            )
        else :        
            pipe_txt2img_sd = StableDiffusionXLPipeline.from_pretrained(
                modelid_txt2img_sd, 
                cache_dir=model_path_txt2img_sd, 
                torch_dtype=torch.float32, 
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
    else :
        if modelid_txt2img_sd[0:9] == "./models/" :
            pipe_txt2img_sd = StableDiffusionPipeline.from_single_file(
                modelid_txt2img_sd, 
                torch_dtype=torch.float32, 
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
            )
        else :        
            pipe_txt2img_sd = StableDiffusionPipeline.from_pretrained(
                modelid_txt2img_sd, 
                cache_dir=model_path_txt2img_sd, 
                torch_dtype=torch.float32, 
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )

    pipe_txt2img_sd = get_scheduler(pipe=pipe_txt2img_sd, scheduler=sampler_txt2img_sd)
    pipe_txt2img_sd = pipe_txt2img_sd.to(device_txt2img_sd)
    pipe_txt2img_sd.enable_attention_slicing("max")
    tomesd.apply_patch(pipe_txt2img_sd, ratio=tkme_txt2img_sd)
   
    if seed_txt2img_sd == 0:
        random_seed = torch.randint(0, 10000000000, (1,))
        generator = torch.manual_seed(random_seed)
    else:
        generator = torch.manual_seed(seed_txt2img_sd)

    prompt_txt2img_sd = str(prompt_txt2img_sd)
    negative_prompt_txt2img_sd = str(negative_prompt_txt2img_sd)
    if prompt_txt2img_sd == "None":
        prompt_txt2img_sd = ""
    if negative_prompt_txt2img_sd == "None":
        negative_prompt_txt2img_sd = ""

    if (is_xl_txt2img_sd == True) :
        compel = Compel(
            tokenizer=[pipe_txt2img_sd.tokenizer, pipe_txt2img_sd.tokenizer_2], 
            text_encoder=[pipe_txt2img_sd.text_encoder, pipe_txt2img_sd.text_encoder_2], 
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, 
            requires_pooled=[False, True], 
        )
        conditioning, pooled = compel(prompt_txt2img_sd)
        neg_conditioning, neg_pooled = compel(negative_prompt_txt2img_sd)
        [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])
    else :
        compel = Compel(tokenizer=pipe_txt2img_sd.tokenizer, text_encoder=pipe_txt2img_sd.text_encoder, truncate_long_prompts=False)
        conditioning = compel.build_conditioning_tensor(prompt_txt2img_sd)
        neg_conditioning = compel.build_conditioning_tensor(negative_prompt_txt2img_sd)    
        [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])
   
    final_image = []
    for i in range (num_prompt_txt2img_sd):
        if (is_xl_txt2img_sd == True) :
            image = pipe_txt2img_sd(
                prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled, 
                negative_prompt_embeds=neg_conditioning,
                negative_pooled_prompt_embeds=neg_pooled,
                height=height_txt2img_sd,
                width=width_txt2img_sd,
                num_images_per_prompt=num_images_per_prompt_txt2img_sd,
                num_inference_steps=num_inference_step_txt2img_sd,
                guidance_scale=guidance_scale_txt2img_sd,
                generator = generator,
                callback = check_txt2img_sd,
            ).images
        else :
            image = pipe_txt2img_sd(
                prompt_embeds=conditioning,
                negative_prompt_embeds=neg_conditioning,
                height=height_txt2img_sd,
                width=width_txt2img_sd,
                num_images_per_prompt=num_images_per_prompt_txt2img_sd,
                num_inference_steps=num_inference_step_txt2img_sd,
                guidance_scale=guidance_scale_txt2img_sd,
                generator = generator,
                callback = check_txt2img_sd,
            ).images

        for j in range(len(image)):
            timestamp = time.time()
            savename = f"outputs/{timestamp}.png"
            if use_gfpgan_txt2img_sd == True :
                image[j] = image_gfpgan_mini(image[j])
            image[j].save(savename)
            final_image.append(image[j])
    
    del nsfw_filter_final, feat_ex, pipe_txt2img_sd, generator, compel, conditioning, neg_conditioning, image
    clean_ram()
    
    return final_image, final_image
