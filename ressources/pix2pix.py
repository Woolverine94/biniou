# https://github.com/Woolverine94/biniou
# pix2pix.py
import gradio as gr
import os
import PIL
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from compel import Compel
import random
from ressources.common import *
from ressources.gfpgan import *
import tomesd

device_label_pix2pix, model_arch = detect_device()
device_pix2pix = torch.device(device_label_pix2pix)

# Gestion des modèles -> pas concerné (safetensors refusé)
model_path_pix2pix = "./models/pix2pix/"
model_path_safety_checker = "./models/Stable_Diffusion/"
os.makedirs(model_path_pix2pix, exist_ok=True)

model_list_pix2pix = []

for filename in os.listdir(model_path_pix2pix):
    f = os.path.join(model_path_pix2pix, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
        model_list_pix2pix.append(f)

model_list_pix2pix_builtin = [
    "timbrooks/instruct-pix2pix",
    "instruction-tuning-sd/low-level-img-proc",
    "instruction-tuning-sd/cartoonizer",
]

for k in range(len(model_list_pix2pix_builtin)):
    model_list_pix2pix.append(model_list_pix2pix_builtin[k])

# Bouton Cancel
stop_pix2pix = False

def initiate_stop_pix2pix() :
    global stop_pix2pix
    stop_pix2pix = True

def check_pix2pix(pipe, step_index, timestep, callback_kwargs) : 
    global stop_pix2pix
    if stop_pix2pix == False :
        return callback_kwargs
    elif stop_pix2pix == True :
        print(">>>[Instruct pix2pix 🖌️ ]: generation canceled by user")
        stop_pix2pix = False
        try:
            del ressources.pix2pix.pipe_pix2pix
        except NameError as e:
            raise Exception("Interrupting ...")
    return

@metrics_decoration
def image_pix2pix(
    modelid_pix2pix, 
    sampler_pix2pix, 
    img_pix2pix, 
    prompt_pix2pix, 
    negative_prompt_pix2pix, 
    num_images_per_prompt_pix2pix, 
    num_prompt_pix2pix, 
    guidance_scale_pix2pix, 
    image_guidance_scale_pix2pix, 
    num_inference_step_pix2pix, 
    height_pix2pix, 
    width_pix2pix, 
    seed_pix2pix, 
    use_gfpgan_pix2pix, 
    nsfw_filter, 
    tkme_pix2pix,
    progress_pix2pix=gr.Progress(track_tqdm=True)
    ):

    print(">>>[Instruct pix2pix 🖌️ ]: starting module")

    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_safety_checker, device_pix2pix, nsfw_filter)
    
    pipe_pix2pix= StableDiffusionInstructPix2PixPipeline.from_pretrained(
        modelid_pix2pix, 
        cache_dir=model_path_pix2pix, 
        torch_dtype=model_arch,
        use_safetensors=True if (modelid_pix2pix == "timbrooks/instruct-pix2pix") else False,
        safety_checker=nsfw_filter_final, 
        feature_extractor=feat_ex,
        resume_download=True,
        local_files_only=True if offline_test() else None
    )
    
    pipe_pix2pix = schedulerer(pipe_pix2pix, sampler_pix2pix)
    pipe_pix2pix.enable_attention_slicing("max")
    tomesd.apply_patch(pipe_pix2pix, ratio=tkme_pix2pix)
    if device_label_pix2pix == "cuda" :
        pipe_pix2pix.enable_sequential_cpu_offload()
    else : 
        pipe_pix2pix = pipe_pix2pix.to(device_pix2pix)
    
    if seed_pix2pix == 0:
        random_seed = torch.randint(0, 10000000000, (1,))
        generator = torch.manual_seed(random_seed)
    else:
        generator = torch.manual_seed(seed_pix2pix)
        
    dim_size = correct_size(width_pix2pix, height_pix2pix, 512)
    image_input = PIL.Image.open(img_pix2pix)
    image_input = image_input.convert("RGB")
    image_input = image_input.resize((dim_size[0], dim_size[1]))
    
    prompt_pix2pix = str(prompt_pix2pix)
    negative_prompt_pix2pix = str(negative_prompt_pix2pix)
    if prompt_pix2pix == "None":
        prompt_pix2pix = ""
    if negative_prompt_pix2pix == "None":
        negative_prompt_pix2pix = ""

    compel = Compel(tokenizer=pipe_pix2pix.tokenizer, text_encoder=pipe_pix2pix.text_encoder, truncate_long_prompts=False, device=device_pix2pix)
    conditioning = compel.build_conditioning_tensor(prompt_pix2pix)
    neg_conditioning = compel.build_conditioning_tensor(negative_prompt_pix2pix)
    [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])

    final_image = []
    
    for i in range (num_prompt_pix2pix):
        image = pipe_pix2pix(
            image=image_input,
            prompt_embeds=conditioning,
            negative_prompt_embeds=neg_conditioning,
            num_images_per_prompt=num_images_per_prompt_pix2pix,
            guidance_scale=guidance_scale_pix2pix,
            image_guidance_scale=image_guidance_scale_pix2pix,
            num_inference_steps=num_inference_step_pix2pix,
            generator = generator,
            callback_on_step_end=check_pix2pix, 
            callback_on_step_end_tensor_inputs=['latents'],
        ).images

        for j in range(len(image)):
            savename = name_image()
            if use_gfpgan_pix2pix == True :
                image[j] = image_gfpgan_mini(image[j])             
            image[j].save(savename)
            final_image.append(savename)

    print(f">>>[Instruct pix2pix 🖌️ ]: generated {num_prompt_pix2pix} batch(es) of {num_images_per_prompt_pix2pix}")
    reporting_pix2pix = f">>>[Instruct pix2pix 🖌️ ]: "+\
        f"Settings : Model={modelid_pix2pix} | "+\
        f"Sampler={sampler_pix2pix} | "+\
        f"Steps={num_inference_step_pix2pix} | "+\
        f"CFG scale={guidance_scale_pix2pix} | "+\
        f"Img CFG scale={image_guidance_scale_pix2pix} | "+\
        f"Size={width_pix2pix}x{height_pix2pix} | "+\
        f"GFPGAN={use_gfpgan_pix2pix} | "+\
        f"Token merging={tkme_pix2pix} | "+\
        f"nsfw_filter={bool(int(nsfw_filter))} | "+\
        f"Prompt={prompt_pix2pix} | "+\
        f"Negative prompt={negative_prompt_pix2pix}"
    print(reporting_pix2pix) 

    exif_writer_png(reporting_pix2pix, final_image)

    del nsfw_filter_final, feat_ex, pipe_pix2pix, generator, image_input, image
    clean_ram()            

    print(f">>>[Instruct pix2pix 🖌️ ]: leaving module")
    return final_image, final_image

