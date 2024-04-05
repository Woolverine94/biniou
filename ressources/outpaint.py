# https://github.com/Woolverine94/biniou
# outpaint.py
import gradio as gr
import os
import PIL
import cv2
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline
from compel import Compel, ReturnedEmbeddingsType
import random
from ressources.common import *
from ressources.gfpgan import *
import tomesd

device_label_outpaint, model_arch = detect_device()
device_outpaint = torch.device(device_label_outpaint)

# Gestion des modèles
model_path_outpaint = "./models/inpaint/"
model_path_safety_checker = "./models/Stable_Diffusion/"
os.makedirs(model_path_outpaint, exist_ok=True)
os.makedirs(model_path_safety_checker, exist_ok=True)
model_list_outpaint = []

for filename in os.listdir(model_path_outpaint):
    f = os.path.join(model_path_outpaint, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
        model_list_outpaint.append(f)

model_list_outpaint_builtin = [
    "Uminosachi/realisticVisionV30_v30VAE-inpainting",
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "runwayml/stable-diffusion-inpainting",
]

for k in range(len(model_list_outpaint_builtin)):
    model_list_outpaint.append(model_list_outpaint_builtin[k])

# Bouton Cancel
stop_outpaint = False

def initiate_stop_outpaint() :
    global stop_outpaint
    stop_outpaint = True

def check_outpaint(pipe, step_index, timestep, callback_kwargs) : 
    global stop_outpaint
    if stop_outpaint == True :
        print(">>>[outpaint 🖌️ ]: generation canceled by user")
        stop_outpaint = False
        pipe._interrupt = True
    return callback_kwargs

def prepare_outpaint(img_outpaint, top, bottom, left, right) :
    image = np.array(img_outpaint)
    mask = np.zeros((image.shape[0], image.shape[1], 3), dtype = np.uint8)
    top = int(top)
    bottom = int(bottom)
    left = int(left)
    right = int(right)
    image = cv2.copyMakeBorder(
        image, 
        top, 
        bottom, 
        left, 
        right, 
        cv2.BORDER_CONSTANT, 
        None, 
        [255, 255, 255]
    )
    mask = cv2.copyMakeBorder(
        mask, 
        top, 
        bottom, 
        left, 
        right, 
        cv2.BORDER_CONSTANT, 
        None, 
        [255, 255, 255]
    )
    return image, image, mask, mask

@metrics_decoration
def image_outpaint(
    modelid_outpaint, 
    sampler_outpaint, 
    img_outpaint, 
    mask_outpaint, 
    rotation_img_outpaint, 
    prompt_outpaint, 
    negative_prompt_outpaint, 
    num_images_per_prompt_outpaint, 
    num_prompt_outpaint, 
    guidance_scale_outpaint,
    denoising_strength_outpaint, 
    num_inference_step_outpaint, 
    height_outpaint, 
    width_outpaint, 
    seed_outpaint, 
    use_gfpgan_outpaint, 
    nsfw_filter, 
    tkme_outpaint,
    progress_outpaint=gr.Progress(track_tqdm=True)
    ):

    print(">>>[outpaint 🖌️ ]: starting module") 
    
    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_safety_checker, device_outpaint, nsfw_filter)

    if ("XL" in modelid_outpaint.upper()):
        is_xl_outpaint: bool = True
    else :        
        is_xl_outpaint: bool = False

    if (is_xl_outpaint == True):
        if modelid_outpaint[0:9] == "./models/" :
            pipe_outpaint = StableDiffusionXLInpaintPipeline.from_single_file(
                modelid_outpaint, 
                torch_dtype=model_arch,
                use_safetensors=True,
                load_safety_checker=False if (nsfw_filter_final == None) else True,
#                safety_checker=nsfw_filter_final,
#                feature_extractor=feat_ex
            )
        else:
            pipe_outpaint = StableDiffusionXLInpaintPipeline.from_pretrained(
                modelid_outpaint, 
                cache_dir=model_path_outpaint, 
                torch_dtype=model_arch,
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
    else:
        if modelid_outpaint[0:9] == "./models/" :
            pipe_outpaint = StableDiffusionInpaintPipeline.from_single_file(
                modelid_outpaint, 
                torch_dtype=model_arch,
                use_safetensors=True,
                load_safety_checker=False if (nsfw_filter_final == None) else True,
#                safety_checker=nsfw_filter_final,
#                feature_extractor=feat_ex
            )
        else:
            pipe_outpaint = StableDiffusionInpaintPipeline.from_pretrained(
                modelid_outpaint, 
                cache_dir=model_path_outpaint, 
                torch_dtype=model_arch,
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
    pipe_outpaint = schedulerer(pipe_outpaint, sampler_outpaint)
    pipe_outpaint.enable_attention_slicing("max")
    tomesd.apply_patch(pipe_outpaint, ratio=tkme_outpaint)
    if device_label_outpaint == "cuda" :
        pipe_outpaint.enable_sequential_cpu_offload()
    else : 
        pipe_outpaint = pipe_outpaint.to(device_outpaint)
    
    if seed_outpaint == 0:
        random_seed = random.randrange(0, 10000000000, 1)
        final_seed = random_seed
    else:
        final_seed = seed_outpaint
    generator = []
    for k in range(num_prompt_outpaint):
        generator.append([torch.Generator(device_outpaint).manual_seed(final_seed + (k*num_images_per_prompt_outpaint) + l ) for l in range(num_images_per_prompt_outpaint)])

#   angle_outpaint = 360 - rotation_img_outpaint   
#   img_outpaint["image"] = img_outpaint["image"].rotate(angle_outpaint, expand=True)
#   dim_size = correct_size(width_outpaint, height_outpaint, 512)
#   image_input = img_outpaint["image"].convert("RGB")
#   mask_image_input = img_outpaint["mask"].convert("RGB")
#   image_input = image_input.resize((dim_size[0],dim_size[1]))
#   mask_image_input = mask_image_input.resize((dim_size[0],dim_size[1]))    
#   savename = f"outputs/mask.png"
#   mask_image_input.save(savename)    


    image_input = img_outpaint.convert("RGB")
    mask_image_input = mask_outpaint.convert("RGB")
    dim_size = round_size(image_input)
    savename_mask = f"outputs/mask.png"
    mask_image_input.save(savename_mask) 

#    mask_image_input = PIL.Image.open(mask_outpaint)
#    mask_image_input = image_input.convert("RGB")    
    
    prompt_outpaint = str(prompt_outpaint)
    negative_prompt_outpaint = str(negative_prompt_outpaint)
    if prompt_outpaint == "None":
        prompt_outpaint = ""
    if negative_prompt_outpaint == "None":
        negative_prompt_outpaint = ""

    if (is_xl_outpaint == True):
        compel = Compel(
            tokenizer=[pipe_outpaint.tokenizer, pipe_outpaint.tokenizer_2],
            text_encoder=[pipe_outpaint.text_encoder, pipe_outpaint.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            device=device_outpaint,
        )
        conditioning, pooled = compel(prompt_outpaint)
        neg_conditioning, neg_pooled = compel(negative_prompt_outpaint)
        [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])
    else:
        compel = Compel(tokenizer=pipe_outpaint.tokenizer, text_encoder=pipe_outpaint.text_encoder, truncate_long_prompts=False, device=device_outpaint)
        conditioning = compel.build_conditioning_tensor(prompt_outpaint)
        neg_conditioning = compel.build_conditioning_tensor(negative_prompt_outpaint)
        [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])

    final_image = []
    final_seed = []    
    for i in range (num_prompt_outpaint):
        if (is_xl_outpaint == True):
            image = pipe_outpaint(
                image=image_input,
                mask_image=mask_image_input,
                prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled, 
                negative_prompt_embeds=neg_conditioning,
                negative_pooled_prompt_embeds=neg_pooled,
                num_images_per_prompt=num_images_per_prompt_outpaint,
                guidance_scale=guidance_scale_outpaint,
                strength=denoising_strength_outpaint,
                width=dim_size[0],
                height=dim_size[1],
                num_inference_steps=num_inference_step_outpaint,
                generator = generator[i],
                callback_on_step_end=check_outpaint, 
                callback_on_step_end_tensor_inputs=['latents'],
            ).images
        else:
            image = pipe_outpaint(
                image=image_input,
                mask_image=mask_image_input,
                prompt_embeds=conditioning,
                negative_prompt_embeds=neg_conditioning,
                num_images_per_prompt=num_images_per_prompt_outpaint,
                guidance_scale=guidance_scale_outpaint,
                strength=denoising_strength_outpaint,
                width=dim_size[0],
                height=dim_size[1],
                num_inference_steps=num_inference_step_outpaint,
                generator = generator[i],
                callback_on_step_end=check_outpaint, 
                callback_on_step_end_tensor_inputs=['latents'],
            ).images

        for j in range(len(image)):
            seed_id = random_seed + i*num_images_per_prompt_outpaint + j if (seed_outpaint == 0) else seed_outpaint + i*num_images_per_prompt_outpaint + j
            savename = name_seeded_image(seed_id)
            if use_gfpgan_outpaint == True :
                image[j] = image_gfpgan_mini(image[j])
            image[j].save(savename)
            final_image.append(savename)
            final_seed.append(seed_id)

    print(f">>>[outpaint 🖌️ ]: generated {num_prompt_outpaint} batch(es) of {num_images_per_prompt_outpaint}")
    reporting_outpaint = f">>>[outpaint 🖌️ ]: "+\
        f"Settings : Model={modelid_outpaint} | "+\
        f"Sampler={sampler_outpaint} | "+\
        f"Steps={num_inference_step_outpaint} | "+\
        f"CFG scale={guidance_scale_outpaint} | "+\
        f"Size={dim_size[0]}x{dim_size[1]} | "+\
        f"GFPGAN={use_gfpgan_outpaint} | "+\
        f"Token merging={tkme_outpaint} | "+\
        f"nsfw_filter={bool(int(nsfw_filter))} | "+\
        f"Denoising strength={denoising_strength_outpaint} | "+\
        f"Prompt={prompt_outpaint} | "+\
        f"Negative prompt={negative_prompt_outpaint} | "+\
        f"Seed List="+ ', '.join([f"{final_seed[m]}" for m in range(len(final_seed))])
    print(reporting_outpaint) 

    final_image.append(savename_mask)

    exif_writer_png(reporting_outpaint, final_image)

    del nsfw_filter_final, feat_ex, pipe_outpaint, generator, image_input, mask_image_input, image
    clean_ram()

    print(f">>>[outpaint 🖌️ ]: leaving module")

    return final_image, final_image
