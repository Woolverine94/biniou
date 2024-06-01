# https://github.com/Woolverine94/biniou
# inpaint.py
import gradio as gr
import os
import PIL
import torch
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline
from compel import Compel, ReturnedEmbeddingsType
import random
from ressources.common import *
from ressources.gfpgan import *
import tomesd
from diffusers.schedulers import AysSchedules

device_label_inpaint, model_arch = detect_device()
device_inpaint = torch.device(device_label_inpaint)

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
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "runwayml/stable-diffusion-inpainting",
]

for k in range(len(model_list_inpaint_builtin)):
    model_list_inpaint.append(model_list_inpaint_builtin[k])

# Bouton Cancel
stop_inpaint = False

def initiate_stop_inpaint() :
    global stop_inpaint
    stop_inpaint = True

def check_inpaint(pipe, step_index, timestep, callback_kwargs):
    global stop_inpaint
    if stop_inpaint == True:
        print(">>>[inpaint 🖌️ ]: generation canceled by user")
        stop_inpaint = False
        pipe._interrupt = True
    return callback_kwargs

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
    clipskip_inpaint,
    use_ays_inpaint,
    progress_inpaint=gr.Progress(track_tqdm=True)
    ):

    print(">>>[inpaint 🖌️ ]: starting module")

    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_safety_checker, device_inpaint, nsfw_filter)

    if clipskip_inpaint == 0:
       clipskip_inpaint = None

    if ("XL" in modelid_inpaint.upper()):
        is_xl_inpaint: bool = True
    else :        
        is_xl_inpaint: bool = False

    if (num_inference_step_inpaint >= 10) and use_ays_inpaint:
        if is_sdxl(modelid_inpaint):
            sampling_schedule_inpaint = AysSchedules["StableDiffusionXLTimesteps"]
            sampler_inpaint = "DPM++ SDE"
        else:
            sampling_schedule_inpaint = AysSchedules["StableDiffusionTimesteps"]
            sampler_inpaint = "Euler"
        num_inference_step_inpaint = 10
    else:
        sampling_schedule_inpaint = None

    if (is_xl_inpaint == True):
        if modelid_inpaint[0:9] == "./models/" :
            pipe_inpaint = StableDiffusionXLInpaintPipeline.from_single_file(
                modelid_inpaint, 
                torch_dtype=model_arch,
                use_safetensors=True, 
                load_safety_checker=False if (nsfw_filter_final == None) else True,
#                safety_checker=nsfw_filter_final, 
#                feature_extractor=feat_ex
            )
        else :        
            pipe_inpaint = StableDiffusionXLInpaintPipeline.from_pretrained(
                modelid_inpaint, 
                cache_dir=model_path_inpaint, 
                torch_dtype=model_arch,
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
    else:
        if modelid_inpaint[0:9] == "./models/" :
            pipe_inpaint = StableDiffusionInpaintPipeline.from_single_file(
                modelid_inpaint, 
                torch_dtype=model_arch,
                use_safetensors=True, 
                load_safety_checker=False if (nsfw_filter_final == None) else True,
#                safety_checker=nsfw_filter_final, 
#                feature_extractor=feat_ex
            )
        else :        
            pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
                modelid_inpaint, 
                cache_dir=model_path_inpaint, 
                torch_dtype=model_arch,
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
    pipe_inpaint = schedulerer(pipe_inpaint, sampler_inpaint)
    pipe_inpaint.enable_attention_slicing("max")
    tomesd.apply_patch(pipe_inpaint, ratio=tkme_inpaint)
    if device_label_inpaint == "cuda" :
        pipe_inpaint.enable_sequential_cpu_offload()
    else : 
        pipe_inpaint = pipe_inpaint.to(device_inpaint)
    
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
    if (is_xl_inpaint == True):
        dim_size = correct_size(width_inpaint, height_inpaint, 1024)
    else:
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

    if (is_xl_inpaint == True):
        compel = Compel(
            tokenizer=[pipe_inpaint.tokenizer, pipe_inpaint.tokenizer_2],
            text_encoder=[pipe_inpaint.text_encoder, pipe_inpaint.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            device=device_inpaint,
        )
        conditioning, pooled = compel(prompt_inpaint)
        neg_conditioning, neg_pooled = compel(negative_prompt_inpaint)
        [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])
    else:
        compel = Compel(tokenizer=pipe_inpaint.tokenizer, text_encoder=pipe_inpaint.text_encoder, truncate_long_prompts=False, device=device_inpaint)
        conditioning = compel.build_conditioning_tensor(prompt_inpaint)
        neg_conditioning = compel.build_conditioning_tensor(negative_prompt_inpaint)
        [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])

    final_image = []
    final_seed = []
    for i in range (num_prompt_inpaint):
        if (is_xl_inpaint == True):
            image = pipe_inpaint(
                image=image_input,
                mask_image=mask_image_input,
                prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled, 
                negative_prompt_embeds=neg_conditioning,
                negative_pooled_prompt_embeds=neg_pooled,
                num_images_per_prompt=num_images_per_prompt_inpaint,
                guidance_scale=guidance_scale_inpaint,
                strength=denoising_strength_inpaint,
                width=dim_size[0],
                height=dim_size[1],
                num_inference_steps=num_inference_step_inpaint,
                timesteps=sampling_schedule_inpaint,
                generator=generator[i],
                clip_skip=clipskip_inpaint,
                callback_on_step_end=check_inpaint, 
                callback_on_step_end_tensor_inputs=['latents'],
            ).images
        else:
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
                timesteps=sampling_schedule_inpaint,
                generator=generator[i],
                clip_skip=clipskip_inpaint,
                callback_on_step_end=check_inpaint, 
                callback_on_step_end_tensor_inputs=['latents'],
            ).images

        for j in range(len(image)):
            seed_id = random_seed + i*num_images_per_prompt_inpaint + j if (seed_inpaint == 0) else seed_inpaint + i*num_images_per_prompt_inpaint + j
            savename = name_seeded_image(seed_id)
            if use_gfpgan_inpaint == True :
                image[j] = image_gfpgan_mini(image[j])
            image[j].save(savename)
            final_image.append(savename)
            final_seed.append(seed_id)

    final_image.append(savename_mask)

    print(f">>>[inpaint 🖌️ ]: generated {num_prompt_inpaint} batch(es) of {num_images_per_prompt_inpaint}")
    reporting_inpaint = f">>>[inpaint 🖌️ ]: "+\
        f"Settings : Model={modelid_inpaint} | "+\
        f"Sampler={sampler_inpaint} | "+\
        f"Steps={num_inference_step_inpaint} | "+\
        f"CFG scale={guidance_scale_inpaint} | "+\
        f"Size={dim_size[0]}x{dim_size[1]} | "+\
        f"GFPGAN={use_gfpgan_inpaint} | "+\
        f"Token merging={tkme_inpaint} | "+\
        f"CLIP skip={clipskip_inpaint} | "+\
        f"AYS={use_ays_inpaint} | "+\
        f"nsfw_filter={bool(int(nsfw_filter))} | "+\
        f"Denoising strength={denoising_strength_inpaint} | "+\
        f"Prompt={prompt_inpaint} | "+\
        f"Negative prompt={negative_prompt_inpaint} | "+\
        f"Seed List="+ ', '.join([f"{final_seed[m]}" for m in range(len(final_seed))])
    print(reporting_inpaint) 

    exif_writer_png(reporting_inpaint, final_image)

    del nsfw_filter_final, feat_ex, pipe_inpaint, generator, image_input, mask_image_input, image
    clean_ram()

    print(f">>>[inpaint 🖌️ ]: leaving module")
    return final_image, final_image
