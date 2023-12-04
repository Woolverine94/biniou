# https://github.com/Woolverine94/biniou
# img2img_ip.py
import gradio as gr
import os
import PIL
import torch
from diffusers import AutoPipelineForImage2Image, StableDiffusionXLImg2ImgPipeline, StableDiffusionImg2ImgPipeline
from compel import Compel, ReturnedEmbeddingsType
import time
import random
from ressources.scheduler import *
from ressources.common import *
from ressources.gfpgan import *
import tomesd

device_label_img2img_ip, model_arch = detect_device()
device_img2img_ip = torch.device(device_label_img2img_ip)

# Gestion des modèles
model_path_img2img_ip = "./models/Stable_Diffusion/"
model_path_ipa_img2img_ip = "./models/Ip-Adapters"
os.makedirs(model_path_img2img_ip, exist_ok=True)
os.makedirs(model_path_ipa_img2img_ip, exist_ok=True)

model_list_img2img_ip = []

for filename in os.listdir(model_path_img2img_ip):
    f = os.path.join(model_path_img2img_ip, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
        model_list_img2img_ip.append(f)

model_list_img2img_ip_builtin = [
    "SG161222/Realistic_Vision_V3.0_VAE",
#    "stabilityai/sd-turbo",
    "stabilityai/sdxl-turbo",
#    "ckpt/anything-v4.5-vae-swapped",
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    "runwayml/stable-diffusion-v1-5",
    "nitrosocke/Ghibli-Diffusion",     
]

for k in range(len(model_list_img2img_ip_builtin)):
    model_list_img2img_ip.append(model_list_img2img_ip_builtin[k])

# Bouton Cancel
stop_img2img_ip = False

def initiate_stop_img2img_ip() :
    global stop_img2img_ip
    stop_img2img_ip = True

def check_img2img_ip(pipe, step_index, timestep, callback_kwargs) : 
    global stop_img2img_ip
    if stop_img2img_ip == False :
        return callback_kwargs
    elif stop_img2img_ip == True :
        print(">>>[img2img_ip 🖌️ ]: generation canceled by user")
        stop_img2img_ip = False
        try:
            del ressources.img2img_ip.pipe_img2img_ip
        except NameError as e:
            raise Exception("Interrupting ...")
    return

@metrics_decoration
def image_img2img_ip(
    modelid_img2img_ip, 
    sampler_img2img_ip, 
    img_img2img_ip, 
    img_ipa_img2img_ip,
    prompt_img2img_ip, 
    negative_prompt_img2img_ip, 
    num_images_per_prompt_img2img_ip, 
    num_prompt_img2img_ip, 
    guidance_scale_img2img_ip, 
    denoising_strength_img2img_ip, 
    num_inference_step_img2img_ip, 
    height_img2img_ip, 
    width_img2img_ip, 
    seed_img2img_ip, 
    use_gfpgan_img2img_ip, 
    nsfw_filter, 
    tkme_img2img_ip,    
    progress_img2img_ip=gr.Progress(track_tqdm=True)
    ):

    print(">>>[img2img_ip 🖌️ ]: starting module")

    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_img2img_ip, device_img2img_ip, nsfw_filter)

    if (modelid_img2img_ip == "stabilityai/sdxl-turbo") or (modelid_img2img_ip == "stabilityai/sd-turbo"):
        is_xlturbo_img2img_ip: bool = True
    else :
        is_xlturbo_img2img_ip: bool = False

    if ('xl' or 'XL' or 'Xl' or 'xL') in modelid_img2img_ip :
        is_xl_img2img_ip: bool = True
    else :        
        is_xl_img2img_ip: bool = False        

    if (is_xlturbo_img2img_ip == True) :
        if modelid_img2img_ip[0:9] == "./models/" :
            pipe_img2img_ip = AutoPipelineForImage2Image.from_single_file(
                modelid_img2img_ip, 
                torch_dtype=model_arch,
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
            )
        else :        
            pipe_img2img_ip = AutoPipelineForImage2Image.from_pretrained(
                modelid_img2img_ip, 
                cache_dir=model_path_img2img_ip, 
                torch_dtype=model_arch,
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
    elif (is_xl_img2img_ip == True) and (is_xlturbo_img2img_ip == False) :
        if modelid_img2img_ip[0:9] == "./models/" :
            pipe_img2img_ip = StableDiffusionXLImg2ImgPipeline.from_single_file(
                modelid_img2img_ip, 
                torch_dtype=model_arch,
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
            )
        else :        
            pipe_img2img_ip = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                modelid_img2img_ip, 
                cache_dir=model_path_img2img_ip, 
                torch_dtype=model_arch,
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
    else :
        if modelid_img2img_ip[0:9] == "./models/" :
            pipe_img2img_ip = StableDiffusionImg2ImgPipeline.from_single_file(
                modelid_img2img_ip, 
                torch_dtype=model_arch,
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
            )
        else :        
            pipe_img2img_ip = StableDiffusionImg2ImgPipeline.from_pretrained(
                modelid_img2img_ip, 
                cache_dir=model_path_img2img_ip, 
                torch_dtype=model_arch,
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )

#    if (is_xl_img2img_ip == True) or (is_xlturbo_img2img_ip == True):
    if (is_xl_img2img_ip == True):
        pipe_img2img_ip.load_ip_adapter(
            "h94/IP-Adapter", 
            cache_dir=model_path_ipa_img2img_ip,
            subfolder="sdxl_models", 
            weight_name="ip-adapter_sdxl.bin",
            torch_dtype=model_arch,
            use_safetensors=True,
            resume_download=True,
            local_files_only=True if offline_test() else None
        )
    else:
        pipe_img2img_ip.load_ip_adapter(
            "h94/IP-Adapter", 
            cache_dir=model_path_ipa_img2img_ip,            
            subfolder="models", 
            weight_name="ip-adapter_sd15.bin",
            torch_dtype=model_arch,
            use_safetensors=True, 
            resume_download=True,
            local_files_only=True if offline_test() else None
        )
 
#    pipe_img2img_ip.set_ip_adapter_scale(denoising_strength_img2img_ip)    
    pipe_img2img_ip = get_scheduler(pipe=pipe_img2img_ip, scheduler=sampler_img2img_ip)
#    pipe_img2img_ip.enable_attention_slicing("max")  
    tomesd.apply_patch(pipe_img2img_ip, ratio=tkme_img2img_ip)
    if device_label_img2img_ip == "cuda" :
        pipe_img2img_ip.enable_sequential_cpu_offload()
    else : 
        pipe_img2img_ip = pipe_img2img_ip.to(device_img2img_ip)
    
    if seed_img2img_ip == 0:
        random_seed = torch.randint(0, 10000000000, (1,))
        generator = torch.manual_seed(random_seed)
    else:
        generator = torch.manual_seed(seed_img2img_ip)

    if (img_img2img_ip != None):
        dim_size = correct_size(width_img2img_ip, height_img2img_ip, 512)
        image_input = PIL.Image.open(img_img2img_ip)
        image_input = image_input.convert("RGB")
        image_input = image_input.resize((dim_size[0], dim_size[1]))
    else:
        image_input = None

    if (img_ipa_img2img_ip != None):
        image_input_ipa = PIL.Image.open(img_ipa_img2img_ip)
        dim_size_ipa = correct_size(image_input_ipa.size[0], image_input_ipa.size[1], 512)
        image_input_ipa = image_input_ipa.convert("RGB")
        image_input_ipa = image_input_ipa.resize((dim_size_ipa[0], dim_size_ipa[1]))
    else:
        image_input_ipa = None

    prompt_img2img_ip = str(prompt_img2img_ip)
    negative_prompt_img2img_ip = str(negative_prompt_img2img_ip)
    if prompt_img2img_ip == "None":
        prompt_img2img_ip = ""
    if negative_prompt_img2img_ip == "None":
        negative_prompt_img2img_ip = ""

    if (is_xl_img2img_ip == True) :
        compel = Compel(
            tokenizer=pipe_img2img_ip.tokenizer_2, 
            text_encoder=pipe_img2img_ip.text_encoder_2, 
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, 
            requires_pooled=[False, True], 
        )
        conditioning, pooled = compel(prompt_img2img_ip)
        neg_conditioning, neg_pooled = compel(negative_prompt_img2img_ip)
        [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])
    else :
        compel = Compel(tokenizer=pipe_img2img_ip.tokenizer, text_encoder=pipe_img2img_ip.text_encoder, truncate_long_prompts=False)
        conditioning = compel.build_conditioning_tensor(prompt_img2img_ip)
        neg_conditioning = compel.build_conditioning_tensor(negative_prompt_img2img_ip)
        [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])
    
    final_image = []

    for i in range (num_prompt_img2img_ip):
        if (is_xlturbo_img2img_ip == True) :
            image = pipe_img2img_ip(        
                image=image_input,
                ip_adapter_image=image_input_ipa,
                prompt=prompt_img2img_ip,
                num_images_per_prompt=num_images_per_prompt_img2img_ip,
                guidance_scale=guidance_scale_img2img_ip,
                strength=denoising_strength_img2img_ip,
                num_inference_steps=num_inference_step_img2img_ip,
                generator = generator,
                callback_on_step_end=check_img2img_ip, 
                callback_on_step_end_tensor_inputs=['latents'], 
            ).images
        elif (is_xl_img2img_ip == True) :
            image = pipe_img2img_ip(        
                image=image_input,
                ip_adapter_image=image_input_ipa,
                prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled,
                negative_prompt_embeds=neg_conditioning,
                negative_pooled_prompt_embeds=neg_pooled,
                num_images_per_prompt=num_images_per_prompt_img2img_ip,
                guidance_scale=guidance_scale_img2img_ip,
                strength=denoising_strength_img2img_ip,
                num_inference_steps=num_inference_step_img2img_ip,
                generator = generator,
                callback_on_step_end=check_img2img_ip, 
                callback_on_step_end_tensor_inputs=['latents'], 
            ).images            
        else : 
            image = pipe_img2img_ip(        
                image=image_input,
                ip_adapter_image=image_input_ipa,
                prompt_embeds=conditioning,
                negative_prompt_embeds=neg_conditioning,
                num_images_per_prompt=num_images_per_prompt_img2img_ip,
                guidance_scale=guidance_scale_img2img_ip,
                strength=denoising_strength_img2img_ip,
                num_inference_steps=num_inference_step_img2img_ip,
                generator = generator,
                callback_on_step_end=check_img2img_ip, 
                callback_on_step_end_tensor_inputs=['latents'], 
            ).images        

        for j in range(len(image)):
            timestamp = time.time()
            savename = f"outputs/{timestamp}.png"
            if use_gfpgan_img2img_ip == True :
                image[j] = image_gfpgan_mini(image[j])             
            image[j].save(savename)
            final_image.append(savename)

    print(f">>>[img2img_ip 🖌️ ]: generated {num_prompt_img2img_ip} batch(es) of {num_images_per_prompt_img2img_ip}")        
    reporting_img2img_ip = f">>>[img2img_ip 🖌️ ]: "+\
        f"Settings : Model={modelid_img2img_ip} | "+\
        f"XL model={is_xl_img2img_ip} | "+\
        f"Sampler={sampler_img2img_ip} | "+\
        f"Steps={num_inference_step_img2img_ip} | "+\
        f"CFG scale={guidance_scale_img2img_ip} | "+\
        f"Size={width_img2img_ip}x{height_img2img_ip} | "+\
        f"GFPGAN={use_gfpgan_img2img_ip} | "+\
        f"Token merging={tkme_img2img_ip} | "+\
        f"nsfw_filter={bool(int(nsfw_filter))} | "+\
        f"Denoising strength={denoising_strength_img2img_ip} | "+\
        f"Prompt={prompt_img2img_ip} | "+\
        f"Negative prompt={negative_prompt_img2img_ip}"
    print(reporting_img2img_ip)         
        
    del nsfw_filter_final, feat_ex, pipe_img2img_ip, generator, image_input, image_input_ipa, compel, conditioning, neg_conditioning, image
    clean_ram()

    print(f">>>[img2img_ip 🖌️ ]: leaving module")   
    return final_image, final_image 
