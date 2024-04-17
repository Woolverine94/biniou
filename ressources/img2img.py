# https://github.com/Woolverine94/biniou
# img2img.py
import gradio as gr
import os
import PIL
import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline, AutoPipelineForImage2Image
from compel import Compel, ReturnedEmbeddingsType
import random
from ressources.common import *
from ressources.gfpgan import *
import tomesd

device_label_img2img, model_arch = detect_device()
device_img2img = torch.device(device_label_img2img)

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
    "IDKiro/sdxs-512-dreamshaper",
    "IDKiro/sdxs-512-0.9",
    "playgroundai/playground-v2-512px-base",
    "playgroundai/playground-v2-1024px-aesthetic",
    "stabilityai/sd-turbo",
    "stabilityai/sdxl-turbo",
    "thibaud/sdxl_dpo_turbo",
    "SG161222/RealVisXL_V4.0_Lightning",
    "cagliostrolab/animagine-xl-3.1",
    "dataautogpt3/OpenDalleV1.1",
    "dataautogpt3/ProteusV0.4",
    "dataautogpt3/ProteusV0.4-Lightning",
    "etri-vilab/koala-1b",
    "etri-vilab/koala-700m",
    "digiplay/AbsoluteReality_v1.8.1",
    "segmind/Segmind-Vega",
    "segmind/SSD-1B",
    "gsdf/Counterfeit-V2.5",
#    "ckpt/anything-v4.5-vae-swapped",
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

def check_img2img(pipe, step_index, timestep, callback_kwargs) : 
    global stop_img2img
    if stop_img2img == True :
        print(">>>[img2img 🖌️ ]: generation canceled by user")
        stop_img2img = False
        pipe._interrupt = True
    return callback_kwargs

@metrics_decoration
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
    lora_model_img2img,
    lora_weight_img2img,
    txtinv_img2img,
    progress_img2img=gr.Progress(track_tqdm=True)
    ):

    print(">>>[img2img 🖌️ ]: starting module")

    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_img2img, device_img2img, nsfw_filter)

    if ("turbo" in modelid_img2img):
        is_turbo_img2img: bool = True
    else :
        is_turbo_img2img: bool = False

    if (("XL" in modelid_img2img.upper()) or ("LIGHTNING" in modelid_img2img.upper()) or ("ETRI-VILAB/KOALA-" in modelid_img2img.upper()) or ("PLAYGROUNDAI/PLAYGROUND-V2-" in modelid_img2img.upper()) or (modelid_img2img == "segmind/SSD-1B") or (modelid_img2img == "segmind/Segmind-Vega") or (modelid_img2img == "dataautogpt3/OpenDalleV1.1") or (modelid_img2img == "dataautogpt3/ProteusV0.4")):
        is_xl_img2img: bool = True
    else :        
        is_xl_img2img: bool = False        

    if ("dataautogpt3/ProteusV0.4" in modelid_img2img):
        is_bin_img2img: bool = True
    else :
        is_bin_img2img: bool = False

    if (is_turbo_img2img == True):
        if modelid_img2img[0:9] == "./models/" :
            pipe_img2img = AutoPipelineForImage2Image.from_single_file(
                modelid_img2img, 
                torch_dtype=model_arch,
                use_safetensors=True if not is_bin_img2img else False,
                load_safety_checker=False if (nsfw_filter_final == None) else True,
#                safety_checker=nsfw_filter_final, 
#                feature_extractor=feat_ex,
            )
        else :        
            pipe_img2img = AutoPipelineForImage2Image.from_pretrained(
                modelid_img2img, 
                cache_dir=model_path_img2img, 
                torch_dtype=model_arch,
                use_safetensors=True if not is_bin_img2img else False,
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=True if offline_test() else None                
            )
    elif (is_xl_img2img == True) and (is_turbo_img2img == False):
        if modelid_img2img[0:9] == "./models/" :
            pipe_img2img = StableDiffusionXLImg2ImgPipeline.from_single_file(
                modelid_img2img, 
                torch_dtype=model_arch,
                use_safetensors=True if not is_bin_img2img else False,
                load_safety_checker=False if (nsfw_filter_final == None) else True,
#                safety_checker=nsfw_filter_final, 
#                feature_extractor=feat_ex,
            )
        else :        
            pipe_img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                modelid_img2img, 
                cache_dir=model_path_img2img, 
                torch_dtype=model_arch,
                use_safetensors=True if not is_bin_img2img else False,
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=True if offline_test() else None                
            )
    else :
        if modelid_img2img[0:9] == "./models/" :
            pipe_img2img = StableDiffusionImg2ImgPipeline.from_single_file(
                modelid_img2img, 
                torch_dtype=model_arch,
                use_safetensors=True if not is_bin_img2img else False,
                load_safety_checker=False if (nsfw_filter_final == None) else True,
#                safety_checker=nsfw_filter_final, 
#                feature_extractor=feat_ex,
            )
        else :        
            pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                modelid_img2img, 
                cache_dir=model_path_img2img, 
                torch_dtype=model_arch,
                use_safetensors=True if not is_bin_img2img else False,
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=True if offline_test() else None                
            )

    pipe_img2img = schedulerer(pipe_img2img, sampler_img2img)
    pipe_img2img.enable_attention_slicing("max")
    tomesd.apply_patch(pipe_img2img, ratio=tkme_img2img)
    if device_label_img2img == "cuda" :
        pipe_img2img.enable_sequential_cpu_offload()
    else : 
        pipe_img2img = pipe_img2img.to(device_img2img)

    if lora_model_img2img != "":
        model_list_lora_img2img = lora_model_list(modelid_img2img)
        if modelid_img2img[0:9] == "./models/":
            pipe_img2img.load_lora_weights(
                os.path.dirname(lora_model_img2img),
                weight_name=model_list_lora_img2img[lora_model_img2img][0],
                use_safetensors=True,
                adapter_name="adapter1",
            )
        else:
            if is_xl_img2img:
                lora_model_path = "./models/lora/SDXL"
            else: 
                lora_model_path = "./models/lora/SD"
            pipe_img2img.load_lora_weights(
                lora_model_img2img,
                weight_name=model_list_lora_img2img[lora_model_img2img][0],
                cache_dir=lora_model_path,
                use_safetensors=True,
                adapter_name="adapter1",
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
        pipe_img2img.fuse_lora(lora_scale=lora_weight_img2img)
#        pipe_img2img.set_adapters(["adapter1"], adapter_weights=[float(lora_weight_img2img)])

    if txtinv_img2img != "":
        model_list_txtinv_img2img = txtinv_list(modelid_img2img)
        weight_img2img = model_list_txtinv_img2img[txtinv_img2img][0]
        token_img2img =  model_list_txtinv_img2img[txtinv_img2img][1]
        if modelid_img2img[0:9] == "./models/":
            model_path_txtinv = "./models/TextualInversion"
            pipe_img2img.load_textual_inversion(
                txtinv_img2img,
                weight_name=weight_img2img,
                use_safetensors=True,
                token=token_img2img,
            )
        else:
            if is_xl_img2img:
                model_path_txtinv = "./models/TextualInversion/SDXL"
            else: 
                model_path_txtinv = "./models/TextualInversion/SD"
            pipe_img2img.load_textual_inversion(
                txtinv_img2img,
                weight_name=weight_img2img,
                cache_dir=model_path_txtinv,
                use_safetensors=True,
                token=token_img2img,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )

    if seed_img2img == 0:
        random_seed = torch.randint(0, 10000000000, (1,))
        generator = torch.manual_seed(random_seed)
    else:
        generator = torch.manual_seed(seed_img2img)

    if source_type_img2img == "sketch" :
        dim_size=[512, 512]
    elif (is_xl_img2img == True) and not (is_turbo_img2img == True) :
        dim_size = correct_size(width_img2img, height_img2img, 1024)
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

    if (is_xl_img2img == True) :
        compel = Compel(
            tokenizer=pipe_img2img.tokenizer_2, 
            text_encoder=pipe_img2img.text_encoder_2, 
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, 
            requires_pooled=[False, True],
            device=device_img2img,
        )
        conditioning, pooled = compel(prompt_img2img)
        neg_conditioning, neg_pooled = compel(negative_prompt_img2img)
        [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])
    else :
        compel = Compel(tokenizer=pipe_img2img.tokenizer, text_encoder=pipe_img2img.text_encoder, truncate_long_prompts=False, device=device_img2img)
        conditioning = compel.build_conditioning_tensor(prompt_img2img)
        neg_conditioning = compel.build_conditioning_tensor(negative_prompt_img2img)
        [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])
    
    final_image = []
    for i in range (num_prompt_img2img):
        if (is_turbo_img2img == True) :
            image = pipe_img2img(        
                image=image_input,
                prompt=prompt_img2img,
                num_images_per_prompt=num_images_per_prompt_img2img,
                guidance_scale=guidance_scale_img2img,
                strength=denoising_strength_img2img,
                num_inference_steps=num_inference_step_img2img,
                generator = generator,
                callback_on_step_end=check_img2img, 
                callback_on_step_end_tensor_inputs=['latents'], 
            ).images
        elif (is_xl_img2img == True) :
            image = pipe_img2img(        
                image=image_input,
                prompt=prompt_img2img,
                negative_prompt=negative_prompt_img2img,
#                prompt_embeds=conditioning, 
#                pooled_prompt_embeds=pooled, 
#                negative_prompt_embeds=neg_conditioning,
#                negative_pooled_prompt_embeds=neg_pooled,
                num_images_per_prompt=num_images_per_prompt_img2img,
                guidance_scale=guidance_scale_img2img,
                strength=denoising_strength_img2img,
                num_inference_steps=num_inference_step_img2img,
                generator = generator,
                callback_on_step_end=check_img2img, 
                callback_on_step_end_tensor_inputs=['latents'], 
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
                callback_on_step_end=check_img2img,
                callback_on_step_end_tensor_inputs=['latents'],
            ).images

        for j in range(len(image)):
            savename = name_image()
            if use_gfpgan_img2img == True :
                image[j] = image_gfpgan_mini(image[j])             
            image[j].save(savename)
            final_image.append(savename)

    if source_type_img2img == "sketch" :
        savename_mask = f"outputs/input_image.png"
        image_input.save(savename_mask)
        final_image.append(savename_mask)

    print(f">>>[img2img 🖌️ ]: generated {num_prompt_img2img} batch(es) of {num_images_per_prompt_img2img}")        
    reporting_img2img = f">>>[img2img 🖌️ ]: "+\
        f"Settings : Model={modelid_img2img} | "+\
        f"XL model={is_xl_img2img} | "+\
        f"Sampler={sampler_img2img} | "+\
        f"Steps={num_inference_step_img2img} | "+\
        f"CFG scale={guidance_scale_img2img} | "+\
        f"Size={width_img2img}x{height_img2img} | "+\
        f"GFPGAN={use_gfpgan_img2img} | "+\
        f"Token merging={tkme_img2img} | "+\
        f"LoRA model={lora_model_img2img} | "+\
        f"LoRA weight={lora_weight_img2img} | "+\
        f"Textual inversion={txtinv_img2img} | "+\
        f"nsfw_filter={bool(int(nsfw_filter))} | "+\
        f"Denoising strength={denoising_strength_img2img} | "+\
        f"Prompt={prompt_img2img} | "+\
        f"Negative prompt={negative_prompt_img2img}"
    print(reporting_img2img)         

    exif_writer_png(reporting_img2img, final_image)

    del nsfw_filter_final, feat_ex, pipe_img2img, generator, image_input, compel, conditioning, neg_conditioning, image
    clean_ram()

    print(f">>>[img2img 🖌️ ]: leaving module")   
    return final_image, final_image 
