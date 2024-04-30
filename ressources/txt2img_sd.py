# https://github.com/Woolverine94/biniou
# txt2img_sd.py
import gradio as gr
import os
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image
from compel import Compel, ReturnedEmbeddingsType
import torch
import random
from ressources.gfpgan import *
import tomesd

# device_txt2img_sd = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_label_txt2img_sd, model_arch = detect_device()
device_txt2img_sd = torch.device(device_label_txt2img_sd)

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
    "IDKiro/sdxs-512-dreamshaper",
    "IDKiro/sdxs-512-0.9",
    "playgroundai/playground-v2-512px-base",
    "playgroundai/playground-v2-1024px-aesthetic",
    "playgroundai/playground-v2.5-1024px-aesthetic",
    "stabilityai/sd-turbo", 
    "stabilityai/sdxl-turbo", 
    "thibaud/sdxl_dpo_turbo",
    "SG161222/RealVisXL_V4.0_Lightning",
    "cagliostrolab/animagine-xl-3.1",
    "aipicasso/emi-2",
    "IDEA-CCNL/Taiyi-Stable-Diffusion-XL-3.5B",
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

def check_txt2img_sd(pipe, step_index, timestep, callback_kwargs) :
    global stop_txt2img_sd
    if stop_txt2img_sd == True :
        print(">>>[Stable Diffusion 🖼️ ]: generation canceled by user")
        stop_txt2img_sd = False
        pipe._interrupt = True
    return callback_kwargs

@metrics_decoration
def image_txt2img_sd(
    modelid_txt2img_sd, 
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
    lora_model_txt2img_sd,
    lora_weight_txt2img_sd,
    txtinv_txt2img_sd,
    progress_txt2img_sd=gr.Progress(track_tqdm=True)
    ):

    print(">>>[Stable Diffusion 🖼️ ]: starting module")

    global pipe_txt2img_sd
    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_txt2img_sd, device_txt2img_sd, nsfw_filter)

    if ("turbo" in modelid_txt2img_sd):
        is_turbo_txt2img_sd: bool = True
    else :
        is_turbo_txt2img_sd: bool = False

    if is_sdxl(modelid_txt2img_sd):
        is_xl_txt2img_sd: bool = True
    else :        
        is_xl_txt2img_sd: bool = False

    if ("dataautogpt3/ProteusV0.4" in modelid_txt2img_sd):
        is_bin_txt2img_sd: bool = True
    else :
        is_bin_txt2img_sd: bool = False

    if (is_turbo_txt2img_sd == True) :
        if modelid_txt2img_sd[0:9] == "./models/" :
            pipe_txt2img_sd =AutoPipelineForText2Image.from_single_file(
                modelid_txt2img_sd, 
#                torch_dtype=torch.float32, 
                torch_dtype=model_arch, 
                use_safetensors=True if not is_bin_txt2img_sd else False,
                load_safety_checker=False if (nsfw_filter_final == None) else True,
#                safety_checker=nsfw_filter_final, 
#                feature_extractor=feat_ex,
            )
        else :        
            pipe_txt2img_sd = AutoPipelineForText2Image.from_pretrained(
                modelid_txt2img_sd, 
                cache_dir=model_path_txt2img_sd, 
#                torch_dtype=torch.float32, 
                torch_dtype=model_arch, 
                use_safetensors=True if not is_bin_txt2img_sd else False,
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
    elif (is_xl_txt2img_sd == True) :
        if modelid_txt2img_sd[0:9] == "./models/" :
            pipe_txt2img_sd = StableDiffusionXLPipeline.from_single_file(
                modelid_txt2img_sd, 
#                torch_dtype=torch.float32, 
                torch_dtype=model_arch, 
                use_safetensors=True if not is_bin_txt2img_sd else False,
                load_safety_checker=False if (nsfw_filter_final == None) else True,
#                safety_checker=nsfw_filter_final, 
#                feature_extractor=feat_ex,
            )
        else :        
            pipe_txt2img_sd = StableDiffusionXLPipeline.from_pretrained(
                modelid_txt2img_sd, 
                cache_dir=model_path_txt2img_sd, 
#                torch_dtype=torch.float32, 
                torch_dtype=model_arch, 
                use_safetensors=True if not is_bin_txt2img_sd else False,
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
    else :
        if modelid_txt2img_sd[0:9] == "./models/" :
            pipe_txt2img_sd = StableDiffusionPipeline.from_single_file(
                modelid_txt2img_sd, 
#                torch_dtype=torch.float32, 
                torch_dtype=model_arch,                 
                use_safetensors=True if not is_bin_txt2img_sd else False,
                load_safety_checker=False if (nsfw_filter_final == None) else True,
#                safety_checker=nsfw_filter_final, 
#                feature_extractor=feat_ex,
            )
        else :        
            pipe_txt2img_sd = StableDiffusionPipeline.from_pretrained(
                modelid_txt2img_sd, 
                cache_dir=model_path_txt2img_sd, 
 #               torch_dtype=torch.float32, 
                torch_dtype=model_arch, 
                use_safetensors=True if not is_bin_txt2img_sd else False,
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )

    pipe_txt2img_sd = schedulerer(pipe_txt2img_sd, sampler_txt2img_sd)
#    if lora_model_txt2img_sd == "":
    pipe_txt2img_sd.enable_attention_slicing("max")
    tomesd.apply_patch(pipe_txt2img_sd, ratio=tkme_txt2img_sd)
    if device_label_txt2img_sd == "cuda" :
        pipe_txt2img_sd.enable_sequential_cpu_offload()
    else : 
        pipe_txt2img_sd = pipe_txt2img_sd.to(device_txt2img_sd)
    pipe_txt2img_sd.enable_vae_slicing()

    if lora_model_txt2img_sd != "":
        model_list_lora_txt2img_sd = lora_model_list(modelid_txt2img_sd)
        if modelid_txt2img_sd[0:9] == "./models/":
            pipe_txt2img_sd.load_lora_weights(
                os.path.dirname(lora_model_txt2img_sd),
                weight_name=model_list_lora_txt2img_sd[lora_model_txt2img_sd][0],
                use_safetensors=True,
                adapter_name="adapter1",
            )
        else:
            if is_xl_txt2img_sd:
                lora_model_path = model_path_lora_sdxl
            else: 
                lora_model_path = model_path_lora_sd
            pipe_txt2img_sd.load_lora_weights(
                lora_model_txt2img_sd,
                weight_name=model_list_lora_txt2img_sd[lora_model_txt2img_sd][0],
                cache_dir=lora_model_path,
                use_safetensors=True,
                adapter_name="adapter1",
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
        pipe_txt2img_sd.fuse_lora(lora_scale=lora_weight_txt2img_sd)
#        pipe_txt2img_sd.set_adapters(["adapter1"], adapter_weights=[float(lora_weight_txt2img_sd)])

    if txtinv_txt2img_sd != "":
        model_list_txtinv_txt2img_sd = txtinv_list(modelid_txt2img_sd)
        weight_txt2img_sd = model_list_txtinv_txt2img_sd[txtinv_txt2img_sd][0]
        token_txt2img_sd =  model_list_txtinv_txt2img_sd[txtinv_txt2img_sd][1]
        if modelid_txt2img_sd[0:9] == "./models/":
            model_path_txtinv = "./models/TextualInversion"
            pipe_txt2img_sd.load_textual_inversion(
                txtinv_txt2img_sd,
                weight_name=weight_txt2img_sd,
                use_safetensors=True,
                token=token_txt2img_sd,
            )
        else:
            if is_xl_txt2img_sd:
                model_path_txtinv = "./models/TextualInversion/SDXL"
            else: 
                model_path_txtinv = "./models/TextualInversion/SD"
            pipe_txt2img_sd.load_textual_inversion(
                txtinv_txt2img_sd,
                weight_name=weight_txt2img_sd,
                cache_dir=model_path_txtinv,
                use_safetensors=True,
                token=token_txt2img_sd,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )

    if seed_txt2img_sd == 0:
        random_seed = random.randrange(0, 10000000000, 1)
        final_seed = random_seed
    else:
        final_seed = seed_txt2img_sd
    generator = []
    for k in range(num_prompt_txt2img_sd):
        generator.append([torch.Generator(device_txt2img_sd).manual_seed(final_seed + (k*num_images_per_prompt_txt2img_sd) + l ) for l in range(num_images_per_prompt_txt2img_sd)])

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
            device=device_txt2img_sd,
        )
        conditioning, pooled = compel(prompt_txt2img_sd)
        neg_conditioning, neg_pooled = compel(negative_prompt_txt2img_sd)
        [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])
    else :
        compel = Compel(tokenizer=pipe_txt2img_sd.tokenizer, text_encoder=pipe_txt2img_sd.text_encoder, truncate_long_prompts=False, device=device_txt2img_sd)
        conditioning = compel.build_conditioning_tensor(prompt_txt2img_sd)
        neg_conditioning = compel.build_conditioning_tensor(negative_prompt_txt2img_sd)    
        [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])

    final_image = []
    final_seed = []
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
                generator = generator[i],
                callback_on_step_end=check_txt2img_sd, 
                callback_on_step_end_tensor_inputs=['latents'], 
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
                generator = generator[i],
                callback_on_step_end=check_txt2img_sd, 
                callback_on_step_end_tensor_inputs=['latents'], 
            ).images
        
        for j in range(len(image)):
            seed_id = random_seed + i*num_images_per_prompt_txt2img_sd + j if (seed_txt2img_sd == 0) else seed_txt2img_sd + i*num_images_per_prompt_txt2img_sd + j
            savename = name_seeded_image(seed_id)
            if use_gfpgan_txt2img_sd == True :
                image[j] = image_gfpgan_mini(image[j])
            image[j].save(savename)
            final_image.append(savename)
            final_seed.append(seed_id)

    print(f">>>[Stable Diffusion 🖼️ ]: generated {num_prompt_txt2img_sd} batch(es) of {num_images_per_prompt_txt2img_sd}")
    reporting_txt2img_sd = f">>>[Stable Diffusion 🖼️ ]: "+\
        f"Settings : Model={modelid_txt2img_sd} | "+\
        f"XL model={is_xl_txt2img_sd} | "+\
        f"Sampler={sampler_txt2img_sd} | "+\
        f"Steps={num_inference_step_txt2img_sd} | "+\
        f"CFG scale={guidance_scale_txt2img_sd} | "+\
        f"Size={width_txt2img_sd}x{height_txt2img_sd} | "+\
        f"GFPGAN={use_gfpgan_txt2img_sd} | "+\
        f"Token merging={tkme_txt2img_sd} | "+\
        f"LoRA model={lora_model_txt2img_sd} | "+\
        f"LoRA weight={lora_weight_txt2img_sd} | "+\
        f"Textual inversion={txtinv_txt2img_sd} | "+\
        f"nsfw_filter={bool(int(nsfw_filter))} | "+\
        f"Prompt={prompt_txt2img_sd} | "+\
        f"Negative prompt={negative_prompt_txt2img_sd} | "+\
        f"Seed List="+ ', '.join([f"{final_seed[m]}" for m in range(len(final_seed))])
    print(reporting_txt2img_sd) 

    exif_writer_png(reporting_txt2img_sd, final_image)

    del nsfw_filter_final, feat_ex, pipe_txt2img_sd, generator, compel, conditioning, neg_conditioning, image
    clean_ram()

    print(f">>>[Stable Diffusion 🖼️ ]: leaving module")
    return final_image, final_image
