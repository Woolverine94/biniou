# https://github.com/Woolverine94/biniou
# txt2img_lcm.py
import gradio as gr
import os
from diffusers import UNet2DConditionModel, DiffusionPipeline, AutoPipelineForText2Image
from huggingface_hub import hf_hub_download
from compel import Compel, ReturnedEmbeddingsType
import torch
import random
from ressources.scheduler import *
from ressources.gfpgan import *
import tomesd

device_label_txt2img_lcm, model_arch = detect_device()
device_txt2img_lcm = torch.device(device_label_txt2img_lcm)

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
    "segmind/Segmind-VegaRT",
    "latent-consistency/lcm-ssd-1b",
    "latent-consistency/lcm-sdxl",
    "latent-consistency/lcm-lora-sdv1-5",
    "latent-consistency/lcm-lora-sdxl",


]

for k in range(len(model_list_txt2img_lcm_builtin)):
    model_list_txt2img_lcm.append(model_list_txt2img_lcm_builtin[k])

# scheduler_list_txt2img_lcm = [
#     "LCMScheduler",
# ]

# Bouton Cancel
stop_txt2img_lcm = False

def initiate_stop_txt2img_lcm() :
    global stop_txt2img_lcm
    stop_txt2img_lcm = True

def check_txt2img_lcm(pipe, step_index, timestep, callback_kwargs) :
    global stop_txt2img_lcm
    if stop_txt2img_lcm == False :
#        result_preview = preview_image(step, timestep, latents, pipe_txt2img_lcm)
        return callback_kwargs
    elif stop_txt2img_lcm == True :
        print(">>>[LCM 🖼️ ]: generation canceled by user")
        stop_txt2img_lcm = False
        try:
            del ressources.txt2img_lcm.pipe_txt2img_lcm
        except NameError as e:
            raise Exception("Interrupting ...")
    return

@metrics_decoration
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
    lora_model_txt2img_lcm,
    lora_weight_txt2img_lcm,
    txtinv_txt2img_lcm,
    progress_txt2img_lcm=gr.Progress(track_tqdm=True)
    ):
    
    print(">>>[LCM 🖼️ ]: starting module")
    
    global pipe_txt2img_lcm
    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_txt2img_lcm_safetychecker, device_txt2img_lcm, nsfw_filter)

    if ("XL" in modelid_txt2img_lcm.upper()) or (modelid_txt2img_lcm == "latent-consistency/lcm-ssd-1b") or (modelid_txt2img_lcm == "segmind/Segmind-VegaRT"):
        is_xl_txt2img_lcm: bool = True
    else :        
        is_xl_txt2img_lcm: bool = False
        
    if (modelid_txt2img_lcm == "latent-consistency/lcm-ssd-1b") or (modelid_txt2img_lcm == "latent-consistency/lcm-sdxl"):
        model_path_SD_txt2img_lcm = "./models/Stable_Diffusion"
        if (modelid_txt2img_lcm == "latent-consistency/lcm-ssd-1b"):
            modelid_SD_txt2img_lcm = "segmind/SSD-1B"
        elif (modelid_txt2img_lcm == "latent-consistency/lcm-sdxl"):
            modelid_SD_txt2img_lcm = "stabilityai/stable-diffusion-xl-base-1.0"
        unet_txt2img_lcm = UNet2DConditionModel.from_pretrained(
            modelid_txt2img_lcm, 
            cache_dir=model_path_txt2img_lcm, 
            torch_dtype=model_arch, 
            use_safetensors=True, 
            safety_checker=None,
            feature_extractor=None,
            resume_download=True,
            local_files_only=True if offline_test() else None
            )
        pipe_txt2img_lcm = DiffusionPipeline.from_pretrained(
            modelid_SD_txt2img_lcm, 
            unet=unet_txt2img_lcm,
            cache_dir=model_path_SD_txt2img_lcm, 
            torch_dtype=model_arch, 
            use_safetensors=True, 
            safety_checker=None,
            feature_extractor=None,
            resume_download=True,
            local_files_only=True if offline_test() else None
        )
        pipe_txt2img_lcm = schedulerer(pipe_txt2img_lcm, sampler_txt2img_lcm)
#        pipe_txt2img_lcm.scheduler = LCMScheduler.from_config(pipe_txt2img_lcm.scheduler.config)
    elif (modelid_txt2img_lcm == "segmind/Segmind-VegaRT") or (modelid_txt2img_lcm == "latent-consistency/lcm-lora-sdv1-5") or (modelid_txt2img_lcm == "latent-consistency/lcm-lora-sdxl"):
        model_path_SD_txt2img_lcm = "./models/Stable_Diffusion"
        model_lora_txt2img_lcm = "pytorch_lora_weights.safetensors"
        if (modelid_txt2img_lcm == "segmind/Segmind-VegaRT"):
            modelid_SD_txt2img_lcm = "segmind/Segmind-Vega"
        elif (modelid_txt2img_lcm == "latent-consistency/lcm-lora-sdv1-5"):
            modelid_SD_txt2img_lcm = "SG161222/Realistic_Vision_V3.0_VAE"
        elif (modelid_txt2img_lcm == "latent-consistency/lcm-lora-sdxl"):
            modelid_SD_txt2img_lcm = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe_txt2img_lcm = AutoPipelineForText2Image.from_pretrained(
            modelid_SD_txt2img_lcm,
            cache_dir=model_path_SD_txt2img_lcm,
#            torch_dtype=torch.float32,
            torch_dtype=model_arch,
            use_safetensors=True,
            safety_checker=None,
            feature_extractor=None,
            resume_download=True,
            local_files_only=True if offline_test() else None
        )
        pipe_txt2img_lcm = schedulerer(pipe_txt2img_lcm, sampler_txt2img_lcm)
#        pipe_txt2img_lcm.scheduler = LCMScheduler.from_config(pipe_txt2img_lcm.scheduler.config)
        pipe_txt2img_lcm.load_lora_weights(
            modelid_txt2img_lcm,
            weight_name=model_lora_txt2img_lcm,
            cache_dir=model_path_txt2img_lcm,
            use_safetensors=True,
            resume_download=True,
            local_files_only=True if offline_test() else None
        )
        pipe_txt2img_lcm.fuse_lora()
    else : 
        if modelid_txt2img_lcm[0:9] == "./models/" :
            pipe_txt2img_lcm = DiffusionPipeline.from_single_file(
                modelid_txt2img_lcm, 
                torch_dtype=model_arch, 
                use_safetensors=True, 
                load_safety_checker=False if (nsfw_filter_final == None) else True,
                local_files_only=True if offline_test() else None,
#                safety_checker=None, 
#                feature_extractor=None,
            )
        else:
            pipe_txt2img_lcm = DiffusionPipeline.from_pretrained(
                modelid_txt2img_lcm, 
                cache_dir=model_path_txt2img_lcm, 
                torch_dtype=model_arch, 
                use_safetensors=True, 
                safety_checker=nsfw_filter_final,
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
        pipe_txt2img_lcm = schedulerer(pipe_txt2img_lcm, sampler_txt2img_lcm)
    pipe_txt2img_lcm.enable_attention_slicing("max")
    tomesd.apply_patch(pipe_txt2img_lcm, ratio=tkme_txt2img_lcm)
    if device_label_txt2img_lcm == "cuda":
        pipe_txt2img_lcm.enable_sequential_cpu_offload()
    else : 
        pipe_txt2img_lcm = pipe_txt2img_lcm.to(device_txt2img_lcm)
    pipe_txt2img_lcm.enable_vae_slicing()

    if lora_model_txt2img_lcm != "":
        model_list_lora_txt2img_lcm = lora_model_list(modelid_txt2img_lcm)
        if lora_model_txt2img_lcm[0:9] == "./models/":
            pipe_txt2img_lcm.load_lora_weights(
                os.path.dirname(lora_model_txt2img_lcm),
                weight_name=model_list_lora_txt2img_lcm[lora_model_txt2img_lcm][0],
                use_safetensors=True,
                adapter_name="adapter1",
                local_files_only=True if offline_test() else None
            )
        else:
            if is_xl_txt2img_lcm:
                lora_model_path = "./models/lora/SDXL"
            else: 
                lora_model_path = "./models/lora/SD"

            local_lora_txt2img_lcm = hf_hub_download(
                repo_id=lora_model_txt2img_lcm,
                filename=model_list_lora_txt2img_lcm[lora_model_txt2img_lcm][0],
                cache_dir=lora_model_path,
                resume_download=True,
                local_files_only=True if offline_test() else None,
            )

            pipe_txt2img_lcm.load_lora_weights(
                local_lora_txt2img_lcm,
                weight_name=model_list_lora_txt2img_lcm[lora_model_txt2img_lcm][0],
                use_safetensors=True,
                adapter_name="adapter1",
            )
        pipe_txt2img_lcm.fuse_lora(lora_scale=lora_weight_txt2img_lcm)
#            pipe_txt2img_lcm.set_adapters(["adapter1"], adapter_weights=[float(lora_weight_txt2img_lcm)])

    if txtinv_txt2img_lcm != "":
        model_list_txtinv_txt2img_lcm = txtinv_list(modelid_txt2img_lcm)
        weight_txt2img_lcm = model_list_txtinv_txt2img_lcm[txtinv_txt2img_lcm][0]
        token_txt2img_lcm =  model_list_txtinv_txt2img_lcm[txtinv_txt2img_lcm][1]
        if txtinv_txt2img_lcm[0:9] == "./models/":
            model_path_txtinv = "./models/TextualInversion"
            pipe_txt2img_lcm.load_textual_inversion(
                txtinv_txt2img_lcm,
                weight_name=weight_txt2img_lcm,
                use_safetensors=True,
                token=token_txt2img_lcm,
                local_files_only=True if offline_test() else None,
            )
        else:
            if is_xl_txt2img_lcm:
                model_path_txtinv = "./models/TextualInversion/SDXL"
            else: 
                model_path_txtinv = "./models/TextualInversion/SD"
            pipe_txt2img_lcm.load_textual_inversion(
                txtinv_txt2img_lcm,
                weight_name=weight_txt2img_lcm,
                cache_dir=model_path_txtinv,
                use_safetensors=True,
                token=token_txt2img_lcm,
                resume_download=True,
                local_files_only=True if offline_test() else None,
            )

    if seed_txt2img_lcm == 0:
        random_seed = random.randrange(0, 10000000000, 1)
        final_seed = random_seed
    else:
        final_seed = seed_txt2img_lcm
    generator = []
    for k in range(num_prompt_txt2img_lcm):
        generator.append([torch.Generator(device_txt2img_lcm).manual_seed(final_seed + (k*num_images_per_prompt_txt2img_lcm) + l ) for l in range(num_images_per_prompt_txt2img_lcm)])

    prompt_txt2img_lcm = str(prompt_txt2img_lcm)
    if prompt_txt2img_lcm == "None":
        prompt_txt2img_lcm = ""

    if (is_xl_txt2img_lcm == True) :
        compel = Compel(
            tokenizer=[pipe_txt2img_lcm.tokenizer, pipe_txt2img_lcm.tokenizer_2], 
            text_encoder=[pipe_txt2img_lcm.text_encoder, pipe_txt2img_lcm.text_encoder_2], 
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, 
            requires_pooled=[False, True], 
            device=device_txt2img_lcm,
        )
        conditioning, pooled = compel(prompt_txt2img_lcm)
    else :
        compel = Compel(tokenizer=pipe_txt2img_lcm.tokenizer, text_encoder=pipe_txt2img_lcm.text_encoder, truncate_long_prompts=False, device=device_txt2img_lcm)
        conditioning = compel.build_conditioning_tensor(prompt_txt2img_lcm)
   
    final_image = []
    final_seed = []
    for i in range (num_prompt_txt2img_lcm):
        if (is_xl_txt2img_lcm == True):
            image = pipe_txt2img_lcm(
                prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled,
                height=height_txt2img_lcm,
                width=width_txt2img_lcm,
                num_images_per_prompt=num_images_per_prompt_txt2img_lcm,
                num_inference_steps=num_inference_step_txt2img_lcm,
                guidance_scale=guidance_scale_txt2img_lcm,
                generator=generator[i],
                callback_on_step_end=check_txt2img_lcm,
                callback_on_step_end_tensor_inputs=['latents'],
#                lcm_origin_steps=lcm_origin_steps_txt2img_lcm,
            ).images

        else:
            image = pipe_txt2img_lcm(
                prompt_embeds=conditioning,
                height=height_txt2img_lcm,
                width=width_txt2img_lcm,
                num_images_per_prompt=num_images_per_prompt_txt2img_lcm,
                num_inference_steps=num_inference_step_txt2img_lcm,
                guidance_scale=guidance_scale_txt2img_lcm,
                generator=generator[i],
                callback_on_step_end=check_txt2img_lcm, 
                callback_on_step_end_tensor_inputs=['latents'], 
#                lcm_origin_steps=lcm_origin_steps_txt2img_lcm,
            ).images			

        for j in range(len(image)):
            if is_xl_txt2img_lcm or (modelid_txt2img_lcm == "latent-consistency/lcm-lora-sdv1-5"):
                image[j] = safety_checker_sdxl(model_path_txt2img_lcm_safetychecker, image[j], nsfw_filter)
            seed_id = random_seed + i*num_images_per_prompt_txt2img_lcm + j if (seed_txt2img_lcm == 0) else seed_txt2img_lcm + i*num_images_per_prompt_txt2img_lcm + j
            savename = name_seeded_image(seed_id)
            if use_gfpgan_txt2img_lcm == True :
                image[j] = image_gfpgan_mini(image[j])
            image[j].save(savename)
            final_image.append(savename)
            final_seed.append(seed_id)

    print(f">>>[LCM 🖼️ ]: generated {num_prompt_txt2img_lcm} batch(es) of {num_images_per_prompt_txt2img_lcm}")
    reporting_txt2img_lcm = f">>>[LCM 🖼️ ]: "+\
        f"Settings : Model={modelid_txt2img_lcm} | "+\
        f"Steps={num_inference_step_txt2img_lcm} | "+\
        f"CFG scale={guidance_scale_txt2img_lcm} | "+\
        f"Size={width_txt2img_lcm}x{height_txt2img_lcm} | "+\
        f"GFPGAN={use_gfpgan_txt2img_lcm} | "+\
        f"LoRA model={lora_model_txt2img_lcm} | "+\
        f"LoRA weight={lora_weight_txt2img_lcm} | "+\
        f"Textual inversion={txtinv_txt2img_lcm} | "+\
        f"nsfw_filter={bool(int(nsfw_filter))} | "+\
        f"Prompt={prompt_txt2img_lcm} | "+\
        f"Seed List="+ ', '.join([f"{final_seed[m]}" for m in range(len(final_seed))])
    print(reporting_txt2img_lcm) 

    exif_writer_png(reporting_txt2img_lcm, final_image)

    del nsfw_filter_final, feat_ex, pipe_txt2img_lcm, generator, compel, conditioning, image
    clean_ram()

    print(f">>>[LCM 🖼️ ]: leaving module")
    
    return final_image, final_image
