# https://github.com/Woolverine94/biniou
# img2img_ip.py
import gradio as gr
import os
import PIL
import torch
from diffusers import AutoPipelineForImage2Image, StableDiffusionXLImg2ImgPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image
from transformers import CLIPVisionModelWithProjection
from huggingface_hub import snapshot_download, hf_hub_download
from compel import Compel, ReturnedEmbeddingsType
import random
from ressources.common import *
from ressources.gfpgan import *
import tomesd
from diffusers.schedulers import AysSchedules

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
#     "SG161222/Realistic_Vision_V3.0_VAE",
#     "SG161222/Paragon_V1.0",
#     "digiplay/majicMIX_realistic_v7",
#     "SPO-Diffusion-Models/SPO-SD-v1-5_4k-p_10ep",
#     "sd-community/sdxl-flash",
#     "dataautogpt3/PrometheusV1",
#     "mann-e/Mann-E_Dreams",
#     "mann-e/Mann-E_Art",
#     "ehristoforu/Visionix-alpha",
#     "RunDiffusion/Juggernaut-X-Hyper",
#     "cutycat2000x/InterDiffusion-4.0",
#     "RunDiffusion/Juggernaut-XL-Lightning",
#     "fluently/Fluently-XL-v3-Lightning",
#     "Corcelio/mobius",
#     "fluently/Fluently-XL-Final",
#     "SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep",
#     "recoilme/ColorfulXL-Lightning",
#     "playgroundai/playground-v2-512px-base",
#     "playgroundai/playground-v2-1024px-aesthetic",
#     "playgroundai/playground-v2.5-1024px-aesthetic",
# #    "SG161222/RealVisXL_V3.0",
# #    "stabilityai/sd-turbo",
#     "stabilityai/sdxl-turbo",
# #    "thibaud/sdxl_dpo_turbo",
#     "SG161222/RealVisXL_V4.0_Lightning",
#     "cagliostrolab/animagine-xl-3.1",
#     "aipicasso/emi-2",
#     "dataautogpt3/OpenDalleV1.1",
#     "dataautogpt3/ProteusV0.5",
#     "dataautogpt3/ProteusV0.4-Lightning",
#     "digiplay/AbsoluteReality_v1.8.1",
# #    "segmind/Segmind-Vega",
# #    "segmind/SSD-1B",
#     "gsdf/Counterfeit-V2.5",
# #    "ckpt/anything-v4.5-vae-swapped",
#     "stabilityai/stable-diffusion-xl-base-1.0",
# #    "stabilityai/stable-diffusion-xl-refiner-1.0",
#     "runwayml/stable-diffusion-v1-5",
#     "nitrosocke/Ghibli-Diffusion",

    "-[ 👍 SD15 ]-",
    "SG161222/Realistic_Vision_V3.0_VAE",
    "SG161222/Paragon_V1.0",
    "digiplay/AbsoluteReality_v1.8.1",
    "digiplay/majicMIX_realistic_v7",
    "SPO-Diffusion-Models/SPO-SD-v1-5_4k-p_10ep",
    "runwayml/stable-diffusion-v1-5",
    "-[ 👍 🇯🇵 Anime SD15 ]-",
    "gsdf/Counterfeit-V2.5",
    "nitrosocke/Ghibli-Diffusion",
    "-[ 👌 🐢 SDXL ]-",
    "fluently/Fluently-XL-Final",
    "Corcelio/mobius",
    "mann-e/Mann-E_Dreams",
    "mann-e/Mann-E_Art",
    "ehristoforu/Visionix-alpha",
    "cutycat2000x/InterDiffusion-4.0",
    "SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep",
    "dataautogpt3/ProteusV0.5",
    "dataautogpt3/PrometheusV1",
    "dataautogpt3/OpenDalleV1.1",
    "playgroundai/playground-v2-512px-base",
    "playgroundai/playground-v2-1024px-aesthetic",
    "playgroundai/playground-v2.5-1024px-aesthetic",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "-[ 👌 🚀 Fast SDXL ]-",
    "sd-community/sdxl-flash",
    "fluently/Fluently-XL-v3-Lightning",
    "RunDiffusion/Juggernaut-XL-Lightning",
    "RunDiffusion/Juggernaut-X-Hyper",
    "SG161222/RealVisXL_V4.0_Lightning",
    "dataautogpt3/ProteusV0.4-Lightning",
    "recoilme/ColorfulXL-Lightning",
    "stabilityai/sdxl-turbo",
    "-[ 👌 🇯🇵 Anime SDXL ]-",
    "cagliostrolab/animagine-xl-3.1",
    "aipicasso/emi-2",
]

for k in range(len(model_list_img2img_ip_builtin)):
    model_list_img2img_ip.append(model_list_img2img_ip_builtin[k])

# Bouton Cancel
stop_img2img_ip = False

def initiate_stop_img2img_ip() :
    global stop_img2img_ip
    stop_img2img_ip = True

def check_img2img_ip(pipe, step_index, timestep, callback_kwargs): 
    global stop_img2img_ip
    if stop_img2img_ip == True:
        print(">>>[IP-Adapter 🖌️ ]: generation canceled by user")
        stop_img2img_ip = False
        pipe._interrupt = True
    return callback_kwargs

@metrics_decoration
def image_img2img_ip(
    modelid_img2img_ip, 
    sampler_img2img_ip, 
    img_img2img_ip, 
    source_type_img2img_ip,
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
    clipskip_img2img_ip,
    use_ays_img2img_ip,
    lora_model_img2img_ip,
    lora_weight_img2img_ip,
    txtinv_img2img_ip,
    progress_img2img_ip=gr.Progress(track_tqdm=True)
    ):

    print(">>>[IP-Adapter 🖌️ ]: starting module")

    modelid_img2img_ip = model_cleaner_sd(modelid_img2img_ip)

    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_img2img_ip, device_img2img_ip, nsfw_filter)

    if clipskip_img2img_ip == 0:
       clipskip_img2img_ip = None

    if ("turbo" in modelid_img2img_ip):
        is_turbo_img2img_ip: bool = True
    else :
        is_turbo_img2img_ip: bool = False

    if is_sdxl(modelid_img2img_ip):
        is_xl_img2img_ip: bool = True
    else :
        is_xl_img2img_ip: bool = False     

    if is_bin(modelid_img2img_ip):
        is_bin_img2img_ip: bool = True
    else :
        is_bin_img2img_ip: bool = False

    if (num_inference_step_img2img_ip >= 10) and use_ays_img2img_ip:
        if is_sdxl(modelid_img2img_ip):
            sampling_schedule_img2img_ip = AysSchedules["StableDiffusionXLTimesteps"]
            sampler_img2img_ip = "DPM++ SDE"
        else:
            sampling_schedule_img2img_ip = AysSchedules["StableDiffusionTimesteps"]
            sampler_img2img_ip = "Euler"
        num_inference_step_img2img_ip = 10
    else:
        sampling_schedule_img2img_ip = None

    if which_os() == "win32" or source_type_img2img_ip == "composition":
        if (is_xl_img2img_ip == True):
            hf_hub_download(
                repo_id="h94/IP-Adapter",
                filename="sdxl_models/image_encoder/config.json",
                repo_type="model",
                local_dir=model_path_ipa_img2img_ip,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
            hf_hub_download(
                repo_id="h94/IP-Adapter",
                filename="sdxl_models/image_encoder/model.safetensors",
                repo_type="model",
                local_dir=model_path_ipa_img2img_ip,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
            if (source_type_img2img_ip == "standard"):
                hf_hub_download(
                    repo_id="h94/IP-Adapter",
                    filename="sdxl_models/ip-adapter_sdxl.safetensors",
                    repo_type="model",
                    local_dir=model_path_ipa_img2img_ip,
                    resume_download=True,
                    local_files_only=True if offline_test() else None
                )
            elif (source_type_img2img_ip == "composition"):
                if not os.path.isfile(f"{model_path_ipa_img2img_ip}/sdxl_models/ip_plus_composition_sdxl.safetensors"):
                    hf_hub_download(
                        repo_id="ostris/ip-composition-adapter",
                        filename="ip_plus_composition_sdxl.safetensors",
                        repo_type="model",
                        local_dir=model_path_ipa_img2img_ip+ "/sdxl_models",
                        resume_download=True,
                        local_files_only=True if offline_test() else None
                    )

                image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    "h94/IP-Adapter",
                    subfolder="models/image_encoder",
                    cache_dir=model_path_ipa_img2img_ip,
                    torch_dtype=model_arch,
                    use_safetensors=True,
                    resume_download=True,
                    local_files_only=True if offline_test() else None
                )

        else:
            hf_hub_download(
                repo_id="h94/IP-Adapter",
                filename="models/image_encoder/config.json",
                repo_type="model",
                local_dir=model_path_ipa_img2img_ip,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
            hf_hub_download(
                repo_id="h94/IP-Adapter",
                filename="models/image_encoder/model.safetensors",
                repo_type="model",
                local_dir=model_path_ipa_img2img_ip,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
            if (source_type_img2img_ip == "standard"):
                hf_hub_download(
                    repo_id="h94/IP-Adapter",
                    filename="models/ip-adapter_sd15.safetensors",
                    repo_type="model",
                    local_dir=model_path_ipa_img2img_ip,
                    resume_download=True,
                    local_files_only=True if offline_test() else None
                )
            elif (source_type_img2img_ip == "composition"):
                if not os.path.isfile(f"{model_path_ipa_img2img_ip}/models/ip_plus_composition_sd15.safetensors"):
                    hf_hub_download(
                        repo_id="ostris/ip-composition-adapter",
                        filename="ip_plus_composition_sd15.safetensors",
                        repo_type="model",
                        local_dir=model_path_ipa_img2img_ip+ "/models",
                        resume_download=True,
                        local_files_only=True if offline_test() else None
                    )

    if (is_xl_img2img_ip == True):
        if (source_type_img2img_ip == "standard"):
            if modelid_img2img_ip[0:9] == "./models/" :
                pipe_img2img_ip = StableDiffusionXLImg2ImgPipeline.from_single_file(
                    modelid_img2img_ip,
                    torch_dtype=model_arch,
                    use_safetensors=True if not is_bin_img2img_ip else False,
#                    load_safety_checker=False if (nsfw_filter_final == None) else True,
                    local_files_only=True if offline_test() else None
#                    safety_checker=nsfw_filter_final, 
#                    feature_extractor=feat_ex,
                )
            else :
                pipe_img2img_ip = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    modelid_img2img_ip,
                    cache_dir=model_path_img2img_ip,
                    torch_dtype=model_arch,
                    use_safetensors=True if not is_bin_img2img_ip else False,
                    resume_download=True,
                    local_files_only=True if offline_test() else None
                )
        elif (source_type_img2img_ip == "composition"):
            if modelid_img2img_ip[0:9] == "./models/" :
                pipe_img2img_ip = StableDiffusionXLPipeline.from_single_file(
                    modelid_img2img_ip,
                    torch_dtype=model_arch,
                    use_safetensors=True if not is_bin_img2img_ip else False,
#                    load_safety_checker=False if (nsfw_filter_final == None) else True,
                    local_files_only=True if offline_test() else None
#                    safety_checker=nsfw_filter_final, 
#                    feature_extractor=feat_ex,
                )
            else :
                pipe_img2img_ip = StableDiffusionXLPipeline.from_pretrained(
                    modelid_img2img_ip,
                    cache_dir=model_path_img2img_ip,
                    torch_dtype=model_arch,
                    use_safetensors=True if not is_bin_img2img_ip else False,
                    image_encoder=image_encoder,
                    resume_download=True,
                    local_files_only=True if offline_test() else None
                )
    else:
        if (source_type_img2img_ip == "standard"):
            if modelid_img2img_ip[0:9] == "./models/" :
                pipe_img2img_ip = StableDiffusionImg2ImgPipeline.from_single_file(
                    modelid_img2img_ip,
                    torch_dtype=model_arch,
                    use_safetensors=True if not is_bin_img2img_ip else False,
                    load_safety_checker=False if (nsfw_filter_final == None) else True,
                    local_files_only=True if offline_test() else None
#                    safety_checker=nsfw_filter_final, 
#                    feature_extractor=feat_ex,
                )
            else :
                pipe_img2img_ip = StableDiffusionImg2ImgPipeline.from_pretrained(
                    modelid_img2img_ip,
                    cache_dir=model_path_img2img_ip,
                    torch_dtype=model_arch,
                    use_safetensors=True if not is_bin_img2img_ip else False,
                    safety_checker=nsfw_filter_final,
                    feature_extractor=feat_ex,
                    resume_download=True,
                    local_files_only=True if offline_test() else None
                )
        elif (source_type_img2img_ip == "composition"):
            if modelid_img2img_ip[0:9] == "./models/" :
                pipe_img2img_ip = StableDiffusionPipeline.from_single_file(
                    modelid_img2img_ip,
                    torch_dtype=model_arch,
                    use_safetensors=True if not is_bin_img2img_ip else False,
                    load_safety_checker=False if (nsfw_filter_final == None) else True,
                    local_files_only=True if offline_test() else None
#                    safety_checker=nsfw_filter_final, 
#                    feature_extractor=feat_ex,
                )
            else :
                pipe_img2img_ip = StableDiffusionPipeline.from_pretrained(
                    modelid_img2img_ip,
                    cache_dir=model_path_img2img_ip,
                    torch_dtype=model_arch,
                    use_safetensors=True if not is_bin_img2img_ip else False,
                    safety_checker=nsfw_filter_final,
                    feature_extractor=feat_ex,
                    resume_download=True,
                    local_files_only=True if offline_test() else None
                )

#    if (is_xl_img2img_ip == True) or (is_turbo_img2img_ip == True):
    if (which_os() == "win32"):
        if (is_xl_img2img_ip == True):
            if (source_type_img2img_ip == "standard"):
                pipe_img2img_ip.load_ip_adapter(
                    model_path_ipa_img2img_ip,
                    subfolder="sdxl_models",
                    weight_name="ip-adapter_sdxl.safetensors",
                    torch_dtype=model_arch,
                    use_safetensors=True,
                    resume_download=True,
                    local_files_only=True if offline_test() else None
                )
            elif (source_type_img2img_ip == "composition"):
                pipe_img2img_ip.load_ip_adapter(
                    model_path_ipa_img2img_ip,
                    subfolder="sdxl_models",
                    weight_name="ip_plus_composition_sdxl.safetensors",
                    torch_dtype=model_arch,
                    use_safetensors=True, 
                    resume_download=True,
                    local_files_only=True if offline_test() else None
                )
        else:
            if (source_type_img2img_ip == "standard"):
                pipe_img2img_ip.load_ip_adapter(
                    model_path_ipa_img2img_ip,
                    subfolder="models",
                    weight_name="ip-adapter_sd15.safetensors",
                    torch_dtype=model_arch,
                    use_safetensors=True, 
                    resume_download=True,
                    local_files_only=True if offline_test() else None
                )
            elif (source_type_img2img_ip == "composition"):
                pipe_img2img_ip.load_ip_adapter(
                    model_path_ipa_img2img_ip,
                    subfolder="models",
                    weight_name="ip_plus_composition_sd15.safetensors",
                    torch_dtype=model_arch,
                    use_safetensors=True, 
                    resume_download=True,
                    local_files_only=True if offline_test() else None
                )
    else:
        if (is_xl_img2img_ip == True):
            if (source_type_img2img_ip == "standard"):
                pipe_img2img_ip.load_ip_adapter(
                    "h94/IP-Adapter", 
                    cache_dir=model_path_ipa_img2img_ip,
                    subfolder="sdxl_models",
                    weight_name="ip-adapter_sdxl.safetensors",
                    torch_dtype=model_arch,
                    use_safetensors=True,
                    resume_download=True,
                    local_files_only=True if offline_test() else None
                )
            elif (source_type_img2img_ip == "composition"):
                pipe_img2img_ip.load_ip_adapter(
                    model_path_ipa_img2img_ip,
                    subfolder="sdxl_models",
                    weight_name="ip_plus_composition_sdxl.safetensors",
                    torch_dtype=model_arch,
                    use_safetensors=True, 
                    resume_download=True,
                    local_files_only=True if offline_test() else None
                )
        else:
            if (source_type_img2img_ip == "standard"):
                pipe_img2img_ip.load_ip_adapter(
                    "h94/IP-Adapter",
                    cache_dir=model_path_ipa_img2img_ip,
                    subfolder="models",
                    weight_name="ip-adapter_sd15.safetensors",
                    torch_dtype=model_arch,
                    use_safetensors=True, 
                    resume_download=True,
                    local_files_only=True if offline_test() else None
                )
            elif (source_type_img2img_ip == "composition"):
                pipe_img2img_ip.load_ip_adapter(
                    model_path_ipa_img2img_ip,
                    subfolder="models",
                    weight_name="ip_plus_composition_sd15.safetensors",
                    torch_dtype=model_arch,
                    use_safetensors=True, 
                    resume_download=True,
                    local_files_only=True if offline_test() else None
                )

#    pipe_img2img_ip.set_ip_adapter_scale(denoising_strength_img2img_ip)    
    pipe_img2img_ip = schedulerer(pipe_img2img_ip, sampler_img2img_ip)
#    pipe_img2img_ip.enable_attention_slicing("max")  
    tomesd.apply_patch(pipe_img2img_ip, ratio=tkme_img2img_ip)
    if device_label_img2img_ip == "cuda" :
        pipe_img2img_ip.enable_sequential_cpu_offload()
    else :
        pipe_img2img_ip = pipe_img2img_ip.to(device_img2img_ip)

    if lora_model_img2img_ip != "":
        model_list_lora_img2img_ip = lora_model_list(modelid_img2img_ip)
        if lora_model_img2img_ip[0:9] == "./models/":
            pipe_img2img_ip.load_lora_weights(
                os.path.dirname(lora_model_img2img_ip),
                weight_name=model_list_lora_img2img_ip[lora_model_img2img_ip][0],
                use_safetensors=True,
                adapter_name="adapter1",
                local_files_only=True if offline_test() else None,
            )
        else:
            if is_xl_img2img_ip:
                lora_model_path = "./models/lora/SDXL"
            else: 
                lora_model_path = "./models/lora/SD"

            local_lora_img2img_ip = hf_hub_download(
                repo_id=lora_model_img2img_ip,
                filename=model_list_lora_img2img_ip[lora_model_img2img_ip][0],
                cache_dir=lora_model_path,
                resume_download=True,
                local_files_only=True if offline_test() else None,
            )

            pipe_img2img_ip.load_lora_weights(
                local_lora_img2img_ip,
                weight_name=model_list_lora_img2img_ip[lora_model_img2img_ip][0],
                use_safetensors=True,
                adapter_name="adapter1",
            )
        pipe_img2img_ip.fuse_lora(lora_scale=lora_weight_img2img_ip)
#        pipe_img2img_ip.set_adapters(["adapter1"], adapter_weights=[float(lora_weight_img2img_ip)])

    if txtinv_img2img_ip != "":
        model_list_txtinv_img2img_ip = txtinv_list(modelid_img2img_ip)
        weight_img2img_ip = model_list_txtinv_img2img_ip[txtinv_img2img_ip][0]
        token_img2img_ip =  model_list_txtinv_img2img_ip[txtinv_img2img_ip][1]
        if txtinv_img2img_ip[0:9] == "./models/":
            model_path_txtinv = "./models/TextualInversion"
            pipe_img2img_ip.load_textual_inversion(
                txtinv_img2img_ip,
                weight_name=weight_img2img_ip,
                use_safetensors=True,
                token=token_img2img_ip,
                local_files_only=True if offline_test() else None,
            )
        else:
            if is_xl_img2img_ip:
                model_path_txtinv = "./models/TextualInversion/SDXL"
            else: 
                model_path_txtinv = "./models/TextualInversion/SD"
            pipe_img2img_ip.load_textual_inversion(
                txtinv_img2img_ip,
                weight_name=weight_img2img_ip,
                cache_dir=model_path_txtinv,
                use_safetensors=True,
                token=token_img2img_ip,
                resume_download=True,
                local_files_only=True if offline_test() else None,
            )

    if seed_img2img_ip == 0:
        random_seed = torch.randint(0, 10000000000, (1,))
        generator = torch.manual_seed(random_seed)
    else:
        generator = torch.manual_seed(seed_img2img_ip)

    if (img_img2img_ip != None):
        if (is_xl_img2img_ip == True) and not (is_turbo_img2img_ip == True):
            dim_size = correct_size(width_img2img_ip, height_img2img_ip, 1024)
        else: 
            dim_size = correct_size(width_img2img_ip, height_img2img_ip, 512)
        image_input = PIL.Image.open(img_img2img_ip)
        image_input = image_input.convert("RGB")
        image_input = image_input.resize((dim_size[0], dim_size[1]))
    else:
        image_input = None

    if (img_ipa_img2img_ip != None):
        image_input_ipa = PIL.Image.open(img_ipa_img2img_ip)
        if (is_xl_img2img_ip == True) and not (is_turbo_img2img_ip == True):
            dim_size_ipa = correct_size(image_input_ipa.size[0], image_input_ipa.size[1], 1024)
        else:
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
            device=device_img2img_ip,
        )
        conditioning, pooled = compel(prompt_img2img_ip)
        neg_conditioning, neg_pooled = compel(negative_prompt_img2img_ip)
        [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])
    else :
        compel = Compel(tokenizer=pipe_img2img_ip.tokenizer, text_encoder=pipe_img2img_ip.text_encoder, truncate_long_prompts=False, device=device_img2img_ip)
        conditioning = compel.build_conditioning_tensor(prompt_img2img_ip)
        neg_conditioning = compel.build_conditioning_tensor(negative_prompt_img2img_ip)
        [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])
    
    final_image = []

    for i in range (num_prompt_img2img_ip):
        if (is_turbo_img2img_ip == True) :
            if (source_type_img2img_ip == "standard"):
                image = pipe_img2img_ip(        
                    image=image_input,
                    ip_adapter_image=image_input_ipa,
                    prompt=prompt_img2img_ip,
                    num_images_per_prompt=num_images_per_prompt_img2img_ip,
                    guidance_scale=guidance_scale_img2img_ip,
                    strength=denoising_strength_img2img_ip,
                    num_inference_steps=num_inference_step_img2img_ip,
                    timesteps=sampling_schedule_img2img_ip,
                    generator = generator,
                    callback_on_step_end=check_img2img_ip, 
                    callback_on_step_end_tensor_inputs=['latents'], 
                ).images
            elif (source_type_img2img_ip == "composition"):
                image = pipe_img2img_ip(        
#                    image=image_input,
                    ip_adapter_image=image_input_ipa,
                    prompt=prompt_img2img_ip,
                    num_images_per_prompt=num_images_per_prompt_img2img_ip,
                    guidance_scale=guidance_scale_img2img_ip,
                    strength=denoising_strength_img2img_ip,
                    num_inference_steps=num_inference_step_img2img_ip,
                    timesteps=sampling_schedule_img2img_ip,
                    generator = generator,
                    callback_on_step_end=check_img2img_ip, 
                    callback_on_step_end_tensor_inputs=['latents'], 
                ).images
        elif (is_xl_img2img_ip == True) :
            if (source_type_img2img_ip == "standard"):
                image = pipe_img2img_ip(        
                    image=image_input,
                    ip_adapter_image=image_input_ipa,
                    prompt=prompt_img2img_ip,
                    negative_prompt=negative_prompt_img2img_ip,
#                    prompt_embeds=conditioning,
#                    pooled_prompt_embeds=pooled,
#                    negative_prompt_embeds=neg_conditioning,
#                    negative_pooled_prompt_embeds=neg_pooled,
                    num_images_per_prompt=num_images_per_prompt_img2img_ip,
                    guidance_scale=guidance_scale_img2img_ip,
                    strength=denoising_strength_img2img_ip,
                    num_inference_steps=num_inference_step_img2img_ip,
                    timesteps=sampling_schedule_img2img_ip,
                    generator = generator,
                    callback_on_step_end=check_img2img_ip, 
                    callback_on_step_end_tensor_inputs=['latents'], 
                ).images            
            if (source_type_img2img_ip == "composition"):
                image = pipe_img2img_ip(        
                    ip_adapter_image=image_input_ipa,
                    prompt=prompt_img2img_ip,
                    negative_prompt=negative_prompt_img2img_ip,
#                    prompt_embeds=conditioning,
#                    pooled_prompt_embeds=pooled,
#                    negative_prompt_embeds=neg_conditioning,
#                    negative_pooled_prompt_embeds=neg_pooled,
                    num_images_per_prompt=num_images_per_prompt_img2img_ip,
                    guidance_scale=guidance_scale_img2img_ip,
                    strength=denoising_strength_img2img_ip,
                    num_inference_steps=num_inference_step_img2img_ip,
                    timesteps=sampling_schedule_img2img_ip,
                    generator = generator,
                    callback_on_step_end=check_img2img_ip, 
                    callback_on_step_end_tensor_inputs=['latents'], 
                ).images   
        else : 
            if (source_type_img2img_ip == "standard"):
                image = pipe_img2img_ip(        
                    image=image_input,
                    ip_adapter_image=image_input_ipa,
                    prompt_embeds=conditioning,
                    negative_prompt_embeds=neg_conditioning,
                    num_images_per_prompt=num_images_per_prompt_img2img_ip,
                    guidance_scale=guidance_scale_img2img_ip,
                    strength=denoising_strength_img2img_ip,
                    num_inference_steps=num_inference_step_img2img_ip,
                    timesteps=sampling_schedule_img2img_ip,
                    generator=generator,
                    clip_skip=clipskip_img2img_ip,
                    callback_on_step_end=check_img2img_ip, 
                    callback_on_step_end_tensor_inputs=['latents'], 
                ).images        
            elif (source_type_img2img_ip == "composition"):
                image = pipe_img2img_ip(        
                    ip_adapter_image=image_input_ipa,
                    prompt_embeds=conditioning,
                    negative_prompt_embeds=neg_conditioning,
                    num_images_per_prompt=num_images_per_prompt_img2img_ip,
                    guidance_scale=guidance_scale_img2img_ip,
                    strength=denoising_strength_img2img_ip,
                    num_inference_steps=num_inference_step_img2img_ip,
                    timesteps=sampling_schedule_img2img_ip,
                    generator=generator,
                    clip_skip=clipskip_img2img_ip,
                    callback_on_step_end=check_img2img_ip, 
                    callback_on_step_end_tensor_inputs=['latents'], 
                ).images        

        for j in range(len(image)):
            if is_xl_img2img_ip:
                image[j] = safety_checker_sdxl(model_path_img2img_ip, image[j], nsfw_filter)
            savename = name_image()
            if use_gfpgan_img2img_ip == True :
                image[j] = image_gfpgan_mini(image[j])
            image[j].save(savename)
            final_image.append(savename)

    print(f">>>[IP-Adapter 🖌️ ]: generated {num_prompt_img2img_ip} batch(es) of {num_images_per_prompt_img2img_ip}")
    reporting_img2img_ip = f">>>[IP-Adapter 🖌️ ]: "+\
        f"Settings : Model={modelid_img2img_ip} | "+\
        f"XL model={is_xl_img2img_ip} | "+\
        f"IP-Adapter type={source_type_img2img_ip} | "+\
        f"Sampler={sampler_img2img_ip} | "+\
        f"Steps={num_inference_step_img2img_ip} | "+\
        f"CFG scale={guidance_scale_img2img_ip} | "+\
        f"Size={width_img2img_ip}x{height_img2img_ip} | "+\
        f"GFPGAN={use_gfpgan_img2img_ip} | "+\
        f"Token merging={tkme_img2img_ip} | "+\
        f"CLIP skip={clipskip_img2img_ip} | "+\
        f"AYS={use_ays_img2img_ip} | "+\
        f"LoRA model={lora_model_img2img_ip} | "+\
        f"LoRA weight={lora_weight_img2img_ip} | "+\
        f"Textual inversion={txtinv_img2img_ip} | "+\
        f"nsfw_filter={bool(int(nsfw_filter))} | "+\
        f"Denoising strength={denoising_strength_img2img_ip} | "+\
        f"Prompt={prompt_img2img_ip} | "+\
        f"Negative prompt={negative_prompt_img2img_ip}"
    print(reporting_img2img_ip)         

    exif_writer_png(reporting_img2img_ip, final_image)

    del nsfw_filter_final, feat_ex, pipe_img2img_ip, generator, image_input, image_input_ipa, compel, conditioning, neg_conditioning, image
    clean_ram()

    print(f">>>[IP-Adapter 🖌️ ]: leaving module")
    return final_image, final_image 
