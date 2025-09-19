# https://github.com/Woolverine94/biniou
# img2img_ip.py
import gradio as gr
import os
import PIL
import torch
from diffusers import AutoPipelineForImage2Image, StableDiffusionXLImg2ImgPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image, FluxImg2ImgPipeline, FluxPipeline
from transformers import CLIPVisionModelWithProjection, CLIPModel
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
model_path_ipa_img2img_ip = "./models/Ip-Adapters/"
model_path_flux_img2img_ip = "./models/Flux/"
os.makedirs(model_path_img2img_ip, exist_ok=True)
os.makedirs(model_path_ipa_img2img_ip, exist_ok=True)
os.makedirs(model_path_flux_img2img_ip, exist_ok=True)

model_list_img2img_ip_local = []

for filename in os.listdir(model_path_img2img_ip):
    f = os.path.join(model_path_img2img_ip, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
        model_list_img2img_ip_local.append(f)

model_list_img2img_ip_builtin = [
    "-[ 👍 SD15 ]-",
    "SG161222/Realistic_Vision_V3.0_VAE",
    "Yntec/VisionVision",
    "fluently/Fluently-epic",
    "SG161222/Paragon_V1.0",
    "digiplay/AbsoluteReality_v1.8.1",
    "digiplay/majicMIX_realistic_v7",
    "SPO-Diffusion-Models/SPO-SD-v1-5_4k-p_10ep",
    "digiplay/PerfectDeliberate_v5",
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "ItsJayQz/GTA5_Artwork_Diffusion",
    "songkey/epicphotogasm_ultimateFidelity",
    "-[ 👍 🇯🇵 Anime SD15 ]-",
    "gsdf/Counterfeit-V2.5",
    "fluently/Fluently-anime",
    "xyn-ai/anything-v4.0",
    "nitrosocke/Ghibli-Diffusion",
    "digiplay/STRANGER-ANIME",
    "Norod78/sd15-jojo-stone-ocean",
    "stablediffusionapi/anything-v5",
    "-[ 👌 🐢 SDXL ]-",
    "fluently/Fluently-XL-Final",
    "SG161222/RealVisXL_V5.0",
    "Corcelio/mobius",
    "misri/juggernautXL_juggXIByRundiffusion",
    "mann-e/Mann-E_Dreams",
    "mann-e/Mann-E_Art",
    "ehristoforu/Visionix-alpha",
    "cutycat2000x/InterDiffusion-4.0",
    "SPO-Diffusion-Models/SPO-SDXL_4k-p_10ep",
    "GraydientPlatformAPI/flashback-xl",
    "dataautogpt3/ProteusV0.5",
    "dataautogpt3/Proteus-v0.6",
    "dataautogpt3/PrometheusV1",
    "dataautogpt3/OpenDalleV1.1",
    "dataautogpt3/ProteusSigma",
    "Chan-Y/Stable-Flash-Lightning",
    "stablediffusionapi/protovision-xl-high-fidel",
    "comin/IterComp",
    "Spestly/OdysseyXL-1.0",
    "eramth/realism-sdxl",
    "yandex/stable-diffusion-xl-base-1.0-alchemist",
    "John6666/stellaratormix-photorealism-v30-sdxl",
    "RunDiffusion/Juggernaut-XL-v6",
    "playgroundai/playground-v2-512px-base",
    "playgroundai/playground-v2-1024px-aesthetic",
    "playgroundai/playground-v2.5-1024px-aesthetic",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "-[ 👌 🚀 Fast SDXL ]-",
    "sd-community/sdxl-flash",
    "fluently/Fluently-XL-v3-Lightning",
    "GraydientPlatformAPI/epicrealism-lightning-xl",
    "Lykon/dreamshaper-xl-lightning",
    "RunDiffusion/Juggernaut-XL-Lightning",
    "RunDiffusion/Juggernaut-X-Hyper",
    "SG161222/RealVisXL_V5.0_Lightning",
    "dataautogpt3/ProteusV0.4-Lightning",
    "recoilme/ColorfulXL-Lightning",
    "GraydientPlatformAPI/lustify-lightning",
    "John6666/comradeship-xl-v9a-spo-dpo-flash-sdxl",
    "stablediffusionapi/dream-diffusion-lightning",
    "John6666/jib-mix-realistic-xl-v15-maximus-sdxl",
    "stabilityai/sdxl-turbo",
    "-[ 👌 🇯🇵 Anime SDXL ]-",
    "GraydientPlatformAPI/geekpower-cellshade-xl",
    "cagliostrolab/animagine-xl-4.0",
    "Bakanayatsu/ponyDiffusion-V6-XL-Turbo-DPO",
    "OnomaAIResearch/Illustrious-xl-early-release-v0",
    "GraydientPlatformAPI/sanae-xl",
    "yodayo-ai/clandestine-xl-1.0",
    "stablediffusionapi/anime-journey-v2",
    "aipicasso/emi-2",
    "zenless-lab/sdxl-anything-xl",
    "-[ 🏆 🐢 Flux ]-",
    "Freepik/flux.1-lite-8B",
    "black-forest-labs/FLUX.1-schnell",
    "sayakpaul/FLUX.1-merged",
    "ChuckMcSneed/FLUX.1-dev",
    "NikolaSigmoid/FLUX.1-Krea-dev",
    "AlekseyCalvin/FluxKrea_HSTurbo_Diffusers",
    "enhanceaiteam/Mystic",
    "AlekseyCalvin/AuraFlux_merge_diffusers",
    "ostris/Flex.1-alpha",
    "shuttleai/shuttle-jaguar",
    "Shakker-Labs/AWPortrait-FL",
    "AlekseyCalvin/PixelWave_Schnell_03_by_humblemikey_Diffusers_fp8_T4bf16",
    "AlekseyCalvin/PixelwaveFluxSchnell_Diffusers",
    "mikeyandfriends/PixelWave_FLUX.1-schnell_04",
    "minpeter/FLUX-Hyperscale-fused-fast",
    "-[ 🏠 Local models ]-",
]

model_list_img2img_ip = model_list_img2img_ip_builtin

for k in range(len(model_list_img2img_ip_local)):
    model_list_img2img_ip.append(model_list_img2img_ip_local[k])

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
    lora_model2_img2img_ip,
    lora_weight2_img2img_ip,
    lora_model3_img2img_ip,
    lora_weight3_img2img_ip,
    lora_model4_img2img_ip,
    lora_weight4_img2img_ip,
    lora_model5_img2img_ip,
    lora_weight5_img2img_ip,
    txtinv_img2img_ip,
    progress_img2img_ip=gr.Progress(track_tqdm=True)
    ):

    print(">>>[IP-Adapter 🖌️ ]: starting module")

    modelid_img2img_ip = model_cleaner_sd(modelid_img2img_ip)

    lora_model_img2img_ip = model_cleaner_lora(lora_model_img2img_ip)
    lora_model2_img2img_ip = model_cleaner_lora(lora_model2_img2img_ip)
    lora_model3_img2img_ip = model_cleaner_lora(lora_model3_img2img_ip)
    lora_model4_img2img_ip = model_cleaner_lora(lora_model4_img2img_ip)
    lora_model5_img2img_ip = model_cleaner_lora(lora_model5_img2img_ip)

    lora_array = []
    lora_weight_array = []
    adapters_list = []

    if lora_model_img2img_ip != "":
        if (is_flux(modelid_img2img_ip)) and ((lora_model_img2img_ip == "ByteDance/Hyper-SD") or (lora_model_img2img_ip == "RED-AIGC/TDD")):
            lora_weight_img2img_ip = 0.12
        elif (is_sdxl(modelid_img2img_ip)) and (lora_model_img2img_ip == "sd-community/sdxl-flash-lora"):
            lora_weight_img2img_ip = 0.55
        lora_array.append(f"{lora_model_img2img_ip}")
        lora_weight_array.append(float(lora_weight_img2img_ip))
    if lora_model2_img2img_ip != "":
        lora_array.append(f"{lora_model2_img2img_ip}")
        lora_weight_array.append(float(lora_weight2_img2img_ip))
    if lora_model3_img2img_ip != "":
        lora_array.append(f"{lora_model3_img2img_ip}")
        lora_weight_array.append(float(lora_weight3_img2img_ip))
    if lora_model4_img2img_ip != "":
        lora_array.append(f"{lora_model4_img2img_ip}")
        lora_weight_array.append(float(lora_weight4_img2img_ip))
    if lora_model5_img2img_ip != "":
        lora_array.append(f"{lora_model5_img2img_ip}")
        lora_weight_array.append(float(lora_weight5_img2img_ip))

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

    if is_sd3(modelid_img2img_ip):
        is_sd3_img2img_ip: bool = True
    else :
        is_sd3_img2img_ip: bool = False

    if is_bin(modelid_img2img_ip):
        is_bin_img2img_ip: bool = True
    else :
        is_bin_img2img_ip: bool = False

    if is_flux(modelid_img2img_ip):
        is_flux_img2img_ip: bool = True
    else :
        is_flux_img2img_ip: bool = False

    if (num_inference_step_img2img_ip >= 10) and use_ays_img2img_ip and not is_flux(modelid_img2img_ip):
        if is_sdxl(modelid_img2img_ip):
            sampling_schedule_img2img_ip = AysSchedules["StableDiffusionXLTimesteps"]
            sampler_img2img_ip = "DPM++ SDE"
        else:
            sampling_schedule_img2img_ip = AysSchedules["StableDiffusionTimesteps"]
            sampler_img2img_ip = "Euler"
        num_inference_step_img2img_ip = 10
    elif use_ays_img2img_ip and is_flux(modelid_img2img_ip):
        sampling_schedule_img2img_ip = AysSchedules["StableDiffusionXLTimesteps"]
        sampler_img2img_ip = "Flow Match Euler"
        if (num_inference_step_img2img_ip >= 10):
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
        elif (is_flux_img2img_ip == True):
            hf_hub_download(
                repo_id="XLabs-AI/flux-ip-adapter-v2",
                filename="ip_adapter.safetensors",
                repo_type="model",
                local_dir=model_path_ipa_img2img_ip,
                torch_dtype=model_arch,
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
    elif (is_flux_img2img_ip == True):
        clip_model_img2img_ip = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            cache_dir=model_path_ipa_img2img_ip,
            torch_dtype=model_arch,
            use_safetensors=True,
            resume_download=True,
            local_files_only=True if offline_test() else None
        )
        if modelid_img2img_ip[0:9] == "./models/" :
#            pipe_img2img_ip = FluxImg2ImgPipeline.from_single_file(
            pipe_img2img_ip = FluxPipeline.from_single_file(
                modelid_img2img_ip,
                torch_dtype=model_arch,
                clip_model=clip_model_img2img_ip,
                use_safetensors=True if not is_bin_img2img_ip else False,
                local_files_only=True if offline_test() else None
            )
        else :
#            pipe_img2img_ip = FluxImg2ImgPipeline.from_pretrained(
            pipe_img2img_ip = FluxPipeline.from_pretrained(
                modelid_img2img_ip,
                cache_dir=model_path_flux_img2img_ip,
                torch_dtype=model_arch,
                clip_model=clip_model_img2img_ip,
                use_safetensors=True if not is_bin_img2img_ip else False,
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
#                    load_safety_checker=False if (nsfw_filter_final == None) else True,
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
#                    load_safety_checker=False if (nsfw_filter_final == None) else True,
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
                    adapter_name="IP Adapter",
                    torch_dtype=model_arch,
                    use_safetensors=True,
                    resume_download=True,
                    local_files_only=True if offline_test() else None
                )
#                adapters_list.append("IP Adapter")
#                lora_weight_array.insert(0, float(denoising_strength_img2img_ip))
            elif (source_type_img2img_ip == "composition"):
                pipe_img2img_ip.load_ip_adapter(
                    model_path_ipa_img2img_ip,
                    subfolder="sdxl_models",
                    weight_name="ip_plus_composition_sdxl.safetensors",
                    adapter_name="Composition",
                    torch_dtype=model_arch,
                    use_safetensors=True, 
                    resume_download=True,
                    local_files_only=True if offline_test() else None
                )
#                adapters_list.append("Composition")
#                lora_weight_array.insert(0, float(denoising_strength_img2img_ip))

        elif (is_flux_img2img_ip == True):
            pipe_img2img_ip.load_ip_adapter(
                "XLabs-AI/flux-ip-adapter-v2",
                cache_dir=model_path_ipa_img2img_ip,
                subfolder="",
                weight_name="ip_adapter.safetensors",
                adapter_name="IP Adapter V2",
                adapter_weights=float(denoising_strength_img2img_ip),
                image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14",
                image_encoder_dtype=model_arch,
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
                    adapter_name="IP Adapter",
                    torch_dtype=model_arch,
                    use_safetensors=True, 
                    resume_download=True,
                    local_files_only=True if offline_test() else None
                )
#                adapters_list.append("IP Adapter")
#                lora_weight_array.insert(0, float(denoising_strength_img2img_ip))
            elif (source_type_img2img_ip == "composition"):
                pipe_img2img_ip.load_ip_adapter(
                    model_path_ipa_img2img_ip,
                    subfolder="models",
                    weight_name="ip_plus_composition_sd15.safetensors",
                    adapter_name="Composition",
                    torch_dtype=model_arch,
                    use_safetensors=True, 
                    resume_download=True,
                    local_files_only=True if offline_test() else None
                )
#                adapters_list.append("Composition")
#                lora_weight_array.insert(0, float(denoising_strength_img2img_ip))

    else:
        if (is_xl_img2img_ip == True):
            if (source_type_img2img_ip == "standard"):
                pipe_img2img_ip.load_ip_adapter(
                    "h94/IP-Adapter", 
                    cache_dir=model_path_ipa_img2img_ip,
                    subfolder="sdxl_models",
                    weight_name="ip-adapter_sdxl.safetensors",
                    adapter_name="IP Adapter",
                    torch_dtype=model_arch,
                    use_safetensors=True,
                    resume_download=True,
                    local_files_only=True if offline_test() else None
                )
#                adapters_list.append("IP Adapter")
#                lora_weight_array.insert(0, float(denoising_strength_img2img_ip))
            elif (source_type_img2img_ip == "composition"):
                pipe_img2img_ip.load_ip_adapter(
                    model_path_ipa_img2img_ip,
                    subfolder="sdxl_models",
                    weight_name="ip_plus_composition_sdxl.safetensors",
                    adapter_name="Composition",
                    torch_dtype=model_arch,
                    use_safetensors=True, 
                    resume_download=True,
                    local_files_only=True if offline_test() else None
                )
#                adapters_list.append("Composition")
#                lora_weight_array.insert(0, float(denoising_strength_img2img_ip))

        elif (is_flux_img2img_ip == True):
            pipe_img2img_ip.load_ip_adapter(
                "XLabs-AI/flux-ip-adapter-v2",
                cache_dir=model_path_ipa_img2img_ip,
                subfolder="",
                weight_name="ip_adapter.safetensors",
                adapter_name="IP Adapter V2",
                adapter_weights=float(denoising_strength_img2img_ip),
                image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14",
                image_encoder_dtype=model_arch,
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
                    adapter_name="IP Adapter",
                    torch_dtype=model_arch,
                    use_safetensors=True, 
                    resume_download=True,
                    local_files_only=True if offline_test() else None
                )
#                adapters_list.append("IP Adapter")
#                lora_weight_array.insert(0, float(denoising_strength_img2img_ip))
            elif (source_type_img2img_ip == "composition"):
                pipe_img2img_ip.load_ip_adapter(
                    model_path_ipa_img2img_ip,
                    subfolder="models",
                    weight_name="ip_plus_composition_sd15.safetensors",
                    adapter_name="Composition",
                    torch_dtype=model_arch,
                    use_safetensors=True, 
                    resume_download=True,
                    local_files_only=True if offline_test() else None
                )
#                adapters_list.append("Composition")
#                lora_weight_array.insert(0, float(denoising_strength_img2img_ip))
#    pipe_img2img_ip.set_ip_adapter_scale(denoising_strength_img2img_ip)    
    if  use_ays_img2img_ip and is_flux(modelid_img2img_ip):
        pipe_img2img_ip = schedulerer(pipe_img2img_ip, sampler_img2img_ip, timesteps=sampling_schedule_img2img_ip)
    else:
        pipe_img2img_ip = schedulerer(pipe_img2img_ip, sampler_img2img_ip)
#    pipe_img2img_ip.enable_attention_slicing("max")  
    if not is_flux_img2img_ip:
        tomesd.apply_patch(pipe_img2img_ip, ratio=tkme_img2img_ip)
    if device_label_img2img_ip == "cuda" :
        pipe_img2img_ip.enable_sequential_cpu_offload()
    else :
        pipe_img2img_ip = pipe_img2img_ip.to(device_img2img_ip)

    if len(lora_array) != 0:
        for e in range(len(lora_array)):
            model_list_lora_img2img_ip = lora_model_list(modelid_img2img_ip)
            if lora_array[e][0:9] == "./models/":
                pipe_img2img_ip.load_lora_weights(
                    os.path.dirname(lora_array[e]),
                    weight_name=model_list_lora_img2img_ip[lora_array[e]][0],
                    use_safetensors=True,
                    adapter_name=f"adapter{e}",
                    local_files_only=True if offline_test() else None,
                )
            else:
                if is_xl_img2img_ip:
                    lora_model_path = model_path_lora_sdxl
                elif is_sd3_img2img_ip:
                    lora_model_path = model_path_lora_sd3
                elif is_flux_img2img_ip:
                    lora_model_path = model_path_lora_flux
                else:
                    lora_model_path = model_path_lora_sd

                local_lora_img2img_ip = hf_hub_download(
                    repo_id=lora_array[e],
                    filename=model_list_lora_img2img_ip[lora_array[e]][0],
                    cache_dir=lora_model_path,
                    resume_download=True,
                    local_files_only=True if offline_test() else None,
                )

                pipe_img2img_ip.load_lora_weights(
                    lora_array[e],
                    weight_name=model_list_lora_img2img_ip[lora_array[e]][0],
                    cache_dir=lora_model_path,
                    use_safetensors=True,
                    adapter_name=f"adapter{e}",
                )
            adapters_list.append(f"adapter{e}")
        pipe_img2img_ip.set_adapters(adapters_list, adapter_weights=lora_weight_array)

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
        image_input = PIL.Image.open(img_img2img_ip)
        if (is_xl_img2img_ip or is_flux_img2img_ip) and not is_turbo_img2img_ip:
            dim_size = correct_size(image_input.size[0], image_input.size[1], 1024)
        else: 
            dim_size = correct_size(image_input.size[0], image_input.size[1], 512)
        image_input = image_input.convert("RGB")
        image_input = image_input.resize((dim_size[0], dim_size[1]))
        width_img2img_ip = dim_size[0]
        height_img2img_ip = dim_size[1]
    else:
        image_input = None

    if (img_ipa_img2img_ip != None):
        image_input_ipa = PIL.Image.open(img_ipa_img2img_ip)
        if (is_xl_img2img_ip or is_flux_img2img_ip) and not is_turbo_img2img_ip:
            dim_size_ipa = correct_size(image_input_ipa.size[0], image_input_ipa.size[1], 1024)
        else:
            dim_size_ipa = correct_size(image_input_ipa.size[0], image_input_ipa.size[1], 512)
        image_input_ipa = image_input_ipa.convert("RGB")
        image_input_ipa = image_input_ipa.resize((dim_size_ipa[0], dim_size_ipa[1]))
        if image_input == None:
            width_img2img_ip = dim_size_ipa[0]
            height_img2img_ip = dim_size_ipa[1]
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
    elif (is_flux_img2img_ip == True) :
        pass
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
        elif (is_xl_img2img_ip == True):
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
        elif (is_flux_img2img_ip == True):
            image = pipe_img2img_ip(
#                image=image_input,
                ip_adapter_image=image_input_ipa,
                prompt=prompt_img2img_ip,
                width=width_img2img_ip,
                height=height_img2img_ip,
                max_sequence_length=512,
                num_images_per_prompt=num_images_per_prompt_img2img_ip,
                guidance_scale=guidance_scale_img2img_ip,
#                strength=denoising_strength_img2img_ip,
                num_inference_steps=num_inference_step_img2img_ip,
#                timesteps=sampling_schedule_img2img_ip,
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
            if is_xl_img2img_ip or is_flux_img2img_ip or (modelid_img2img_ip[0:9] == "./models/"):
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
        f"LoRA model={lora_array} | "+\
        f"LoRA weight={lora_weight_array} | "+\
        f"Textual inversion={txtinv_img2img_ip} | "+\
        f"nsfw_filter={bool(int(nsfw_filter))} | "+\
        f"Denoising strength={denoising_strength_img2img_ip} | "+\
        f"Prompt={prompt_img2img_ip} | "+\
        f"Negative prompt={negative_prompt_img2img_ip}"
    print(reporting_img2img_ip)         

    exif_writer_png(reporting_img2img_ip, final_image)

    if is_flux_img2img_ip:
        del nsfw_filter_final, feat_ex, pipe_img2img_ip, generator, image_input, image_input_ipa, image
    else:
        del nsfw_filter_final, feat_ex, pipe_img2img_ip, generator, image_input, image_input_ipa, compel, conditioning, neg_conditioning, image
    clean_ram()

    print(f">>>[IP-Adapter 🖌️ ]: leaving module")
    return final_image, final_image 
