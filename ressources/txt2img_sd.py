# https://github.com/Woolverine94/biniou
# txt2img_sd.py
import gradio as gr
import os
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image, StableDiffusion3Pipeline, FluxPipeline
from huggingface_hub import hf_hub_download
from compel import Compel, ReturnedEmbeddingsType
import torch
import random
from ressources.gfpgan import *
import tomesd
from diffusers.schedulers import AysSchedules

# device_txt2img_sd = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_label_txt2img_sd, model_arch = detect_device()
device_txt2img_sd = torch.device(device_label_txt2img_sd)

# Gestion des modèles
model_path_txt2img_sd = "./models/Stable_Diffusion/"
os.makedirs(model_path_txt2img_sd, exist_ok=True)
model_path_flux_txt2img_sd = "./models/Flux/"
os.makedirs(model_path_flux_txt2img_sd, exist_ok=True)

model_list_txt2img_sd_local = []

for filename in os.listdir(model_path_txt2img_sd):
    f = os.path.join(model_path_txt2img_sd, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
        model_list_txt2img_sd_local.append(f)

model_list_txt2img_sd_builtin = [
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
    "-[ 👍 🚀 Fast SD15 ]-",
    "IDKiro/sdxs-512-0.9",
    "IDKiro/sdxs-512-dreamshaper",
    "stabilityai/sd-turbo",
    "-[ 👍 🇯🇵 Anime SD15 ]-",
    "gsdf/Counterfeit-V2.5",
    "fluently/Fluently-anime",
    "xyn-ai/anything-v4.0",
    "nitrosocke/Ghibli-Diffusion",
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
    "IDEA-CCNL/Taiyi-Stable-Diffusion-XL-3.5B",
    "etri-vilab/koala-lightning-700m",
    "etri-vilab/koala-lightning-1b",
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
    "segmind/SSD-1B",
    "segmind/Segmind-Vega",
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
    "thibaud/sdxl_dpo_turbo",
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
    "aipicasso/emi-3",
    "-[ 👏 🐢 SD3 ]-",
    "v2ray/stable-diffusion-3-medium-diffusers",
    "ptx0/sd3-reality-mix",
    "-[ 👏 🐢 SD3.5 Large ]-",
    "adamo1139/stable-diffusion-3.5-large-turbo-ungated",
    "ariG23498/sd-3.5-merged",
    "aipicasso/emi-3",
    "-[ 👏 🐢 SD3.5 Medium ]-",
    "adamo1139/stable-diffusion-3.5-medium-ungated",
    "tensorart/stable-diffusion-3.5-medium-turbo",
    "-[ 🏆 🐢 Flux ]-",
    "Freepik/flux.1-lite-8B",
    "black-forest-labs/FLUX.1-schnell",
    "sayakpaul/FLUX.1-merged",
    "ChuckMcSneed/FLUX.1-dev",
    "enhanceaiteam/Mystic",
    "AlekseyCalvin/AuraFlux_merge_diffusers",
    "ostris/Flex.1-alpha",
    "shuttleai/shuttle-jaguar",
    "Shakker-Labs/AWPortrait-FL",
    "AlekseyCalvin/PixelWave_Schnell_03_by_humblemikey_Diffusers_fp8_T4bf16",
    "-[ 🏠 Local models ]-",
]

model_list_txt2img_sd = model_list_txt2img_sd_builtin

for k in range(len(model_list_txt2img_sd_local)):
    model_list_txt2img_sd.append(model_list_txt2img_sd_local[k])

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
    clipskip_txt2img_sd,
    use_ays_txt2img_sd,
    lora_model_txt2img_sd,
    lora_weight_txt2img_sd,
    lora_model2_txt2img_sd,
    lora_weight2_txt2img_sd,
    lora_model3_txt2img_sd,
    lora_weight3_txt2img_sd,
    lora_model4_txt2img_sd,
    lora_weight4_txt2img_sd,
    lora_model5_txt2img_sd,
    lora_weight5_txt2img_sd,
    txtinv_txt2img_sd,
    progress_txt2img_sd=gr.Progress(track_tqdm=True)
    ):

    print(">>>[Stable Diffusion 🖼️ ]: starting module")

    modelid_txt2img_sd = model_cleaner_sd(modelid_txt2img_sd)
    lora_model_txt2img_sd = model_cleaner_lora(lora_model_txt2img_sd)
    lora_model2_txt2img_sd = model_cleaner_lora(lora_model2_txt2img_sd)
    lora_model3_txt2img_sd = model_cleaner_lora(lora_model3_txt2img_sd)
    lora_model4_txt2img_sd = model_cleaner_lora(lora_model4_txt2img_sd)
    lora_model5_txt2img_sd = model_cleaner_lora(lora_model5_txt2img_sd)

    lora_array = []
    lora_weight_array = []

    if lora_model_txt2img_sd != "":
        if (is_sd3(modelid_txt2img_sd) or is_flux(modelid_txt2img_sd)) and ((lora_model_txt2img_sd == "ByteDance/Hyper-SD") or (lora_model_txt2img_sd == "RED-AIGC/TDD")):
            lora_weight_txt2img_sd = 0.12
        lora_array.append(f"{lora_model_txt2img_sd}")
        lora_weight_array.append(float(lora_weight_txt2img_sd))
    if lora_model2_txt2img_sd != "":
        lora_array.append(f"{lora_model2_txt2img_sd}")
        lora_weight_array.append(float(lora_weight2_txt2img_sd))
    if lora_model3_txt2img_sd != "":
        lora_array.append(f"{lora_model3_txt2img_sd}")
        lora_weight_array.append(float(lora_weight3_txt2img_sd))
    if lora_model4_txt2img_sd != "":
        lora_array.append(f"{lora_model4_txt2img_sd}")
        lora_weight_array.append(float(lora_weight4_txt2img_sd))
    if lora_model5_txt2img_sd != "":
        lora_array.append(f"{lora_model5_txt2img_sd}")
        lora_weight_array.append(float(lora_weight5_txt2img_sd))

    global pipe_txt2img_sd
    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_txt2img_sd, device_txt2img_sd, nsfw_filter)

    if clipskip_txt2img_sd == 0:
       clipskip_txt2img_sd = None

    if ("turbo" in modelid_txt2img_sd):
        is_turbo_txt2img_sd: bool = True
    else :
        is_turbo_txt2img_sd: bool = False

    if is_sdxl(modelid_txt2img_sd):
        is_xl_txt2img_sd: bool = True
    else :        
        is_xl_txt2img_sd: bool = False

    if is_sd3(modelid_txt2img_sd):
        is_sd3_txt2img_sd: bool = True
    else :        
        is_sd3_txt2img_sd: bool = False

    if is_sd35(modelid_txt2img_sd):
        is_sd35_txt2img_sd: bool = True
    else :
        is_sd35_txt2img_sd: bool = False

    if is_sd35m(modelid_txt2img_sd):
        is_sd35m_txt2img_sd: bool = True
    else :
        is_sd35m_txt2img_sd: bool = False

    if is_bin(modelid_txt2img_sd):
        is_bin_txt2img_sd: bool = True
    else :
        is_bin_txt2img_sd: bool = False

    if is_flux(modelid_txt2img_sd):
        is_flux_txt2img_sd: bool = True
    else :        
        is_flux_txt2img_sd: bool = False

    if is_turbo_txt2img_sd and is_sd35_txt2img_sd:
        is_turbo_txt2img_sd: bool = False

    if (num_inference_step_txt2img_sd >= 10) and use_ays_txt2img_sd:
        if is_sdxl(modelid_txt2img_sd):
            sampling_schedule_txt2img_sd = AysSchedules["StableDiffusionXLTimesteps"]
            sampler_txt2img_sd = "DPM++ SDE"
        elif is_sd3(modelid_txt2img_sd):
            pass
        else:
            sampling_schedule_txt2img_sd = AysSchedules["StableDiffusionTimesteps"]
            sampler_txt2img_sd = "Euler"
        num_inference_step_txt2img_sd = 10
    else:
        sampling_schedule_txt2img_sd = None

    if (is_turbo_txt2img_sd == True) :
        if modelid_txt2img_sd[0:9] == "./models/" :
            pipe_txt2img_sd =AutoPipelineForText2Image.from_single_file(
                modelid_txt2img_sd, 
#                torch_dtype=torch.float32, 
                torch_dtype=model_arch, 
                use_safetensors=True if not is_bin_txt2img_sd else False,
#                load_safety_checker=False if (nsfw_filter_final == None) else True,
                local_files_only=True if offline_test() else None
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
                torch_dtype=model_arch, 
                use_safetensors=True if not is_bin_txt2img_sd else False,
#                load_safety_checker=False if (nsfw_filter_final == None) else True,
                local_files_only=True if offline_test() else None
            )
        else :        
            pipe_txt2img_sd = StableDiffusionXLPipeline.from_pretrained(
                modelid_txt2img_sd, 
                cache_dir=model_path_txt2img_sd, 
                torch_dtype=model_arch, 
                use_safetensors=True if not is_bin_txt2img_sd else False,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
    elif is_sd3_txt2img_sd or is_sd35_txt2img_sd or is_sd35m_txt2img_sd:
        if modelid_txt2img_sd[0:9] == "./models/" :
            pipe_txt2img_sd = StableDiffusion3Pipeline.from_single_file(
                modelid_txt2img_sd, 
                text_encoder_3=None,
                tokenizer_3=None,
                torch_dtype=model_arch, 
                use_safetensors=True if not is_bin_txt2img_sd else False,
#                load_safety_checker=False if (nsfw_filter_final == None) else True,
                local_files_only=True if offline_test() else None
            )
        else :
            pipe_txt2img_sd = StableDiffusion3Pipeline.from_pretrained(
                modelid_txt2img_sd,
                text_encoder_3=None,
                tokenizer_3=None,
                cache_dir=model_path_txt2img_sd, 
                torch_dtype=model_arch, 
                use_safetensors=True if not is_bin_txt2img_sd else False,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
    elif (is_flux_txt2img_sd == True):
        if modelid_txt2img_sd[0:9] == "./models/" :
            pipe_txt2img_sd = FluxPipeline.from_single_file(
                modelid_txt2img_sd, 
                torch_dtype=model_arch, 
                use_safetensors=True if not is_bin_txt2img_sd else False,
#                load_safety_checker=False if (nsfw_filter_final == None) else True,
                local_files_only=True if offline_test() else None
            )
        else :
            pipe_txt2img_sd = FluxPipeline.from_pretrained(
                modelid_txt2img_sd,
                cache_dir=model_path_flux_txt2img_sd,
                torch_dtype=model_arch,
                use_safetensors=True if not is_bin_txt2img_sd else False,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
    else :
        if modelid_txt2img_sd[0:9] == "./models/" :
            pipe_txt2img_sd = StableDiffusionPipeline.from_single_file(
                modelid_txt2img_sd, 
                torch_dtype=model_arch,                 
                use_safetensors=True if not is_bin_txt2img_sd else False,
#                load_safety_checker=False if (nsfw_filter_final == None) else True,
                local_files_only=True if offline_test() else None
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
    if not is_sd3_txt2img_sd and not is_sd35_txt2img_sd and not is_sd35m_txt2img_sd and not is_flux_txt2img_sd:
        tomesd.apply_patch(pipe_txt2img_sd, ratio=tkme_txt2img_sd)
    if device_label_txt2img_sd == "cuda" :
        pipe_txt2img_sd.enable_sequential_cpu_offload()
    else: 
        pipe_txt2img_sd = pipe_txt2img_sd.to(device_txt2img_sd)
    if not is_sd3_txt2img_sd and not is_sd35_txt2img_sd and not is_sd35m_txt2img_sd:
        pipe_txt2img_sd.enable_vae_slicing()

    adapters_list = []

    if len(lora_array) != 0:
        for e in range(len(lora_array)):
            model_list_lora_txt2img_sd = lora_model_list(modelid_txt2img_sd)
            if lora_array[e][0:9] == "./models/":
                pipe_txt2img_sd.load_lora_weights(
                    os.path.dirname(lora_array[e]),
                    weight_name=model_list_lora_txt2img_sd[lora_array[e]][0],
                    use_safetensors=True,
                    adapter_name=f"adapter{e}",
                    local_files_only=True if offline_test() else None,
                )
            else:
                if is_xl_txt2img_sd:
                    lora_model_path = model_path_lora_sdxl
                elif is_sd3_txt2img_sd:
                    lora_model_path = model_path_lora_sd3
                elif is_sd35_txt2img_sd or is_sd35m_txt2img_sd:
                    lora_model_path = model_path_lora_sd35
                elif is_flux_txt2img_sd:
                    lora_model_path = model_path_lora_flux
                else: 
                    lora_model_path = model_path_lora_sd

                local_lora_txt2img_sd = hf_hub_download(
                    repo_id=lora_array[e],
                    filename=model_list_lora_txt2img_sd[lora_array[e]][0],
                    cache_dir=lora_model_path,
                    resume_download=True,
                    local_files_only=True if offline_test() else None,
                )

                pipe_txt2img_sd.load_lora_weights(
                    lora_array[e],
                    weight_name=model_list_lora_txt2img_sd[lora_array[e]][0],
                    cache_dir=lora_model_path,
                    use_safetensors=True,
                    adapter_name=f"adapter{e}",
                )
            adapters_list.append(f"adapter{e}")

        pipe_txt2img_sd.set_adapters(adapters_list, adapter_weights=lora_weight_array)

    if txtinv_txt2img_sd != "":
        model_list_txtinv_txt2img_sd = txtinv_list(modelid_txt2img_sd)
        weight_txt2img_sd = model_list_txtinv_txt2img_sd[txtinv_txt2img_sd][0]
        token_txt2img_sd =  model_list_txtinv_txt2img_sd[txtinv_txt2img_sd][1]
        if txtinv_txt2img_sd[0:9] == "./models/":
            model_path_txtinv = "./models/TextualInversion"
            pipe_txt2img_sd.load_textual_inversion(
                txtinv_txt2img_sd,
                weight_name=weight_txt2img_sd,
                use_safetensors=True,
                token=token_txt2img_sd,
                local_files_only=True if offline_test() else None,
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
                local_files_only=True if offline_test() else None,
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
    elif is_sd3_txt2img_sd or is_sd35_txt2img_sd or is_sd35m_txt2img_sd or is_flux_txt2img_sd:
        pass
    else :
        compel = Compel(tokenizer=pipe_txt2img_sd.tokenizer, text_encoder=pipe_txt2img_sd.text_encoder, truncate_long_prompts=False, device=device_txt2img_sd)
        conditioning = compel.build_conditioning_tensor(prompt_txt2img_sd)
        neg_conditioning = compel.build_conditioning_tensor(negative_prompt_txt2img_sd)
        [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])

    final_image = []
    final_seed = []
    for i in range (num_prompt_txt2img_sd):
        if (is_xl_txt2img_sd == True):
            image = pipe_txt2img_sd(
                prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled, 
                negative_prompt_embeds=neg_conditioning,
                negative_pooled_prompt_embeds=neg_pooled,
                height=height_txt2img_sd,
                width=width_txt2img_sd,
                num_images_per_prompt=num_images_per_prompt_txt2img_sd,
                num_inference_steps=num_inference_step_txt2img_sd,
                timesteps=sampling_schedule_txt2img_sd,
                guidance_scale=guidance_scale_txt2img_sd,
                generator=generator[i],
                callback_on_step_end=check_txt2img_sd, 
                callback_on_step_end_tensor_inputs=['latents'], 
            ).images
        elif is_sd3_txt2img_sd or is_sd35_txt2img_sd or is_sd35m_txt2img_sd:
            image = pipe_txt2img_sd(
                prompt=prompt_txt2img_sd,
                negative_prompt=negative_prompt_txt2img_sd,
                height=height_txt2img_sd,
                width=width_txt2img_sd,
                num_images_per_prompt=num_images_per_prompt_txt2img_sd,
                num_inference_steps=num_inference_step_txt2img_sd,
                timesteps=sampling_schedule_txt2img_sd,
                guidance_scale=guidance_scale_txt2img_sd,
                generator=generator[i],
                callback_on_step_end=check_txt2img_sd,
                callback_on_step_end_tensor_inputs=['latents'],
            ).images
        elif is_flux_txt2img_sd:
            image = pipe_txt2img_sd(
                prompt=prompt_txt2img_sd,
#                negative_prompt=negative_prompt_txt2img_sd,
                height=height_txt2img_sd,
                width=width_txt2img_sd,
                num_images_per_prompt=num_images_per_prompt_txt2img_sd,
                num_inference_steps=num_inference_step_txt2img_sd,
#                timesteps=sampling_schedule_txt2img_sd,
                guidance_scale=guidance_scale_txt2img_sd,
                generator=generator[i],
                callback_on_step_end=check_txt2img_sd,
                callback_on_step_end_tensor_inputs=['latents'],
            ).images
        else:
            image = pipe_txt2img_sd(
                prompt_embeds=conditioning,
                negative_prompt_embeds=neg_conditioning,
                height=height_txt2img_sd,
                width=width_txt2img_sd,
                num_images_per_prompt=num_images_per_prompt_txt2img_sd,
                num_inference_steps=num_inference_step_txt2img_sd,
                timesteps=sampling_schedule_txt2img_sd,
                guidance_scale=guidance_scale_txt2img_sd,
                generator=generator[i],
                clip_skip=clipskip_txt2img_sd,
                callback_on_step_end=check_txt2img_sd, 
                callback_on_step_end_tensor_inputs=['latents'], 
            ).images
        
        for j in range(len(image)):
            if is_xl_txt2img_sd or is_sd3_txt2img_sd or is_sd35_txt2img_sd or is_sd35m_txt2img_sd or is_flux_txt2img_sd or (modelid_txt2img_sd[0:9] == "./models/"):
                image[j] = safety_checker_sdxl(model_path_txt2img_sd, image[j], nsfw_filter)
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
        f"CLIP skip={clipskip_txt2img_sd} | "+\
        f"AYS={use_ays_txt2img_sd} | "+\
        f"LoRA model={lora_array} | "+\
        f"LoRA weight={lora_weight_array} | "+\
        f"Textual inversion={txtinv_txt2img_sd} | "+\
        f"nsfw_filter={bool(int(nsfw_filter))} | "+\
        f"Prompt={prompt_txt2img_sd} | "+\
        f"Negative prompt={negative_prompt_txt2img_sd} | "+\
        f"Seed List="+ ', '.join([f"{final_seed[m]}" for m in range(len(final_seed))])
    print(reporting_txt2img_sd)

    exif_writer_png(reporting_txt2img_sd, final_image)
    
    if is_sd3_txt2img_sd or is_sd35_txt2img_sd or is_sd35m_txt2img_sd or is_flux_txt2img_sd:
        del nsfw_filter_final, feat_ex, pipe_txt2img_sd, generator, image
    else:
        del nsfw_filter_final, feat_ex, pipe_txt2img_sd, generator, compel, conditioning, neg_conditioning, image
    clean_ram()

    print(f">>>[Stable Diffusion 🖼️ ]: leaving module")
    return final_image, final_image
