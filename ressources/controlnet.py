# https://github.com/Woolverine94/biniou
# controlnet.py
import gradio as gr
import os
import cv2
import torch
from diffusers import StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline, ControlNetModel, StableDiffusion3ControlNetPipeline, FluxControlNetPipeline, FluxControlNetModel
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from huggingface_hub import hf_hub_download
from compel import Compel, ReturnedEmbeddingsType
import random
from ressources.gfpgan import *
from controlnet_aux.processor import Processor
import tomesd
from diffusers.schedulers import AysSchedules

device_label_controlnet, model_arch = detect_device()
device_controlnet = torch.device(device_label_controlnet)

# Gestion des modèles
model_path_controlnet = "./models/Stable_Diffusion/"
model_path_flux_controlnet = "./models/Flux/"
os.makedirs(model_path_controlnet, exist_ok=True)
os.makedirs(model_path_flux_controlnet, exist_ok=True)
model_list_controlnet_local = []

for filename in os.listdir(model_path_controlnet):
    f = os.path.join(model_path_controlnet, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
        model_list_controlnet_local.append(f)

model_list_controlnet_builtin = [
    "-[ 👍 SD15 ]-",
    "SG161222/Realistic_Vision_V3.0_VAE",
    "Yntec/VisionVision",
    "fluently/Fluently-epic",
    "SG161222/Paragon_V1.0",
    "digiplay/AbsoluteReality_v1.8.1",
    "digiplay/majicMIX_realistic_v7",
    "SPO-Diffusion-Models/SPO-SD-v1-5_4k-p_10ep",
    "digiplay/PerfectDeliberate_v5",
    "runwayml/stable-diffusion-v1-5",
    "-[ 👍 🇯🇵 Anime SD15 ]-",
    "gsdf/Counterfeit-V2.5",
    "fluently/Fluently-anime",
    "xyn-ai/anything-v4.0",
    "nitrosocke/Ghibli-Diffusion",
    "-[ 👌 🐢 SDXL ]-",
    "fluently/Fluently-XL-Final",
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
    "recoilme/ColorfulXL-Lightning",
    "GraydientPlatformAPI/lustify-lightning",
    "John6666/comradeship-xl-v9a-spo-dpo-flash-sdxl",
    "stablediffusionapi/dream-diffusion-lightning",
    "John6666/jib-mix-realistic-xl-v15-maximus-sdxl",
    "thibaud/sdxl_dpo_turbo",
    "stabilityai/sdxl-turbo",
    "-[ 👌 🇯🇵 Anime SDXL ]-",
    "cagliostrolab/animagine-xl-3.1",
    "GraydientPlatformAPI/geekpower-cellshade-xl",
    "Bakanayatsu/ponyDiffusion-V6-XL-Turbo-DPO",
    "OnomaAIResearch/Illustrious-xl-early-release-v0",
    "GraydientPlatformAPI/sanae-xl",
    "yodayo-ai/clandestine-xl-1.0",
    "stablediffusionapi/anime-journey-v2",
    "aipicasso/emi-2",
    "-[ 👏 🐢 SD3 ]-",
    "v2ray/stable-diffusion-3-medium-diffusers",
    "ptx0/sd3-reality-mix",
    "-[ 🏆 🐢 Flux ]-",
    "Freepik/flux.1-lite-8B-alpha",
    "black-forest-labs/FLUX.1-schnell",
    "sayakpaul/FLUX.1-merged",
    "ChuckMcSneed/FLUX.1-dev",
    "enhanceaiteam/Mystic",
    "AlekseyCalvin/AuraFlux_merge_diffusers",
    "-[ 🏠 Local models ]-",
]

model_list_controlnet = model_list_controlnet_builtin

for k in range(len(model_list_controlnet_local)):
    model_list_controlnet.append(model_list_controlnet_local[k])

model_path_base_controlnet = "./models/controlnet"
os.makedirs(model_path_base_controlnet, exist_ok=True)

base_controlnet = "lllyasviel/ControlNet-v1-1"

variant_list_controlnet = [
    "lllyasviel/control_v11p_sd15_canny",
    "lllyasviel/control_v11f1p_sd15_depth",
    "lllyasviel/control_v11p_sd15s2_lineart_anime",
    "lllyasviel/control_v11p_sd15_lineart",
    "lllyasviel/control_v11p_sd15_mlsd",
    "lllyasviel/control_v11p_sd15_normalbae",
    "lllyasviel/control_v11p_sd15_openpose",
    "lllyasviel/control_v11p_sd15_scribble",
    "lllyasviel/control_v11p_sd15_softedge",
    "lllyasviel/control_v11f1e_sd15_tile",
    "Nacholmo/controlnet-qr-pattern-v2",
    "monster-labs/control_v1p_sd15_qrcode_monster",
    "patrickvonplaten/controlnet-canny-sdxl-1.0",
    "patrickvonplaten/controlnet-depth-sdxl-1.0",
    "thibaud/controlnet-openpose-sdxl-1.0",
    "SargeZT/controlnet-sd-xl-1.0-softedge-dexined",
    "ValouF-pimento/ControlNet_SDXL_tile_upscale",
    "Nacholmo/controlnet-qr-pattern-sdxl",
    "monster-labs/control_v1p_sdxl_qrcode_monster",
    "TheMistoAI/MistoLine",
    "brad-twinkl/controlnet-union-sdxl-1.0-promax",
    "xinsir/controlnet-union-sdxl-1.0",
    "InstantX/SD3-Controlnet-Canny",
    "InstantX/SD3-Controlnet-Pose",
    "InstantX/SD3-Controlnet-Tile",
    "XLabs-AI/flux-controlnet-canny-diffusers",
    "XLabs-AI/flux-controlnet-depth-diffusers",
#    "George0667/Flux.1-dev-ControlNet-LineCombo",
    "jasperai/Flux.1-dev-Controlnet-Upscaler",
#    "InstantX/FLUX.1-dev-Controlnet-Union",
    "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro",
]

preprocessor_list_controlnet = [
    "canny",
    "depth_leres",
    "depth_leres++",
    "depth_midas",
    "lineart_anime",
    "lineart_coarse",
    "lineart_realistic",
    "mlsd",
    "normal_bae",
    "openpose",
    "openpose_face",
    "openpose_faceonly",
    "openpose_full",
    "openpose_hand",
    "scribble_hed",
    "scribble_pidinet",
    "softedge_hed",
    "softedge_hedsafe",
    "softedge_pidinet",
    "softedge_pidsafe",
    "tile",
    "qr",
    "qr_invert",
    "qr_monster",
    "qr_monster_invert",
]

# Bouton Cancel
stop_controlnet = False

def initiate_stop_controlnet() :
    global stop_controlnet
    stop_controlnet = True

def check_controlnet(pipe, step_index, timestep, callback_kwargs) :
    global stop_controlnet
    if stop_controlnet == False :
        return callback_kwargs
    elif stop_controlnet == True :
        print(">>>[ControlNet 🖼️ ]: generation canceled by user")
        stop_controlnet = False
        try:
            del ressources.controlnet.pipe_controlnet
        except NameError as e:
            raise Exception("Interrupting ...")
            return "Canceled ..."
    return

def dispatch_controlnet_preview(
    modelid_controlnet,
    low_threshold_controlnet,
    high_threshold_controlnet,
    img_source_controlnet,
    preprocessor_controlnet,
    progress_controlnet=gr.Progress(track_tqdm=True)
    ):

    modelid_controlnet = model_cleaner_sd(modelid_controlnet)

    if is_sdxl(modelid_controlnet):
        is_xl_controlnet: bool = True
    else :
        is_xl_controlnet: bool = False

    if is_sd3(modelid_controlnet):
        is_sd3_controlnet: bool = True
    else :
        is_sd3_controlnet: bool = False

    if is_flux(modelid_controlnet):
        is_flux_controlnet: bool = True
    else :
        is_flux_controlnet: bool = False

    img_source_controlnet = Image.open(img_source_controlnet)
    img_source_controlnet = np.array(img_source_controlnet)
    if not (('qr' in preprocessor_controlnet) or ('tile' in preprocessor_controlnet)):
        processor_controlnet = Processor(preprocessor_controlnet)

    match preprocessor_controlnet:
# 01
        case "canny":
            result = canny_controlnet(img_source_controlnet, low_threshold_controlnet, high_threshold_controlnet)
#            return result, result, variant_list_controlnet[12] if is_xl_controlnet else variant_list_controlnet[0]
            if is_xl_controlnet:
                return result, result, variant_list_controlnet[20]
            elif is_sd3_controlnet:
                return result, result, variant_list_controlnet[22]
            elif is_flux_controlnet:
                return result, result, variant_list_controlnet[25]
            else:
                return result, result, variant_list_controlnet[0]
#            return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[0]
# 02
        case "depth_leres":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            if is_xl_controlnet:
                return result, result, variant_list_controlnet[13] 
            elif is_flux_controlnet:
                return result, result, variant_list_controlnet[26] 
            else:
                return result, result, variant_list_controlnet[1]
#            return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[1]
# 03
        case "depth_leres++":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            if is_xl_controlnet:
                return result, result, variant_list_controlnet[13] 
            elif is_flux_controlnet:
                return result, result, variant_list_controlnet[26] 
            else:
                return result, result, variant_list_controlnet[1]
#            return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[1]
# 04
        case "depth_midas":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
#            return result, result, variant_list_controlnet[13] if is_xl_controlnet else variant_list_controlnet[1]
#            return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[1]
            if is_xl_controlnet:
                return result, result, variant_list_controlnet[20] 
            elif is_flux_controlnet:
                return result, result, variant_list_controlnet[26] 
            else:
                return result, result, variant_list_controlnet[1]
#         case "depth_zoe":
#             result = processor_controlnet(img_source_controlnet, to_pil=True)
# #            return result, result, variant_list_controlnet[13] if is_xl_controlnet else variant_list_controlnet[1]
#             return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[1]
# 05
        case "lineart_anime":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
#            return result, result, variant_list_controlnet[12] if is_xl_controlnet else variant_list_controlnet[2]
            if is_xl_controlnet:
                return result, result, variant_list_controlnet[20]
            elif is_sd3_controlnet:
                return result, result, variant_list_controlnet[22]
            elif is_flux_controlnet:
                return result, result, variant_list_controlnet[25]
            else:
                return result, result, variant_list_controlnet[2]
#            return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[2]
# 06
        case "lineart_coarse":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
#            return result, result, variant_list_controlnet[12] if is_xl_controlnet else variant_list_controlnet[3]
            if is_xl_controlnet:
                return result, result, variant_list_controlnet[12]
            elif is_sd3_controlnet:
                return result, result, variant_list_controlnet[22]
            elif is_flux_controlnet:
                return result, result, variant_list_controlnet[25]
            else:
                return result, result, variant_list_controlnet[3]
#            return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[3]
# 07
        case "lineart_realistic":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
#            return result, result, variant_list_controlnet[12] if is_xl_controlnet else variant_list_controlnet[3]
            if is_xl_controlnet:
                return result, result, variant_list_controlnet[20]
            elif is_sd3_controlnet:
                return result, result, variant_list_controlnet[22]
            elif is_flux_controlnet:
                return result, result, variant_list_controlnet[25]
            else:
                return result, result, variant_list_controlnet[3]
#            return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[3]
# 08
        case "mlsd":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
#            return result, result, variant_list_controlnet[12] if is_xl_controlnet else variant_list_controlnet[4]
            if is_xl_controlnet:
                return result, result, variant_list_controlnet[20]
            elif is_sd3_controlnet:
                return result, result, variant_list_controlnet[22]
            elif is_flux_controlnet:
                return result, result, variant_list_controlnet[25]
            else:
                return result, result, variant_list_controlnet[4]
#            return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[4]
# 09
        case "normal_bae":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
#            return result, result, variant_list_controlnet[13] if is_xl_controlnet else variant_list_controlnet[5]
            return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[5]
#         case "normal_midas":
#             result = processor_controlnet(img_source_controlnet, to_pil=True)
# #            return result, result, variant_list_controlnet[13] if is_xl_controlnet else variant_list_controlnet[5]
#             return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[5]
# 10
        case "openpose":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
#            return result, result, variant_list_controlnet[14] if is_xl_controlnet else variant_list_controlnet[6]
            if is_xl_controlnet:
                return result, result, variant_list_controlnet[20]
            elif is_sd3_controlnet:
                return result, result, variant_list_controlnet[23]
            elif is_flux_controlnet:
                return result, result, variant_list_controlnet[25]
            else:
                return result, result, variant_list_controlnet[6]
#            return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[6]
# 11
        case "openpose_face":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
#            return result, result, variant_list_controlnet[14] if is_xl_controlnet else variant_list_controlnet[6]
            if is_xl_controlnet:
                return result, result, variant_list_controlnet[14]
            elif is_sd3_controlnet:
                return result, result, variant_list_controlnet[23]
            elif is_flux_controlnet:
                return result, result, variant_list_controlnet[25]
            else:
                return result, result, variant_list_controlnet[6]
#            return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[6]
# 12
        case "openpose_faceonly":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
#            return result, result, variant_list_controlnet[14] if is_xl_controlnet else variant_list_controlnet[6]
            if is_xl_controlnet:
                return result, result, variant_list_controlnet[14]
            elif is_sd3_controlnet:
                return result, result, variant_list_controlnet[23]
            elif is_flux_controlnet:
                return result, result, variant_list_controlnet[25]
            else:
                return result, result, variant_list_controlnet[6]
#            return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[6]
# 13
        case "openpose_full":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
#            return result, result, variant_list_controlnet[14] if is_xl_controlnet else variant_list_controlnet[6]
            if is_xl_controlnet:
                return result, result, variant_list_controlnet[14]
            elif is_sd3_controlnet:
                return result, result, variant_list_controlnet[23]
            elif is_flux_controlnet:
                return result, result, variant_list_controlnet[25]
            else:
                return result, result, variant_list_controlnet[6]
#            return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[6]
# 14
        case "openpose_hand":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
#            return result, result, variant_list_controlnet[14] if is_xl_controlnet else variant_list_controlnet[6]
            if is_xl_controlnet:
                return result, result, variant_list_controlnet[20]
            elif is_sd3_controlnet:
                return result, result, variant_list_controlnet[23]
            elif is_flux_controlnet:
                return result, result, variant_list_controlnet[25]
            else:
                return result, result, variant_list_controlnet[6]
#            return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[6]

# 15
        case "scribble_hed":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            if is_xl_controlnet:
                return result, result, variant_list_controlnet[20] 
            elif is_flux_controlnet:
                return result, result, variant_list_controlnet[25] 
            else:
                return result, result, variant_list_controlnet[7]
#            return result, result, variant_list_controlnet[15] if is_xl_controlnet else variant_list_controlnet[7]
#            return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[7]
# 16
        case "scribble_pidinet":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            if is_xl_controlnet:
                return result, result, variant_list_controlnet[20] 
            elif is_flux_controlnet:
                return result, result, variant_list_controlnet[25] 
            else:
                return result, result, variant_list_controlnet[7]
#            return result, result, variant_list_controlnet[15] if is_xl_controlnet else variant_list_controlnet[7]
#            return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[7]
# 17
        case "softedge_hed":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            if is_xl_controlnet:
                return result, result, variant_list_controlnet[20]
            elif is_flux_controlnet:
                return result, result, variant_list_controlnet[25]
            else:
                return result, result, variant_list_controlnet[8]
#            return result, result, variant_list_controlnet[15] if is_xl_controlnet else variant_list_controlnet[8]
#            return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[8]
# 18
        case "softedge_hedsafe":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            if is_xl_controlnet:
                return result, result, variant_list_controlnet[20] 
            elif is_flux_controlnet:
                return result, result, variant_list_controlnet[25] 
            else:
                return result, result, variant_list_controlnet[8]
#            return result, result, variant_list_controlnet[15] if is_xl_controlnet else variant_list_controlnet[8]
#            return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[8]
# 19
        case "softedge_pidinet":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            if is_xl_controlnet:
                return result, result, variant_list_controlnet[20] 
            elif is_flux_controlnet:
                return result, result, variant_list_controlnet[25] 
            else:
                return result, result, variant_list_controlnet[8]
#            return result, result, variant_list_controlnet[15] if is_xl_controlnet else variant_list_controlnet[8]
#            return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[8]
# 20
        case "softedge_pidsafe":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            if is_xl_controlnet:
                return result, result, variant_list_controlnet[20] 
            elif is_flux_controlnet:
                return result, result, variant_list_controlnet[25] 
            else:
                return result, result, variant_list_controlnet[8]
#            return result, result, variant_list_controlnet[15] if is_xl_controlnet else variant_list_controlnet[8]
#            return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[8]
# 21
        case "tile":
            result = tile_controlnet(img_source_controlnet, is_xl_controlnet, modelid_controlnet)
#            return result, result, variant_list_controlnet[16] if is_xl_controlnet else variant_list_controlnet[9]
            if is_xl_controlnet:
                return result, result, variant_list_controlnet[20]
            elif is_sd3_controlnet:
                return result, result, variant_list_controlnet[24]
            elif is_flux_controlnet:
                return result, result, variant_list_controlnet[27]
            else:
                return result, result, variant_list_controlnet[9]
#            return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[9]
# 22
        case "qr":
            result = qr_controlnet(img_source_controlnet, 0)
            return result, result, variant_list_controlnet[17] if is_xl_controlnet else variant_list_controlnet[10]
#            return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[10]
# 23
        case "qr_invert":
            result = qr_controlnet(img_source_controlnet, 1)
            return result, result, variant_list_controlnet[17] if is_xl_controlnet else variant_list_controlnet[10]
#            return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[10]
# 24
        case "qr_monster":
            result = qr_controlnet(img_source_controlnet, 0)
            return result, result, variant_list_controlnet[18] if is_xl_controlnet else variant_list_controlnet[11]
#            return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[11]
# 25
        case "qr_monster_invert":
            result = qr_controlnet(img_source_controlnet, 1)
            return result, result, variant_list_controlnet[18] if is_xl_controlnet else variant_list_controlnet[11]
#            return result, result, variant_list_controlnet[20] if is_xl_controlnet else variant_list_controlnet[11]

def canny_controlnet(image, low_threshold, high_threshold):
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

def tile_controlnet(image, is_xl_controlnet, modelid_controlnet):
    image = Image.fromarray(image)
    width, height = image.size
    if (is_xl_controlnet) and ("TURBO" not in modelid_controlnet.upper()):
        dim_size = correct_size(width, height, 1024)
    else :
        dim_size = correct_size(width, height, 512)
    image = image.convert("RGB")
    image = image.resize((dim_size[0], dim_size[1]), resample=Image.LANCZOS)
    return image

def qr_controlnet(image, switch):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, image) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    if (switch == 1):
        image = cv2.bitwise_not(image)
    return image

@metrics_decoration
def image_controlnet(
    modelid_controlnet,
    sampler_controlnet,
    prompt_controlnet,
    negative_prompt_controlnet,
    num_images_per_prompt_controlnet,
    num_prompt_controlnet,
    guidance_scale_controlnet,
    num_inference_step_controlnet,
    height_controlnet,
    width_controlnet,
    seed_controlnet,
    low_threshold_controlnet,
    high_threshold_controlnet,
    strength_controlnet,
    start_controlnet,
    stop_controlnet,
    use_gfpgan_controlnet,
    preprocessor_controlnet,
    variant_controlnet,
    img_preview_controlnet,
    nsfw_filter, 
    tkme_controlnet,
    clipskip_controlnet,
    use_ays_controlnet,
    lora_model_controlnet,
    lora_weight_controlnet,
    lora_model2_controlnet,
    lora_weight2_controlnet,
    lora_model3_controlnet,
    lora_weight3_controlnet,
    lora_model4_controlnet,
    lora_weight4_controlnet,
    lora_model5_controlnet,
    lora_weight5_controlnet,
    txtinv_controlnet,
    progress_controlnet=gr.Progress(track_tqdm=True)
    ):

    print(">>>[ControlNet 🖼️ ]: starting module")

    modelid_controlnet = model_cleaner_sd(modelid_controlnet)

    lora_model_controlnet = model_cleaner_lora(lora_model_controlnet)
    lora_model_controlnet = model_cleaner_lora(lora_model_controlnet)
    lora_model2_controlnet = model_cleaner_lora(lora_model2_controlnet)
    lora_model3_controlnet = model_cleaner_lora(lora_model3_controlnet)
    lora_model4_controlnet = model_cleaner_lora(lora_model4_controlnet)
    lora_model5_controlnet = model_cleaner_lora(lora_model5_controlnet)

    lora_array = []
    lora_weight_array = []

    if lora_model_controlnet != "":
        if (is_sd3(modelid_controlnet) or is_flux(modelid_controlnet)) and ((lora_model_controlnet == "ByteDance/Hyper-SD") or (lora_model_controlnet == "RED-AIGC/TDD")):
            lora_weight_controlnet = 0.12
        lora_array.append(f"{lora_model_controlnet}")
        lora_weight_array.append(float(lora_weight_controlnet))
    if lora_model2_controlnet != "":
        lora_array.append(f"{lora_model2_controlnet}")
        lora_weight_array.append(float(lora_weight2_controlnet))
    if lora_model3_controlnet != "":
        lora_array.append(f"{lora_model3_controlnet}")
        lora_weight_array.append(float(lora_weight3_controlnet))
    if lora_model4_controlnet != "":
        lora_array.append(f"{lora_model4_controlnet}")
        lora_weight_array.append(float(lora_weight4_controlnet))
    if lora_model5_controlnet != "":
        lora_array.append(f"{lora_model5_controlnet}")
        lora_weight_array.append(float(lora_weight5_controlnet))

    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_controlnet, device_controlnet, nsfw_filter)

    if clipskip_controlnet == 0:
       clipskip_controlnet = None

    if ("turbo" in modelid_controlnet):
        is_turbo_controlnet: bool = True
    else :
        is_turbo_controlnet: bool = False

    if is_sdxl(modelid_controlnet):
        is_xl_controlnet: bool = True
    else :
        is_xl_controlnet: bool = False

    if is_sd3(modelid_controlnet):
        is_sd3_controlnet: bool = True
    else :
        is_sd3_controlnet: bool = False

    if is_bin(modelid_controlnet):
        is_bin_controlnet: bool = True
    else :
        is_bin_controlnet: bool = False

    if is_flux(modelid_controlnet):
        is_flux_controlnet: bool = True
    else :
        is_flux_controlnet: bool = False

    if is_sd3_controlnet:
        controlnet = SD3ControlNetModel.from_pretrained(
            variant_controlnet,
            cache_dir=model_path_base_controlnet,
            torch_dtype=model_arch,
#            variant="fp16" if (variant_controlnet == "TheMistoAI/MistoLine" or variant_controlnet == "ValouF-pimento/ControlNet_SDXL_tile_upscale") else None,
    #        use_safetensors=True,
            resume_download=True,
            local_files_only=True if offline_test() else None
            )
    elif is_flux_controlnet:
        controlnet = FluxControlNetModel.from_pretrained(
            variant_controlnet,
            cache_dir=model_path_base_controlnet,
            torch_dtype=model_arch,
            resume_download=True,
            local_files_only=True if offline_test() else None
            )
    else:
        controlnet = ControlNetModel.from_pretrained(
            variant_controlnet,
            cache_dir=model_path_base_controlnet,
            torch_dtype=model_arch,
            variant="fp16" if (variant_controlnet == "TheMistoAI/MistoLine" or variant_controlnet == "ValouF-pimento/ControlNet_SDXL_tile_upscale") else None,
    #        use_safetensors=True,
            resume_download=True,
            local_files_only=True if offline_test() else None
            )

#    img_preview_controlnet = Image.open(img_preview_controlnet)
    strength_controlnet = float(strength_controlnet)
    start_controlnet = float(start_controlnet)
    stop_controlnet = float(stop_controlnet)

    if (num_inference_step_controlnet >= 10) and use_ays_controlnet:
        if is_sdxl(modelid_controlnet):
            sampling_schedule_controlnet = AysSchedules["StableDiffusionXLTimesteps"]
            sampler_controlnet = "DPM++ SDE"
        else:
            sampling_schedule_controlnet = AysSchedules["StableDiffusionTimesteps"]
            sampler_controlnet = "Euler"
        num_inference_step_controlnet = 10
    else:
        sampling_schedule_controlnet = None

    if (is_xl_controlnet == True) :
        if modelid_controlnet[0:9] == "./models/" :
            pipe_controlnet = StableDiffusionXLControlNetPipeline.from_single_file(
                modelid_controlnet,
                controlnet=controlnet,
                torch_dtype=model_arch,
                use_safetensors=True if not is_bin_controlnet else False,
#                load_safety_checker=False if (nsfw_filter_final == None) else True,
                local_files_only=True if offline_test() else None
#                safety_checker=nsfw_filter_final,
#                feature_extractor=feat_ex
            )
        else :
            pipe_controlnet = StableDiffusionXLControlNetPipeline.from_pretrained(
                modelid_controlnet,
                controlnet=controlnet,
                cache_dir=model_path_controlnet,
                torch_dtype=model_arch,
                use_safetensors=True if not is_bin_controlnet else False,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
    elif (is_sd3_controlnet == True) :
        if modelid_controlnet[0:9] == "./models/" :
            pipe_controlnet = StableDiffusion3ControlNetPipeline.from_single_file(
                modelid_controlnet,
                controlnet=controlnet,
                torch_dtype=model_arch,
                use_safetensors=True if not is_bin_controlnet else False,
#                load_safety_checker=False if (nsfw_filter_final == None) else True,
                local_files_only=True if offline_test() else None
#                safety_checker=nsfw_filter_final,
#                feature_extractor=feat_ex
            )
        else :
            pipe_controlnet = StableDiffusion3ControlNetPipeline.from_pretrained(
                modelid_controlnet,
                controlnet=controlnet,
                cache_dir=model_path_controlnet,
                torch_dtype=model_arch,
                use_safetensors=True if not is_bin_controlnet else False,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
    elif is_flux_controlnet:
        if modelid_controlnet[0:9] == "./models/" :
            pipe_controlnet = FluxControlNetPipeline.from_single_file(
                modelid_controlnet,
                controlnet=controlnet,
                torch_dtype=model_arch,
                use_safetensors=True if not is_bin_controlnet else False,
                local_files_only=True if offline_test() else None
            )
        else :
            pipe_controlnet = FluxControlNetPipeline.from_pretrained(
                modelid_controlnet,
                controlnet=controlnet,
                cache_dir=model_path_flux_controlnet,
                torch_dtype=model_arch,
                use_safetensors=True if not is_bin_controlnet else False,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
    else :
        if modelid_controlnet[0:9] == "./models/" :
            pipe_controlnet = StableDiffusionControlNetPipeline.from_single_file(
                modelid_controlnet,
                controlnet=controlnet,
                torch_dtype=model_arch,
                use_safetensors=True if not is_bin_controlnet else False,
#                load_safety_checker=False if (nsfw_filter_final == None) else True,
                local_files_only=True if offline_test() else None
#                safety_checker=nsfw_filter_final,
#                feature_extractor=feat_ex
            )
        else :        
            pipe_controlnet = StableDiffusionControlNetPipeline.from_pretrained(
                modelid_controlnet,
                controlnet=controlnet,
                cache_dir=model_path_controlnet,
                torch_dtype=model_arch,
                use_safetensors=True if not is_bin_controlnet else False,
                safety_checker=nsfw_filter_final,
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
   
    pipe_controlnet = schedulerer(pipe_controlnet, sampler_controlnet)
    pipe_controlnet.enable_attention_slicing("max")
    if not is_sd3_controlnet and not is_flux_controlnet:
        tomesd.apply_patch(pipe_controlnet, ratio=tkme_controlnet)
    if device_label_controlnet == "cuda" :
        pipe_controlnet.enable_sequential_cpu_offload()
    else : 
        pipe_controlnet = pipe_controlnet.to(device_controlnet)
    if not is_sd3_controlnet and not is_flux_controlnet:
        pipe_controlnet.enable_vae_slicing()

    adapters_list = []

    if len(lora_array) != 0:
        for e in range(len(lora_array)):
            model_list_lora_controlnet = lora_model_list(modelid_controlnet)
            if lora_array[e][0:9] == "./models/":
                pipe_controlnet.load_lora_weights(
                    os.path.dirname(lora_array[e]),
                    weight_name=model_list_lora_controlnet[lora_array[e]][0],
                    use_safetensors=True,
                    adapter_name=f"adapter{e}",
                    local_files_only=True if offline_test() else None,
                )
            else:
                if is_xl_controlnet:
                    lora_model_path = model_path_lora_sdxl
                elif is_sd3_controlnet:
                    lora_model_path = model_path_lora_sd3
                elif is_flux_controlnet:
                    lora_model_path = model_path_lora_flux
                else: 
                    lora_model_path = model_path_lora_sd

                local_lora_controlnet = hf_hub_download(
                    repo_id=lora_array[e],
                    filename=model_list_lora_controlnet[lora_array[e]][0],
                    cache_dir=lora_model_path,
                    resume_download=True,
                    local_files_only=True if offline_test() else None,
                )

                pipe_controlnet.load_lora_weights(
                    lora_array[e],
                    weight_name=model_list_lora_controlnet[lora_array[e]][0],
                    cache_dir=lora_model_path,
                    use_safetensors=True,
                    adapter_name=f"adapter{e}",
                )
            adapters_list.append(f"adapter{e}")
        pipe_controlnet.set_adapters(adapters_list, adapter_weights=lora_weight_array)

    if txtinv_controlnet != "":
        model_list_txtinv_controlnet = txtinv_list(modelid_controlnet)
        weight_controlnet = model_list_txtinv_controlnet[txtinv_controlnet][0]
        token_controlnet =  model_list_txtinv_controlnet[txtinv_controlnet][1]
        if txtinv_controlnet[0:9] == "./models/":
            model_path_txtinv = "./models/TextualInversion"
            pipe_controlnet.load_textual_inversion(
                txtinv_controlnet,
                weight_name=weight_controlnet,
                use_safetensors=True,
                token=token_controlnet,
                local_files_only=True if offline_test() else None,
            )
        else:
            if is_xl_controlnet:
                model_path_txtinv = "./models/TextualInversion/SDXL"
            else: 
                model_path_txtinv = "./models/TextualInversion/SD"
            pipe_controlnet.load_textual_inversion(
                txtinv_controlnet,
                weight_name=weight_controlnet,
                cache_dir=model_path_txtinv,
                use_safetensors=True,
                token=token_controlnet,
                resume_download=True,
                local_files_only=True if offline_test() else None,
            )

    if seed_controlnet == 0:
        random_seed = random.randrange(0, 10000000000, 1)
        final_seed = random_seed
    else:
        final_seed = seed_controlnet
    generator = []
    for k in range(num_prompt_controlnet):
        generator.append([torch.Generator(device_controlnet).manual_seed(final_seed + (k*num_images_per_prompt_controlnet) + l ) for l in range(num_images_per_prompt_controlnet)])

    if (is_xl_controlnet or is_sd3_controlnet or is_flux_controlnet) and not is_turbo_controlnet:
        dim_size = correct_size(width_controlnet, height_controlnet, 1024)
    else :
        dim_size = correct_size(width_controlnet, height_controlnet, 512)
    image_input = PIL.Image.open(img_preview_controlnet)
    image_input = image_input.convert("RGB")
    image_input = image_input.resize((dim_size[0], dim_size[1]))
    width_controlnet, height_controlnet = dim_size

    prompt_controlnet = str(prompt_controlnet)
    negative_prompt_controlnet = str(negative_prompt_controlnet)
    if prompt_controlnet == "None":
        prompt_controlnet = ""
    if negative_prompt_controlnet == "None":
        negative_prompt_controlnet = ""

    if (is_xl_controlnet == True) :
        compel = Compel(
            tokenizer=[pipe_controlnet.tokenizer, pipe_controlnet.tokenizer_2], 
            text_encoder=[pipe_controlnet.text_encoder, pipe_controlnet.text_encoder_2], 
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, 
            requires_pooled=[False, True],
            device=device_controlnet,
        )
        conditioning, pooled = compel(prompt_controlnet)
        neg_conditioning, neg_pooled = compel(negative_prompt_controlnet)
        [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])
    elif is_sd3_controlnet or is_flux_controlnet:
        pass
    else : 
        compel = Compel(tokenizer=pipe_controlnet.tokenizer, text_encoder=pipe_controlnet.text_encoder, truncate_long_prompts=False, device=device_controlnet)
        conditioning = compel.build_conditioning_tensor(prompt_controlnet)
        neg_conditioning = compel.build_conditioning_tensor(negative_prompt_controlnet)
        [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])

    union_flux_mode = None
    if (is_flux_controlnet and variant_controlnet == "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro"):
        if preprocessor_controlnet in ["canny", "lineart_anime", "lineart_coarse", "lineart_realistic", "mlsd", "scribble_hed", "scribble_pidinet", "softedge_hed", "softedge_hedsafe", "softedge_pidinet", "softedge_pidsafe"]:
            union_flux_mode=0
        elif preprocessor_controlnet in ["tile"]:
            union_flux_mode=1
        elif preprocessor_controlnet in ["depth_leres", "depth_leres++", "depth_midas", "normal_bae"]:
            union_flux_mode=2
        elif preprocessor_controlnet in ["openpose", "openpose_face", "openpose_faceonly", "openpose_full", "openpose_hand"]:
            union_flux_mode=4
        else:
            union_flux_mode=None

    final_image = []
    final_seed = []
    for i in range (num_prompt_controlnet):
        if (is_turbo_controlnet == True):
            image = pipe_controlnet(
                prompt=prompt_controlnet,
                image=image_input,
                height=height_controlnet,
                width=width_controlnet,
                num_images_per_prompt=num_images_per_prompt_controlnet,
                num_inference_steps=num_inference_step_controlnet,
                timesteps=sampling_schedule_controlnet,
                guidance_scale=guidance_scale_controlnet,
                controlnet_conditioning_scale=strength_controlnet,
                control_guidance_start=start_controlnet,
                control_guidance_end=stop_controlnet,
                generator=generator[i],
                callback_on_step_end=check_controlnet, 
                callback_on_step_end_tensor_inputs=['latents'],
            ).images
        elif (is_xl_controlnet == True) : 
            image = pipe_controlnet(
                prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled,
                negative_prompt_embeds=neg_conditioning,
                negative_pooled_prompt_embeds=neg_pooled,
                image=image_input,
                height=height_controlnet,
                width=width_controlnet,
                num_images_per_prompt=num_images_per_prompt_controlnet,
                num_inference_steps=num_inference_step_controlnet,
                timesteps=sampling_schedule_controlnet,
                guidance_scale=guidance_scale_controlnet,
                controlnet_conditioning_scale=strength_controlnet,
                control_guidance_start=start_controlnet,
                control_guidance_end=stop_controlnet,                
                generator=generator[i],
                callback_on_step_end=check_controlnet,
                callback_on_step_end_tensor_inputs=['latents'],
            ).images
        elif is_sd3_controlnet:
            image = pipe_controlnet(
                prompt=prompt_controlnet,
                negative_prompt=negative_prompt_controlnet,
                control_image=image_input,
                height=height_controlnet,
                width=width_controlnet,
                num_images_per_prompt=num_images_per_prompt_controlnet,
                num_inference_steps=num_inference_step_controlnet,
                timesteps=sampling_schedule_controlnet,
                guidance_scale=guidance_scale_controlnet,
                controlnet_conditioning_scale=strength_controlnet,
                control_guidance_start=start_controlnet,
                control_guidance_end=stop_controlnet,
                generator=generator[i],
                callback_on_step_end=check_controlnet,
                callback_on_step_end_tensor_inputs=['latents'],
            ).images
        elif is_flux_controlnet:
            image = pipe_controlnet(
                prompt=prompt_controlnet,
                control_image=image_input,
                height=height_controlnet,
                width=width_controlnet,
                max_sequence_length=512,
                num_images_per_prompt=num_images_per_prompt_controlnet,
                num_inference_steps=num_inference_step_controlnet,
#                timesteps=sampling_schedule_controlnet,
                guidance_scale=guidance_scale_controlnet,
                controlnet_conditioning_scale=strength_controlnet,
                control_guidance_start=start_controlnet,
                control_guidance_end=stop_controlnet,
                control_mode=union_flux_mode,
                generator=generator[i],
                callback_on_step_end=check_controlnet,
                callback_on_step_end_tensor_inputs=['latents'],
            ).images
        else :
            image = pipe_controlnet(
                prompt_embeds=conditioning,
                negative_prompt_embeds=neg_conditioning,
                image=image_input,
                height=height_controlnet,
                width=width_controlnet,
                num_images_per_prompt=num_images_per_prompt_controlnet,
                num_inference_steps=num_inference_step_controlnet,
                timesteps=sampling_schedule_controlnet,
                guidance_scale=guidance_scale_controlnet,
                controlnet_conditioning_scale=strength_controlnet,
                control_guidance_start=start_controlnet,
                control_guidance_end=stop_controlnet,
                generator=generator[i],
                clip_skip=clipskip_controlnet,
                callback_on_step_end=check_controlnet,
                callback_on_step_end_tensor_inputs=['latents'],
            ).images

        for j in range(len(image)):
            if is_xl_controlnet or is_sd3_controlnet or is_flux_controlnet or (modelid_controlnet[0:9] == "./models/"):
                image[j] = safety_checker_sdxl(model_path_controlnet, image[j], nsfw_filter)
            seed_id = random_seed + i*num_images_per_prompt_controlnet + j if (seed_controlnet == 0) else seed_controlnet + i*num_images_per_prompt_controlnet + j
            savename = name_seeded_image(seed_id)
            if use_gfpgan_controlnet == True :
                image[j] = image_gfpgan_mini(image[j])
            image[j].save(savename)
            final_image.append(savename)
            final_seed.append(seed_id)

    print(f">>>[ControlNet 🖼️ ]: generated {num_prompt_controlnet} batch(es) of {num_images_per_prompt_controlnet}")
    reporting_controlnet = f">>>[ControlNet 🖼️ ]: "+\
        f"Settings : Model={modelid_controlnet} | "+\
        f"XL model={is_xl_controlnet} | "+\
        f"Sampler={sampler_controlnet} | "+\
        f"Steps={num_inference_step_controlnet} | "+\
        f"CFG scale={guidance_scale_controlnet} | "+\
        f"Size={width_controlnet}x{height_controlnet} | "+\
        f"ControlNet strength={strength_controlnet} | "+\
        f"Start ControlNet={start_controlnet} | "+\
        f"Stop ControlNet={stop_controlnet} | "+\
        f"GFPGAN={use_gfpgan_controlnet} | "+\
        f"Token merging={tkme_controlnet} | "+\
        f"CLIP skip={clipskip_controlnet} | "+\
        f"AYS={use_ays_controlnet} | "+\
        f"LoRA model={lora_array} | "+\
        f"LoRA weight={lora_weight_array} | "+\
        f"Textual inversion={txtinv_controlnet} | "+\
        f"nsfw_filter={bool(int(nsfw_filter))} | "+\
        f"Pre-processor={preprocessor_controlnet} | "+\
        f"ControlNet model={variant_controlnet} | "+\
        f"Prompt={prompt_controlnet} | "+\
        f"Negative prompt={negative_prompt_controlnet} | "+\
        f"Seed List="+ ', '.join([f"{final_seed[m]}" for m in range(len(final_seed))])
    print(reporting_controlnet)

    savename_controlnet = f"outputs/controlnet.png"
    image_input.save(savename_controlnet)
    final_image.append(savename_controlnet)

    exif_writer_png(reporting_controlnet, final_image)

    del nsfw_filter_final, feat_ex, controlnet, img_preview_controlnet, pipe_controlnet, generator, image_input, image
    clean_ram()

    print(f">>>[ControlNet 🖼️ ]: leaving module")
    return final_image, final_image
