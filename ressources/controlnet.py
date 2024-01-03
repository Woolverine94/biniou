# https://github.com/Woolverine94/biniou
# controlnet.py
import gradio as gr
import os
import cv2
import torch
from diffusers import StableDiffusionControlNetPipeline, StableDiffusionXLControlNetPipeline, ControlNetModel
from compel import Compel, ReturnedEmbeddingsType
import time
import random
from ressources.scheduler import *
from ressources.gfpgan import *
from controlnet_aux.processor import Processor
import tomesd

device_label_controlnet, model_arch = detect_device()
device_controlnet = torch.device(device_label_controlnet)

# Gestion des modèles
model_path_controlnet = "./models/Stable_Diffusion/"
os.makedirs(model_path_controlnet, exist_ok=True)
model_list_controlnet = []

for filename in os.listdir(model_path_controlnet):
    f = os.path.join(model_path_controlnet, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
        model_list_controlnet.append(f)

model_list_controlnet_builtin = [
    "SG161222/Realistic_Vision_V3.0_VAE",
#    "stabilityai/sd-turbo",
    "stabilityai/sdxl-turbo",
    "dataautogpt3/OpenDalleV1.1",
    "digiplay/AbsoluteReality_v1.8.1",
    "segmind/Segmind-Vega",
    "segmind/SSD-1B",
#    "ckpt/anything-v4.5-vae-swapped",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "runwayml/stable-diffusion-v1-5",
    "nitrosocke/Ghibli-Diffusion", 
]

for k in range(len(model_list_controlnet_builtin)):
    model_list_controlnet.append(model_list_controlnet_builtin[k])

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
    "patrickvonplaten/controlnet-canny-sdxl-1.0",
    "patrickvonplaten/controlnet-depth-sdxl-1.0",
    "zbulrush/controlnet-sd-xl-1.0-lineart",
    "thibaud/controlnet-openpose-sdxl-1.0",
    "SargeZT/controlnet-sd-xl-1.0-softedge-dexined",
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

    if (('xl' or 'XL' or 'Xl' or 'xL') in modelid_controlnet or (modelid_controlnet == "segmind/SSD-1B") or (modelid_controlnet == "segmind/Segmind-Vega") or (modelid_controlnet == "dataautogpt3/OpenDalleV1.1")):
        is_xl_controlnet: bool = True
    else :
        is_xl_controlnet: bool = False

    img_source_controlnet = Image.open(img_source_controlnet)
    img_source_controlnet = np.array(img_source_controlnet)
    processor_controlnet = Processor(preprocessor_controlnet)

    match preprocessor_controlnet:
        case "canny":
            result = canny_controlnet(img_source_controlnet, low_threshold_controlnet, high_threshold_controlnet)
            return result, result, variant_list_controlnet[9] if is_xl_controlnet else variant_list_controlnet[0]
        case "depth_leres":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            return result, result, variant_list_controlnet[10] if is_xl_controlnet else variant_list_controlnet[1]
        case "depth_leres++":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            return result, result, variant_list_controlnet[10] if is_xl_controlnet else variant_list_controlnet[1]
        case "depth_midas":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            return result, result, variant_list_controlnet[10] if is_xl_controlnet else variant_list_controlnet[1]
        case "depth_zoe":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            return result, result, variant_list_controlnet[10] if is_xl_controlnet else variant_list_controlnet[1]
        case "lineart_anime":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            return result, result, variant_list_controlnet[11] if is_xl_controlnet else variant_list_controlnet[2]
        case "lineart_coarse":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            return result, result, variant_list_controlnet[11] if is_xl_controlnet else variant_list_controlnet[3]
        case "lineart_realistic":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            return result, result, variant_list_controlnet[11] if is_xl_controlnet else variant_list_controlnet[3]
        case "mlsd":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            return result, result, variant_list_controlnet[9] if is_xl_controlnet else variant_list_controlnet[4]
        case "normal_bae":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            return result, result, variant_list_controlnet[10] if is_xl_controlnet else variant_list_controlnet[5]
        case "normal_midas":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            return result, result, variant_list_controlnet[10] if is_xl_controlnet else variant_list_controlnet[5]
        case "openpose":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            return result, result, variant_list_controlnet[12] if is_xl_controlnet else variant_list_controlnet[6]
        case "openpose_face":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            return result, result, variant_list_controlnet[12] if is_xl_controlnet else variant_list_controlnet[6]
        case "openpose_faceonly":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            return result, result, variant_list_controlnet[12] if is_xl_controlnet else variant_list_controlnet[6]
        case "openpose_full":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            return result, result, variant_list_controlnet[12] if is_xl_controlnet else variant_list_controlnet[6]
        case "openpose_hand":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            return result, result, variant_list_controlnet[12] if is_xl_controlnet else variant_list_controlnet[6]
        case "scribble_hed":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            return result, result, variant_list_controlnet[13] if is_xl_controlnet else variant_list_controlnet[7]
        case "scribble_pidinet":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            return result, result, variant_list_controlnet[13] if is_xl_controlnet else variant_list_controlnet[7]
        case "softedge_hed":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            return result, result, variant_list_controlnet[13] if is_xl_controlnet else variant_list_controlnet[8]
        case "softedge_hedsafe":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            return result, result, variant_list_controlnet[13] if is_xl_controlnet else variant_list_controlnet[8]
        case "softedge_pidinet":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            return result, result, variant_list_controlnet[13] if is_xl_controlnet else variant_list_controlnet[8]
        case "softedge_pidsafe":
            result = processor_controlnet(img_source_controlnet, to_pil=True)
            return result, result, variant_list_controlnet[13] if is_xl_controlnet else variant_list_controlnet[8]

def canny_controlnet(image, low_threshold, high_threshold):
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

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
    variant_controlnet,
    img_preview_controlnet,
    nsfw_filter, 
    tkme_controlnet,
    lora_model_controlnet,
    lora_weight_controlnet,
    progress_controlnet=gr.Progress(track_tqdm=True)
    ):

    print(">>>[ControlNet 🖼️ ]: starting module")

    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_controlnet, device_controlnet, nsfw_filter)

    controlnet = ControlNetModel.from_pretrained(
        variant_controlnet,
        cache_dir=model_path_base_controlnet,
        torch_dtype=model_arch,
        use_safetensors=True,
        resume_download=True,
        local_files_only=True if offline_test() else None
        )

#    img_preview_controlnet = Image.open(img_preview_controlnet)
    strength_controlnet = float(strength_controlnet)
    start_controlnet = float(start_controlnet)
    stop_controlnet = float(stop_controlnet)

    if (modelid_controlnet == "stabilityai/sdxl-turbo") or  (modelid_controlnet == "stabilityai/sd-turbo"):
        is_xlturbo_controlnet: bool = True
    else :
        is_xlturbo_controlnet: bool = False

    if (('xl' or 'XL' or 'Xl' or 'xL') in modelid_controlnet or (modelid_controlnet == "segmind/SSD-1B") or (modelid_controlnet == "segmind/Segmind-Vega") or (modelid_controlnet == "dataautogpt3/OpenDalleV1.1")):
        is_xl_controlnet: bool = True
    else :        
        is_xl_controlnet: bool = False

    if (is_xl_controlnet == True) :
        if modelid_controlnet[0:9] == "./models/" :
            pipe_controlnet = StableDiffusionXLControlNetPipeline.from_single_file(
                modelid_controlnet,
                controlnet=controlnet,
                torch_dtype=model_arch,
                use_safetensors=True,
                safety_checker=nsfw_filter_final,
                feature_extractor=feat_ex
            )
        else :
            pipe_controlnet = StableDiffusionXLControlNetPipeline.from_pretrained(
                modelid_controlnet,
                controlnet=controlnet,
                cache_dir=model_path_controlnet,
                torch_dtype=model_arch,
                use_safetensors=True, 
                safety_checker=nsfw_filter_final,
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
    else :
        if modelid_controlnet[0:9] == "./models/" :
            pipe_controlnet = StableDiffusionControlNetPipeline.from_single_file(
                modelid_controlnet,
                controlnet=controlnet,
                torch_dtype=model_arch,
                use_safetensors=True, 
                safety_checker=nsfw_filter_final,
                feature_extractor=feat_ex
            )
        else :        
            pipe_controlnet = StableDiffusionControlNetPipeline.from_pretrained(
                modelid_controlnet,
                controlnet=controlnet,
                cache_dir=model_path_controlnet,
                torch_dtype=model_arch,
                use_safetensors=True, 
                safety_checker=nsfw_filter_final,
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
   
    pipe_controlnet = get_scheduler(pipe=pipe_controlnet, scheduler=sampler_controlnet)
    pipe_controlnet.enable_attention_slicing("max")
    tomesd.apply_patch(pipe_controlnet, ratio=tkme_controlnet)
    if device_label_controlnet == "cuda" :
        pipe_controlnet.enable_sequential_cpu_offload()
    else : 
        pipe_controlnet = pipe_controlnet.to(device_controlnet)
    pipe_controlnet.enable_vae_slicing()

    if lora_model_controlnet != "":
        model_list_lora_controlnet = lora_model_list(modelid_controlnet)
        if modelid_controlnet[0:9] == "./models/":
            pipe_controlnet.load_lora_weights(
                os.path.dirname(lora_model_controlnet),
                weight_name=model_list_lora_controlnet[lora_model_controlnet][0],
                use_safetensors=True,
                adapter_name="adapter1",
            )
        else:
            if is_xl_controlnet:
                lora_model_path = "./models/lora/SDXL"
            else: 
                lora_model_path = "./models/lora/SD"
            pipe_controlnet.load_lora_weights(
                lora_model_controlnet,
                weight_name=model_list_lora_controlnet[lora_model_controlnet][0],
                cache_dir=lora_model_path,
                use_safetensors=True,
                adapter_name="adapter1",
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
        pipe_controlnet.set_adapters(["adapter1"], adapter_weights=[float(lora_weight_controlnet)])

    if seed_controlnet == 0:
        random_seed = random.randrange(0, 10000000000, 1)
        final_seed = random_seed
    else:
        final_seed = seed_controlnet
    generator = []
    for k in range(num_prompt_controlnet):
        generator.append([torch.Generator(device_controlnet).manual_seed(final_seed + (k*num_images_per_prompt_controlnet) + l ) for l in range(num_images_per_prompt_controlnet)])

    if (is_xl_controlnet == True) and not (is_xlturbo_controlnet == True) :
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
        )
        conditioning, pooled = compel(prompt_controlnet)
        neg_conditioning, neg_pooled = compel(negative_prompt_controlnet)
        [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])
    else : 
        compel = Compel(tokenizer=pipe_controlnet.tokenizer, text_encoder=pipe_controlnet.text_encoder, truncate_long_prompts=False)
        conditioning = compel.build_conditioning_tensor(prompt_controlnet)
        neg_conditioning = compel.build_conditioning_tensor(negative_prompt_controlnet)
        [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])
   
    final_image = []
    final_seed = []
    for i in range (num_prompt_controlnet):
        if (is_xlturbo_controlnet == True):
            image = pipe_controlnet(
                prompt=prompt_controlnet,
                image=image_input,
                height=height_controlnet,
                width=width_controlnet,
                num_images_per_prompt=num_images_per_prompt_controlnet,
                num_inference_steps=num_inference_step_controlnet,
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
                guidance_scale=guidance_scale_controlnet,
                controlnet_conditioning_scale=strength_controlnet,
                control_guidance_start=start_controlnet,
                control_guidance_end=stop_controlnet,                
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
                guidance_scale=guidance_scale_controlnet,
                controlnet_conditioning_scale=strength_controlnet,
                control_guidance_start=start_controlnet,
                control_guidance_end=stop_controlnet,
                generator=generator[i],
                callback_on_step_end=check_controlnet,
                callback_on_step_end_tensor_inputs=['latents'],
            ).images

        for j in range(len(image)):
            timestamp = time.time()
            seed_id = random_seed + i*num_images_per_prompt_controlnet + j if (seed_controlnet == 0) else seed_controlnet + i*num_images_per_prompt_controlnet + j
            savename = f"outputs/{seed_id}_{timestamp}.png"
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
        f"nsfw_filter={bool(int(nsfw_filter))} | "+\
        f"ControlNet model={variant_controlnet} | "+\
        f"Prompt={prompt_controlnet} | "+\
        f"Negative prompt={negative_prompt_controlnet} | "+\
        f"Seed List="+ ', '.join([f"{final_seed[m]}" for m in range(len(final_seed))])
    print(reporting_controlnet)


    savename_controlnet = f"outputs/controlnet.png"
    image_input.save(savename_controlnet)
    final_image.append(savename_controlnet)

    del nsfw_filter_final, feat_ex, controlnet, img_preview_controlnet, pipe_controlnet, generator, image_input, compel, conditioning, neg_conditioning, image
    clean_ram()

    print(f">>>[ControlNet 🖼️ ]: leaving module")
    return final_image, final_image
