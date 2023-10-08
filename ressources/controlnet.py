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

device_controlnet = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def check_controlnet(step, timestep, latents) :
    global stop_controlnet
    if stop_controlnet == False :
        return
    elif stop_controlnet == True :
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

    if ('xl' or 'XL' or 'Xl' or 'xL') in modelid_controlnet :
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
    progress_controlnet=gr.Progress(track_tqdm=True)
    ):

    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_controlnet, device_controlnet, nsfw_filter)
    
    controlnet = ControlNetModel.from_pretrained(
        variant_controlnet, 
        cache_dir=model_path_base_controlnet, 
        torch_dtype=torch.float32, 
        use_safetensors=True,
        resume_download=True,
        local_files_only=True if offline_test() else None        
        )
        
    img_preview_controlnet = Image.open(img_preview_controlnet)
    strength_controlnet = float(strength_controlnet)
    start_controlnet = float(start_controlnet)
    stop_controlnet = float(stop_controlnet)
    
    if ('xl' or 'XL' or 'Xl' or 'xL') in modelid_controlnet :
        is_xl_controlnet: bool = True
    else :        
        is_xl_controlnet: bool = False
        
    if (is_xl_controlnet == True) :
        if modelid_controlnet[0:9] == "./models/" :
            pipe_controlnet = StableDiffusionXLControlNetPipeline.from_single_file(
                modelid_controlnet, 
                controlnet=controlnet, 
                torch_dtype=torch.float32, 
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex
            )
        else :        
            pipe_controlnet = StableDiffusionXLControlNetPipeline.from_pretrained(
                modelid_controlnet, 
                controlnet=controlnet, 
                cache_dir=model_path_controlnet, 
                torch_dtype=torch.float32, 
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
                torch_dtype=torch.float32, 
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex
            )
        else :        
            pipe_controlnet = StableDiffusionControlNetPipeline.from_pretrained(
                modelid_controlnet, 
                controlnet=controlnet, 
                cache_dir=model_path_controlnet, 
                torch_dtype=torch.float32, 
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
   
    pipe_controlnet = get_scheduler(pipe=pipe_controlnet, scheduler=sampler_controlnet)
    pipe_controlnet = pipe_controlnet.to(device_controlnet)
    pipe_controlnet.enable_attention_slicing("max")
    tomesd.apply_patch(pipe_controlnet, ratio=tkme_controlnet)

    if seed_controlnet == 0:
        random_seed = torch.randint(0, 10000000000, (1,))
        generator = torch.manual_seed(random_seed)
    else:
        generator = torch.manual_seed(seed_controlnet)

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
    
    for i in range (num_prompt_controlnet):
        if (is_xl_controlnet == True) : 
            image = pipe_controlnet(
                prompt_embeds=conditioning, 
                pooled_prompt_embeds=pooled, 
                negative_prompt_embeds=neg_conditioning,
                negative_pooled_prompt_embeds=neg_pooled,            
                image=img_preview_controlnet,
                height=height_controlnet,
                width=width_controlnet,
                num_images_per_prompt=num_images_per_prompt_controlnet,
                num_inference_steps=num_inference_step_controlnet,
                guidance_scale=guidance_scale_controlnet,
                controlnet_conditioning_scale=strength_controlnet,
                control_guidance_start=start_controlnet,
                control_guidance_end=stop_controlnet,                
                generator=generator,
                callback=check_controlnet,
            ).images
        else :            
            image = pipe_controlnet(
                prompt_embeds=conditioning,
                negative_prompt_embeds=neg_conditioning,
                image=img_preview_controlnet,                
                height=height_controlnet,
                width=width_controlnet,
                num_images_per_prompt=num_images_per_prompt_controlnet,
                num_inference_steps=num_inference_step_controlnet,
                guidance_scale=guidance_scale_controlnet,
                controlnet_conditioning_scale=strength_controlnet,
                control_guidance_start=start_controlnet,
                control_guidance_end=stop_controlnet,
                generator=generator,
                callback=check_controlnet,
            ).images

        for j in range(len(image)):
            timestamp = time.time()
            savename = f"outputs/{timestamp}.png"
            if use_gfpgan_controlnet == True :
                image[j] = image_gfpgan_mini(image[j])
            image[j].save(savename)
            final_image.append(image[j])

    final_image.append(img_preview_controlnet)

    del nsfw_filter_final, feat_ex, controlnet, img_preview_controlnet, pipe_controlnet, generator, compel, conditioning, neg_conditioning, image 
    clean_ram()
    
    return final_image, final_image
