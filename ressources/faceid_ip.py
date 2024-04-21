# https://github.com/Woolverine94/biniou
# faceid_ip.py
import gradio as gr
import os
import PIL
import cv2
# from insightface.app import FaceAnalysis
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image
from photomaker import PhotoMakerStableDiffusionXLPipeline
from huggingface_hub import snapshot_download, hf_hub_download
from compel import Compel, ReturnedEmbeddingsType
import random
from ressources.common import *
from ressources.gfpgan import *
import tomesd
import requests

device_label_faceid_ip, model_arch = detect_device()
device_faceid_ip = torch.device(device_label_faceid_ip)

# Gestion des modèles
model_path_faceid_ip = "./models/Stable_Diffusion/"
model_path_ipa_faceid_ip = "./models/Ip-Adapters"
os.makedirs(model_path_faceid_ip, exist_ok=True)
os.makedirs(model_path_ipa_faceid_ip, exist_ok=True)
model_path_community_faceid_ip = "./.community"
os.makedirs(model_path_community_faceid_ip, exist_ok=True)

# if offline_test() == True:
url_community_faceid_ip = "https://raw.githubusercontent.com/huggingface/diffusers/c0f5346a207bdbf1f7be0b3a539fefae89287ca4/examples/community/ip_adapter_face_id.py"
response_community_faceid_ip = requests.get(url_community_faceid_ip)
filename_community_faceid_ip = model_path_community_faceid_ip+ "/ip_adapter_face_id.py"
with open(filename_community_faceid_ip, "wb") as f:
    f.write(response_community_faceid_ip.content)

model_list_faceid_ip = []

# .from_single_file NOT compatible with FaceID community pipeline
# for filename in os.listdir(model_path_faceid_ip):
#     f = os.path.join(model_path_faceid_ip, filename)
#     if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
#         model_list_faceid_ip.append(f)

model_list_faceid_ip_builtin = [
    "SG161222/Realistic_Vision_V3.0_VAE",
    "playgroundai/playground-v2-512px-base",
    "playgroundai/playground-v2-1024px-aesthetic",
    "playgroundai/playground-v2.5-1024px-aesthetic",
    "SG161222/RealVisXL_V3.0",
    "SG161222/RealVisXL_V4.0_Lightning",
    "cagliostrolab/animagine-xl-3.1",
#    "stabilityai/sd-turbo",
#    "stabilityai/sdxl-turbo",
#    "dataautogpt3/OpenDalleV1.1",
#    "dataautogpt3/ProteusV0.4",
    "dataautogpt3/ProteusV0.4-Lightning",
    "digiplay/AbsoluteReality_v1.8.1",
#    "segmind/Segmind-Vega",
#    "segmind/SSD-1B",
    "gsdf/Counterfeit-V2.5",
#    "ckpt/anything-v4.5-vae-swapped",
    "stabilityai/stable-diffusion-xl-base-1.0",
#    "stabilityai/stable-diffusion-xl-refiner-1.0",
    "runwayml/stable-diffusion-v1-5",
    "nitrosocke/Ghibli-Diffusion",
]

for k in range(len(model_list_faceid_ip_builtin)):
    model_list_faceid_ip.append(model_list_faceid_ip_builtin[k])

# Bouton Cancel
stop_faceid_ip = False

def initiate_stop_faceid_ip() :
    global stop_faceid_ip
    stop_faceid_ip = True

def check_faceid_ip(pipe, step_index, timestep, callback_kwargs): 
    global stop_faceid_ip
    if stop_faceid_ip == True:
        print(">>>[Photobooth 🖼️ ]: generation canceled by user")
        stop_faceid_ip = False
        pipe._interrupt = True
    return callback_kwargs

# def face_extractor(image_src):
#     app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# #    app.prepare(ctx_id=0, det_size=(640, 640))
#     app.prepare(ctx_id=0, det_size=(320, 320))
#     
#     image = cv2.imread(image_src)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     faces = app.get(image)
#     faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
#     return faceid_embeds

@metrics_decoration
def image_faceid_ip(
    modelid_faceid_ip, 
    sampler_faceid_ip, 
    img_faceid_ip, 
    prompt_faceid_ip, 
    negative_prompt_faceid_ip, 
    num_images_per_prompt_faceid_ip, 
    num_prompt_faceid_ip, 
    guidance_scale_faceid_ip, 
    denoising_strength_faceid_ip, 
    num_inference_step_faceid_ip, 
    height_faceid_ip, 
    width_faceid_ip, 
    seed_faceid_ip, 
    use_gfpgan_faceid_ip, 
    nsfw_filter, 
    tkme_faceid_ip,    
    lora_model_faceid_ip,
    lora_weight_faceid_ip,
    txtinv_faceid_ip,
    progress_faceid_ip=gr.Progress(track_tqdm=True)
    ):

    print(">>>[Photobooth 🖼️ ]: starting module")

    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_faceid_ip, device_faceid_ip, nsfw_filter)

    if ("turbo" in modelid_faceid_ip):
        is_turbo_faceid_ip: bool = True
    else :
        is_turbo_faceid_ip: bool = False

    if (("XL" in modelid_faceid_ip.upper()) or ("LIGHTNING" in modelid_faceid_ip.upper()) or ("PLAYGROUNDAI/PLAYGROUND-V2" in modelid_faceid_ip.upper()) or (modelid_faceid_ip == "segmind/SSD-1B") or (modelid_faceid_ip == "segmind/Segmind-Vega") or (modelid_faceid_ip == "dataautogpt3/OpenDalleV1.1") or (modelid_faceid_ip == "dataautogpt3/ProteusV0.4")):
        is_xl_faceid_ip: bool = True
    else :
        is_xl_faceid_ip: bool = False

    if ("dataautogpt3/ProteusV0.4" in modelid_faceid_ip):
        is_bin_faceid_ip: bool = True
    else :
        is_bin_faceid_ip: bool = False

    if (is_turbo_faceid_ip == True) :
        if modelid_faceid_ip[0:9] == "./models/" :
            pipe_faceid_ip = AutoPipelineForText2Image.from_single_file(
                modelid_faceid_ip, 
                torch_dtype=model_arch,
                use_safetensors=True if not is_bin_faceid_ip else False,
                load_safety_checker=False if (nsfw_filter_final == None) else True,
            )
        else :        
            pipe_faceid_ip = AutoPipelineForText2Image.from_pretrained(
                modelid_faceid_ip, 
                cache_dir=model_path_faceid_ip, 
                torch_dtype=model_arch,
                use_safetensors=True if not is_bin_faceid_ip else False,
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
        pipe_faceid_ip = schedulerer(pipe_faceid_ip, sampler_faceid_ip)
        pipe_faceid_ip.load_photomaker_adapter(
            "TencentARC/PhotoMaker",
            subfolder="",
            weight_name="photomaker-v1.bin",
            cache_dir=model_path_ipa_faceid_ip,
            trigger_word="img",
            use_safetensors=True,
            resume_download=True,
            local_files_only=True if offline_test() else None
        )
        pipe_faceid_ip.id_encoder.to(device_faceid_ip)
        pipe_faceid_ip.fuse_lora()
    elif (is_xl_faceid_ip == True) and (is_turbo_faceid_ip == False) :
        if modelid_faceid_ip[0:9] == "./models/" :
            pipe_faceid_ip = PhotoMakerStableDiffusionXLPipeline.from_single_file(
                modelid_faceid_ip, 
                torch_dtype=model_arch,
                use_safetensors=True if not is_bin_faceid_ip else False,
                load_safety_checker=False if (nsfw_filter_final == None) else True,
            )
        else :        
            pipe_faceid_ip = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
                modelid_faceid_ip, 
                cache_dir=model_path_faceid_ip, 
                torch_dtype=model_arch,
                use_safetensors=True if not is_bin_faceid_ip else False,
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
        pipe_faceid_ip = schedulerer(pipe_faceid_ip, sampler_faceid_ip)
        pipe_faceid_ip.load_photomaker_adapter(
            "TencentARC/PhotoMaker",
            subfolder="",
            weight_name="photomaker-v1.bin",
            cache_dir=model_path_ipa_faceid_ip,
            trigger_word="img",
            use_safetensors=True,
            resume_download=True,
            local_files_only=True if offline_test() else None
        )
        pipe_faceid_ip.id_encoder.to(device_faceid_ip)
        pipe_faceid_ip.fuse_lora()
    else :
        if modelid_faceid_ip[0:9] == "./models/" :
            pipe_faceid_ip = StableDiffusionPipeline.from_single_file(
                modelid_faceid_ip, 
                torch_dtype=model_arch,
                use_safetensors=True if not is_bin_faceid_ip else False,
                load_safety_checker=False if (nsfw_filter_final == None) else True,
#                safety_checker=nsfw_filter_final, 
#                feature_extractor=feat_ex,
#                custom_pipeline=filename_community_faceid_ip,
#                custom_revision=filename_community_faceid_ip,
            )
        else :        
            pipe_faceid_ip = StableDiffusionPipeline.from_pretrained(
                modelid_faceid_ip, 
                cache_dir=model_path_faceid_ip, 
                torch_dtype=model_arch,
                use_safetensors=True if not is_bin_faceid_ip else False,
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
#                custom_pipeline=filename_community_faceid_ip,
#                custom_revision=filename_community_faceid_ip,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
        pipe_faceid_ip = schedulerer(pipe_faceid_ip, sampler_faceid_ip)

    if (is_xl_faceid_ip == True):
#        pipe_faceid_ip.load_ip_adapter_face_id(
#            "h94/IP-Adapter-FaceID",
#            cache_dir=model_path_ipa_faceid_ip,
#            weight_name="ip-adapter-faceid_sdxl.bin",
#            resume_download=True,
#            local_files_only=True if offline_test() else None
#        )
        pass
    else:
#        pipe_faceid_ip.load_ip_adapter_face_id(
        pipe_faceid_ip.load_ip_adapter(
            "h94/IP-Adapter",
            cache_dir=model_path_ipa_faceid_ip,
            subfolder="models",
            weight_name="ip-adapter-plus-face_sd15.safetensors",
            use_safetensors=True,
            resume_download=True,
            local_files_only=True if offline_test() else None
        )

    pipe_faceid_ip.set_ip_adapter_scale(denoising_strength_faceid_ip)
#    pipe_faceid_ip = schedulerer(pipe_faceid_ip, sampler_faceid_ip)
#    pipe_faceid_ip.enable_attention_slicing("max")

    tomesd.apply_patch(pipe_faceid_ip, ratio=tkme_faceid_ip)
    if device_label_faceid_ip == "cuda" :
        pipe_faceid_ip.enable_sequential_cpu_offload()
    else : 
        pipe_faceid_ip = pipe_faceid_ip.to(device_faceid_ip)

    if lora_model_faceid_ip != "":
        model_list_lora_faceid_ip = lora_model_list(modelid_faceid_ip)
        if modelid_faceid_ip[0:9] == "./models/":
            pipe_faceid_ip.load_lora_weights(
                os.path.dirname(lora_model_faceid_ip),
                weight_name=model_list_lora_faceid_ip[lora_model_faceid_ip][0],
                use_safetensors=True,
                adapter_name="adapter1",
            )
        else:
            if is_xl_faceid_ip:
                lora_model_path = "./models/lora/SDXL"
            else: 
                lora_model_path = "./models/lora/SD"
            pipe_faceid_ip.load_lora_weights(
                lora_model_faceid_ip,
                weight_name=model_list_lora_faceid_ip[lora_model_faceid_ip][0],
                cache_dir=lora_model_path,
                use_safetensors=True,
                adapter_name="adapter1",
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
        pipe_faceid_ip.fuse_lora(lora_scale=lora_weight_faceid_ip)
#        pipe_faceid_ip.set_adapters(["adapter1"], adapter_weights=[float(lora_weight_faceid_ip)])

    if txtinv_faceid_ip != "":
        model_list_txtinv_faceid_ip = txtinv_list(modelid_faceid_ip)
        weight_faceid_ip = model_list_txtinv_faceid_ip[txtinv_faceid_ip][0]
        token_faceid_ip =  model_list_txtinv_faceid_ip[txtinv_faceid_ip][1]
        if modelid_faceid_ip[0:9] == "./models/":
            model_path_txtinv = "./models/TextualInversion"
            pipe_faceid_ip.load_textual_inversion(
                txtinv_faceid_ip,
                weight_name=weight_faceid_ip,
                use_safetensors=True,
                token=token_faceid_ip,
            )
        else:
            if is_xl_faceid_ip:
                model_path_txtinv = "./models/TextualInversion/SDXL"
            else: 
                model_path_txtinv = "./models/TextualInversion/SD"
            pipe_faceid_ip.load_textual_inversion(
                txtinv_faceid_ip,
                weight_name=weight_faceid_ip,
                cache_dir=model_path_txtinv,
                use_safetensors=True,
                token=token_faceid_ip,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )

    if seed_faceid_ip == 0:
        random_seed = random.randrange(0, 10000000000, 1)
        final_seed = random_seed
    else:
        final_seed = seed_faceid_ip
    generator = []
    for k in range(num_prompt_faceid_ip):
        generator.append([torch.Generator(device_faceid_ip).manual_seed(final_seed + (k*num_images_per_prompt_faceid_ip) + l ) for l in range(num_images_per_prompt_faceid_ip)])

    prompt_faceid_ip = str(prompt_faceid_ip)
    negative_prompt_faceid_ip = str(negative_prompt_faceid_ip)
    if prompt_faceid_ip == "None":
        prompt_faceid_ip = ""
    if negative_prompt_faceid_ip == "None":
        negative_prompt_faceid_ip = ""

    if (is_xl_faceid_ip == True) :
#        compel = Compel(
#            tokenizer=pipe_faceid_ip.tokenizer_2, 
#            text_encoder=pipe_faceid_ip.text_encoder_2, 
#            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, 
#            requires_pooled=[False, True], 
#            device=device_faceid_ip,
#        )
#        conditioning, pooled = compel(prompt_faceid_ip)
#        neg_conditioning, neg_pooled = compel(negative_prompt_faceid_ip)
#        [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])
        pass
    else :
        compel = Compel(tokenizer=pipe_faceid_ip.tokenizer, text_encoder=pipe_faceid_ip.text_encoder, truncate_long_prompts=False, device=device_faceid_ip)
        conditioning = compel.build_conditioning_tensor(prompt_faceid_ip)
        neg_conditioning = compel.build_conditioning_tensor(negative_prompt_faceid_ip)
        [conditioning, neg_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, neg_conditioning])

    if (is_xl_faceid_ip == False):
#        faceid_embeds_faceid_ip = face_extractor(img_faceid_ip)
        image_input = PIL.Image.open(img_faceid_ip)
        image_input = image_input.convert("RGB")
    else:
        input_id_images_faceid_ip = []
        input_id_images_faceid_ip.append(PIL.Image.open(img_faceid_ip))
        start_merge_step_faceid_ip = round(num_inference_step_faceid_ip - (num_inference_step_faceid_ip*denoising_strength_faceid_ip))

    final_image = []
    final_seed = []
    for i in range (num_prompt_faceid_ip):
        if (is_turbo_faceid_ip == True) :
            image = pipe_faceid_ip(
                input_id_images=input_id_images_faceid_ip,
                prompt=prompt_faceid_ip,
                num_images_per_prompt=num_images_per_prompt_faceid_ip,
                guidance_scale=guidance_scale_faceid_ip,
                strength=denoising_strength_faceid_ip,
                num_inference_steps=num_inference_step_faceid_ip,
                height=height_faceid_ip,
                width=width_faceid_ip,
                generator = generator[i],
                callback_on_step_end=check_faceid_ip, 
                callback_on_step_end_tensor_inputs=['latents'], 
            ).images
        elif (is_xl_faceid_ip == True) :
            image = pipe_faceid_ip(
                input_id_images=input_id_images_faceid_ip,
                prompt=prompt_faceid_ip,
                negative_prompt=negative_prompt_faceid_ip,
#                prompt_embeds=conditioning,
#                pooled_prompt_embeds=pooled,
#                negative_prompt_embeds=neg_conditioning,
#                negative_pooled_prompt_embeds=neg_pooled,
                num_images_per_prompt=num_images_per_prompt_faceid_ip,
                guidance_scale=guidance_scale_faceid_ip,
                start_merge_step=start_merge_step_faceid_ip,
                num_inference_steps=num_inference_step_faceid_ip,
                height=height_faceid_ip,
                width=width_faceid_ip,
                generator = generator[i],
                callback_on_step_end=check_faceid_ip, 
                callback_on_step_end_tensor_inputs=['latents'], 
            ).images            
        else : 
            image = pipe_faceid_ip(
#                image_embeds=faceid_embeds_faceid_ip,
                prompt_embeds=conditioning,
                negative_prompt_embeds=neg_conditioning,
                ip_adapter_image=image_input,
                num_images_per_prompt=num_images_per_prompt_faceid_ip,
                guidance_scale=guidance_scale_faceid_ip,
                strength=denoising_strength_faceid_ip,
                num_inference_steps=num_inference_step_faceid_ip,
                height=height_faceid_ip,
                width=width_faceid_ip,
                generator = generator[i],
                callback_on_step_end=check_faceid_ip, 
                callback_on_step_end_tensor_inputs=['latents'], 
            ).images        

        for j in range(len(image)):
            seed_id = random_seed + i*num_images_per_prompt_faceid_ip + j if (seed_faceid_ip == 0) else seed_faceid_ip + i*num_images_per_prompt_faceid_ip + j
            savename = name_seeded_image(seed_id)
            if use_gfpgan_faceid_ip == True:
                image[j] = image_gfpgan_mini(image[j])
            image[j].save(savename)
            final_image.append(savename)
            final_seed.append(seed_id)

    print(f">>>[Photobooth 🖼️ ]: generated {num_prompt_faceid_ip} batch(es) of {num_images_per_prompt_faceid_ip}")
    reporting_faceid_ip = f">>>[Photobooth 🖼️ ]: "+\
        f"Settings : Model={modelid_faceid_ip} | "+\
        f"XL model={is_xl_faceid_ip} | "+\
        f"Sampler={sampler_faceid_ip} | "+\
        f"Steps={num_inference_step_faceid_ip} | "+\
        f"CFG scale={guidance_scale_faceid_ip} | "+\
        f"Size={width_faceid_ip}x{height_faceid_ip} | "+\
        f"GFPGAN={use_gfpgan_faceid_ip} | "+\
        f"Token merging={tkme_faceid_ip} | "+\
        f"LoRA model={lora_model_faceid_ip} | "+\
        f"LoRA weight={lora_weight_faceid_ip} | "+\
        f"Textual inversion={txtinv_faceid_ip} | "+\
        f"nsfw_filter={bool(int(nsfw_filter))} | "+\
        f"Denoising strength={denoising_strength_faceid_ip} | "+\
        f"Prompt={prompt_faceid_ip} | "+\
        f"Negative prompt={negative_prompt_faceid_ip} | "+\
        f"Seed List="+ ', '.join([f"{final_seed[m]}" for m in range(len(final_seed))])
    print(reporting_faceid_ip)         

    exif_writer_png(reporting_faceid_ip, final_image)

#    del nsfw_filter_final, feat_ex, pipe_faceid_ip, generator, faceid_embeds_faceid_ip, compel, conditioning, neg_conditioning, image
    del nsfw_filter_final, feat_ex, pipe_faceid_ip, generator, image
    clean_ram()

    print(f">>>[Photobooth 🖼️ ]: leaving module")
    return final_image, final_image 
