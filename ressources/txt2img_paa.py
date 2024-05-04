# https://github.com/Woolverine94/biniou
# txt2img_paa.py
import gradio as gr
import os
from diffusers import PixArtAlphaPipeline, Transformer2DModel, LCMScheduler, PixArtSigmaPipeline
import torch
import random
from ressources.gfpgan import *
# import tomesd

device_label_txt2img_paa, model_arch = detect_device()
device_txt2img_paa = torch.device(device_label_txt2img_paa)

# Gestion des modèles
model_path_txt2img_paa = "./models/PixArtAlpha/"
model_path_txt2img_paa_safetychecker = "./models/Stable_Diffusion/" 
os.makedirs(model_path_txt2img_paa, exist_ok=True)
model_list_txt2img_paa = []

for filename in os.listdir(model_path_txt2img_paa):
    f = os.path.join(model_path_txt2img_paa, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
        model_list_txt2img_paa.append(f)

model_list_txt2img_paa_builtin = [
    "PixArt-alpha/PixArt-XL-2-512x512",
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    "PixArt-alpha/PixArt-LCM-XL-2-1024-MS",
    "Luo-Yihong/yoso_pixart512",
    "Luo-Yihong/yoso_pixart1024", 
]

for k in range(len(model_list_txt2img_paa_builtin)):
    model_list_txt2img_paa.append(model_list_txt2img_paa_builtin[k])

# Bouton Cancel
stop_txt2img_paa = False

def initiate_stop_txt2img_paa() :
    global stop_txt2img_paa
    stop_txt2img_paa = True

def check_txt2img_paa(step, timestep, latents) :
    global stop_txt2img_paa
    if stop_txt2img_paa == False :
#        result_preview = preview_image(step, timestep, latents, pipe_txt2img_paa)
        return
    elif stop_txt2img_paa == True :
        print(">>>[PixArt-Alpha 🖼️ ]: generation canceled by user")
        stop_txt2img_paa = False
        try:
            del ressources.txt2img_paa.pipe_txt2img_paa
        except NameError as e:
            raise Exception("Interrupting ...")
    return

@metrics_decoration
def image_txt2img_paa(
    modelid_txt2img_paa,
    sampler_txt2img_paa,
    prompt_txt2img_paa,
    negative_prompt_txt2img_paa,
    num_images_per_prompt_txt2img_paa,
    num_prompt_txt2img_paa,
    guidance_scale_txt2img_paa,
    num_inference_step_txt2img_paa,
    height_txt2img_paa,
    width_txt2img_paa,
    seed_txt2img_paa,
    use_gfpgan_txt2img_paa,
    nsfw_filter,
    tkme_txt2img_paa,
    progress_txt2img_paa=gr.Progress(track_tqdm=True)
    ):

    print(">>>[PixArt-Alpha 🖼️ ]: starting module")
    
#    global pipe_txt2img_paa
    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_txt2img_paa_safetychecker, device_txt2img_paa, nsfw_filter)

    if ("YOSO" in modelid_txt2img_paa.upper()):
        if (modelid_txt2img_paa == "Luo-Yihong/yoso_pixart512"):
            transformerid_txt2img_paa = modelid_txt2img_paa
            modelid_txt2img_paa = "PixArt-alpha/PixArt-XL-2-512x512"
        elif (modelid_txt2img_paa == "Luo-Yihong/yoso_pixart1024"):
            transformerid_txt2img_paa = modelid_txt2img_paa
            modelid_txt2img_paa = "PixArt-alpha/PixArt-XL-2-1024-MS"

        transformer_txt2img_paa = Transformer2DModel.from_pretrained(
            transformerid_txt2img_paa,
            cache_dir=model_path_txt2img_paa,
            torch_dtype=model_arch,
            use_safetensors=True,
            resume_download=True,
            local_files_only=True if offline_test() else None
        )
        if modelid_txt2img_paa[0:9] == "./models/" :
            pipe_txt2img_paa = PixArtAlphaPipeline.from_single_file(
                modelid_txt2img_paa, 
                transformer=transformer_txt2img_paa,
                torch_dtype=model_arch,
                use_safetensors=True, 
                load_safety_checker=False if (nsfw_filter_final == None) else True,
    #            safety_checker=nsfw_filter_final, 
    #            feature_extractor=feat_ex,
            )
        else :        
            pipe_txt2img_paa = PixArtAlphaPipeline.from_pretrained(
                modelid_txt2img_paa, 
                transformer=transformer_txt2img_paa,
                cache_dir=model_path_txt2img_paa, 
                torch_dtype=model_arch,
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
        pipe_txt2img_paa.scheduler = LCMScheduler.from_config(pipe_txt2img_paa.scheduler.config)
        pipe_txt2img_paa.scheduler.config.prediction_type = "v_prediction"

    elif ("PIXART-SIGMA-XL-2" in modelid_txt2img_paa.upper()):
        if modelid_txt2img_paa[0:9] == "./models/" :
            pipe_txt2img_paa = PixArtSigmaPipeline.from_single_file(
                modelid_txt2img_paa,
                torch_dtype=model_arch,
                use_safetensors=True,
                load_safety_checker=False if (nsfw_filter_final == None) else True,
    #            safety_checker=nsfw_filter_final,
    #            feature_extractor=feat_ex,
            )
        else:
            pipe_txt2img_paa = PixArtSigmaPipeline.from_pretrained(
                modelid_txt2img_paa,
                cache_dir=model_path_txt2img_paa,
                torch_dtype=model_arch,
                use_safetensors=True,
                safety_checker=nsfw_filter_final,
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
        pipe_txt2img_paa = schedulerer(pipe_txt2img_paa, sampler_txt2img_paa)

    else:
        if modelid_txt2img_paa[0:9] == "./models/" :
            pipe_txt2img_paa = PixArtAlphaPipeline.from_single_file(
                modelid_txt2img_paa, 
                torch_dtype=model_arch,
                use_safetensors=True, 
                load_safety_checker=False if (nsfw_filter_final == None) else True,
    #            safety_checker=nsfw_filter_final, 
    #            feature_extractor=feat_ex,
            )
        else :        
            pipe_txt2img_paa = PixArtAlphaPipeline.from_pretrained(
                modelid_txt2img_paa, 
                cache_dir=model_path_txt2img_paa, 
                torch_dtype=model_arch,
                use_safetensors=True, 
                safety_checker=nsfw_filter_final, 
                feature_extractor=feat_ex,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
        pipe_txt2img_paa = schedulerer(pipe_txt2img_paa, sampler_txt2img_paa)

    pipe_txt2img_paa.enable_attention_slicing("max")
#    tomesd.apply_patch(pipe_txt2img_paa, ratio=tkme_txt2img_paa)
    if device_label_txt2img_paa == "cuda" :
        pipe_txt2img_paa.enable_sequential_cpu_offload()
    else : 
        pipe_txt2img_paa = pipe_txt2img_paa.to(device_txt2img_paa)
    
    if seed_txt2img_paa == 0:
        random_seed = random.randrange(0, 10000000000, 1)
        final_seed = random_seed
    else:
        final_seed = seed_txt2img_paa
    generator = []
    for k in range(num_prompt_txt2img_paa):
        generator.append([torch.Generator(device_txt2img_paa).manual_seed(final_seed + (k*num_images_per_prompt_txt2img_paa) + l ) for l in range(num_images_per_prompt_txt2img_paa)])

    prompt_txt2img_paa = str(prompt_txt2img_paa)
    negative_prompt_txt2img_paa = str(negative_prompt_txt2img_paa) 
    if prompt_txt2img_paa == "None":
        prompt_txt2img_paa = ""
    if negative_prompt_txt2img_paa == "None":
        negative_prompt_txt2img_paa = ""

    final_image = []
    final_seed = []
    for i in range (num_prompt_txt2img_paa):
        image = pipe_txt2img_paa(
            prompt=prompt_txt2img_paa,
            negative_prompt=negative_prompt_txt2img_paa,
            height=height_txt2img_paa,
            width=width_txt2img_paa,
            num_images_per_prompt=num_images_per_prompt_txt2img_paa,
            num_inference_steps=num_inference_step_txt2img_paa,
            guidance_scale=guidance_scale_txt2img_paa,
            generator = generator[i],
            clean_caption=False,
            callback=check_txt2img_paa, 
        ).images

        for j in range(len(image)):
            seed_id = random_seed + i*num_images_per_prompt_txt2img_paa + j if (seed_txt2img_paa == 0) else seed_txt2img_paa + i*num_images_per_prompt_txt2img_paa + j
            savename = name_seeded_image(seed_id)
            if use_gfpgan_txt2img_paa == True :
                image[j] = image_gfpgan_mini(image[j])
            image[j].save(savename)
            final_image.append(savename)
            final_seed.append(seed_id)

    print(f">>>[PixArt-Alpha 🖼️ ]: generated {num_prompt_txt2img_paa} batch(es) of {num_images_per_prompt_txt2img_paa}")
    reporting_txt2img_paa = f">>>[PixArt-Alpha 🖼️ ]: "+\
        f"Settings : Model={modelid_txt2img_paa} | "+\
        f"Sampler={sampler_txt2img_paa} | "+\
        f"Steps={num_inference_step_txt2img_paa} | "+\
        f"CFG scale={guidance_scale_txt2img_paa} | "+\
        f"Size={width_txt2img_paa}x{height_txt2img_paa} | "+\
        f"GFPGAN={use_gfpgan_txt2img_paa} | "+\
        f"Token merging={tkme_txt2img_paa} | "+\
        f"nsfw_filter={bool(int(nsfw_filter))} | "+\
        f"Prompt={prompt_txt2img_paa} | "+\
        f"Negative prompt={negative_prompt_txt2img_paa} | "+\
        f"Seed List="+ ', '.join([f"{final_seed[m]}" for m in range(len(final_seed))])
    print(reporting_txt2img_paa) 

    exif_writer_png(reporting_txt2img_paa, final_image)

    del nsfw_filter_final, feat_ex, pipe_txt2img_paa, generator, image
    clean_ram()

    print(f">>>[PixArt-Alpha 🖼️ ]: leaving module") 
    return final_image, final_image
