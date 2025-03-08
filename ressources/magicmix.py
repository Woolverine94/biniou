# https://github.com/Woolverine94/biniou
# magicmix.py
import gradio as gr
import os
import torch
from diffusers import DiffusionPipeline
import random
from ressources.gfpgan import *
import tomesd

# device_magicmix = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_label_magicmix, model_arch = detect_device()
device_magicmix = torch.device(device_label_magicmix)

# Gestion des modèles
model_path_magicmix = "./models/Stable_Diffusion/"
os.makedirs(model_path_magicmix, exist_ok=True)
model_list_magicmix = []

for filename in os.listdir(model_path_magicmix):
    f = os.path.join(model_path_magicmix, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
        model_list_magicmix.append(f)

model_list_magicmix_builtin = [
    "SG161222/Realistic_Vision_V3.0_VAE",
#    "ckpt/anything-v4.5-vae-swapped",
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "nitrosocke/Ghibli-Diffusion", 
]

for k in range(len(model_list_magicmix_builtin)):
    model_list_magicmix.append(model_list_magicmix_builtin[k])

# Bouton Cancel
stop_magicmix = False

def initiate_stop_magicmix() :
    global stop_magicmix
    stop_magicmix = True

def check_magicmix(step, timestep, latents) :
    global stop_magicmix
    if stop_magicmix == False :
        return
    elif stop_magicmix == True :
        print(">>>[MagicMix 🖼️ ]: generation canceled by user")
        stop_magicmix = False
        try:
            del ressources.magicmix.pipe_magicmix
        except NameError as e:
            raise Exception("Interrupting ...")
            return "Canceled ..."
    return

@metrics_decoration
def image_magicmix(
    modelid_magicmix, 
    sampler_magicmix, 
    num_inference_step_magicmix,
    guidance_scale_magicmix,
    kmin_magicmix,
    kmax_magicmix,
    num_prompt_magicmix,
    seed_magicmix,
    img_magicmix,
    prompt_magicmix,
    mix_factor_magicmix,
    use_gfpgan_magicmix, 
    nsfw_filter, 
    tkme_magicmix,
    progress_magicmix=gr.Progress(track_tqdm=True)
    ):

    print(">>>[MagicMix 🖼️ ]: starting module")

    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_magicmix, device_magicmix, nsfw_filter)
        
    if modelid_magicmix[0:9] == "./models/" :
        pipe_magicmix = DiffusionPipeline.from_single_file(
            modelid_magicmix, 
            magicmix=magicmix, 
#            torch_dtype=torch.float32, 
            torch_dtype=model_arch,
            use_safetensors=True, 
#            load_safety_checker=False if (nsfw_filter_final == None) else True,
            local_files_only=True if offline_test() else None,
#            safety_checker=nsfw_filter_final, 
#            feature_extractor=feat_ex
        )
    else :        
        pipe_magicmix = DiffusionPipeline.from_pretrained(
            modelid_magicmix, 
            custom_pipeline="magic_mix",
            cache_dir=model_path_magicmix, 
#            torch_dtype=torch.float32, 
            torch_dtype=model_arch,
            use_safetensors=True, 
            safety_checker=nsfw_filter_final, 
            feature_extractor=feat_ex,
            resume_download=True,
            local_files_only=True if offline_test() else None,
        )
   
    pipe_magicmix = schedulerer(pipe_magicmix, sampler_magicmix)
    pipe_magicmix.enable_attention_slicing("max")
    tomesd.apply_patch(pipe_magicmix, ratio=tkme_magicmix)
    if device_label_magicmix == "cuda" :
        pipe_magicmix.enable_sequential_cpu_offload()
    else : 
        pipe_magicmix = pipe_magicmix.to(device_magicmix)

    if seed_magicmix == 0:
        random_seed = random.randrange(0, 10000000000, 1)
        final_seed = random_seed
    else:
        final_seed = seed_magicmix
    generator = []
    for k in range(num_prompt_magicmix):
        generator.append(final_seed + k)

    image_input = PIL.Image.open(img_magicmix)
    dim_size = correct_size(image_input.size[0], image_input.size[1], 512)
    image_input = image_input.convert("RGB")
    image_input = image_input.resize((dim_size[0], dim_size[1]))
     
    prompt_magicmix = str(prompt_magicmix)
    if prompt_magicmix == "None":
        prompt_magicmix = ""
 
    final_image = []
    final_seed = []    
    for i in range (num_prompt_magicmix):
        image = pipe_magicmix(
            img=image_input,
            prompt=prompt_magicmix,
            kmin=kmin_magicmix,
            kmax=kmax_magicmix,
            mix_factor=mix_factor_magicmix,
            seed=generator[i],
            steps=num_inference_step_magicmix,
            guidance_scale=guidance_scale_magicmix,
        )
        if (modelid_magicmix[0:9] == "./models/"):
            image = safety_checker_sdxl(model_path_magicmix, image, nsfw_filter)
        seed_id = random_seed + i if (seed_magicmix == 0) else seed_magicmix + i
        savename = name_seeded_image(seed_id)
        if use_gfpgan_magicmix == True :
            image = image_gfpgan_mini(image)
        image.save(savename)
        final_image.append(savename)
        final_seed.append(seed_id)

    print(f">>>[MagicMix 🖼️ ]: generated {num_prompt_magicmix} batch(es) of 1")
    reporting_magicmix = f">>>[MagicMix 🖼️ ]: "+\
        f"Settings : Model={modelid_magicmix} | "+\
        f"Steps={num_inference_step_magicmix} | "+\
        f"CFG scale={guidance_scale_magicmix} | "+\
        f"Mix factor={mix_factor_magicmix} | "+\
        f"Kmin={kmin_magicmix} | "+\
        f"Kmax={kmax_magicmix} | "+\
        f"GFPGAN={use_gfpgan_magicmix} | "+\
        f"Token merging={tkme_magicmix} | "+\
        f"nsfw_filter={bool(int(nsfw_filter))} | "+\
        f"Mix factor={mix_factor_magicmix} | "+\
        f"Prompt={prompt_magicmix} | "+\
        f"Seed List="+ ', '.join([f"{final_seed[m]}" for m in range(len(final_seed))])
    print(reporting_magicmix) 

    exif_writer_png(reporting_magicmix, final_image)

    del nsfw_filter_final, feat_ex, pipe_magicmix, generator, image 
    clean_ram()

    print(f">>>[MagicMix 🖼️ ]: leaving module")
    return final_image, final_image
