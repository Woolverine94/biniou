# https://github.com/Woolverine94/biniou
# txt2img_kd.py
import gradio as gr
import os
from diffusers import AutoPipelineForText2Image
from compel import Compel
import torch
import time
import random
from ressources.scheduler import *
from ressources.gfpgan import *
import tomesd

device_txt2img_kd = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Gestion des modèles
model_path_txt2img_kd = "./models/Kandinsky/"
os.makedirs(model_path_txt2img_kd, exist_ok=True)

model_list_txt2img_kd = []

for filename in os.listdir(model_path_txt2img_kd):
    f = os.path.join(model_path_txt2img_kd, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
        model_list_txt2img_kd.append(f)

model_list_txt2img_kd_builtin = [
    "kandinsky-community/kandinsky-2-2-decoder",
    "kandinsky-community/kandinsky-2-1",
]

for k in range(len(model_list_txt2img_kd_builtin)):
    model_list_txt2img_kd.append(model_list_txt2img_kd_builtin[k])

# Bouton Cancel
stop_txt2img_kd = False

def initiate_stop_txt2img_kd() :
    global stop_txt2img_kd
    stop_txt2img_kd = True

def check_txt2img_kd(step, timestep, latents) : 
    global stop_txt2img_kd
    if stop_txt2img_kd == False :
        return
    elif stop_txt2img_kd == True :
        stop_txt2img_kd = False
        try:
            del ressources.txt2img_kd.pipe_txt2img_kd
        except NameError as e:
            raise Exception("Interrupting ...")
    return

@metrics_decoration
def image_txt2img_kd(
    modelid_txt2img_kd, 
    sampler_txt2img_kd, 
    prompt_txt2img_kd, 
    negative_prompt_txt2img_kd, 
    num_images_per_prompt_txt2img_kd, 
    num_prompt_txt2img_kd, 
    guidance_scale_txt2img_kd, 
    num_inference_step_txt2img_kd, 
    height_txt2img_kd, 
    width_txt2img_kd, 
    seed_txt2img_kd, 
    use_gfpgan_txt2img_kd, 
#    tkme_txt2img_kd,
    progress_txt2img_kd=gr.Progress(track_tqdm=True)
    ):
        
    if modelid_txt2img_kd[0:9] == "./models/" :
        pipe_txt2img_kd = AutoPipelineForText2Image.from_single_file(
            modelid_txt2img_kd, 
            torch_dtype=torch.float32, 
            use_safetensors=True,
        )
    else :        
        pipe_txt2img_kd = AutoPipelineForText2Image.from_pretrained(
            modelid_txt2img_kd, 
            cache_dir=model_path_txt2img_kd, 
            torch_dtype=torch.float32, 
            use_safetensors=True,
            resume_download=True,
            local_files_only=True if offline_test() else None
        )
        
    pipe_txt2img_kd = get_scheduler(pipe=pipe_txt2img_kd, scheduler=sampler_txt2img_kd)
    pipe_txt2img_kd = pipe_txt2img_kd.to(device_txt2img_kd)
    pipe_txt2img_kd.enable_attention_slicing("max")  
#    tomesd.apply_patch(pipe_txt2img_kd, ratio=tkme_txt2img_kd)

    if seed_txt2img_kd == 0:
        random_seed = torch.randint(0, 10000000000, (1,))
        generator = torch.manual_seed(random_seed)
    else:
        generator = torch.manual_seed(seed_txt2img_kd)

    final_image = []
    for i in range (num_prompt_txt2img_kd):
        image = pipe_txt2img_kd(
            prompt=prompt_txt2img_kd,
            negative_prompt=negative_prompt_txt2img_kd,
            height=height_txt2img_kd,
            width=width_txt2img_kd,
            num_inference_steps=num_inference_step_txt2img_kd,
            guidance_scale=guidance_scale_txt2img_kd,
            num_images_per_prompt=num_images_per_prompt_txt2img_kd,
            generator = generator,
            callback = check_txt2img_kd,            
        ).images

        for j in range(len(image)):
            timestamp = time.time()
            savename = f"outputs/{timestamp}.png"
            if use_gfpgan_txt2img_kd == True :
                image[j] = image_gfpgan_mini(image[j])            
            image[j].save(savename)
            final_image.append(image[j])
            
    del pipe_txt2img_kd, generator, image 
    clean_ram()            
    
    return final_image, final_image
