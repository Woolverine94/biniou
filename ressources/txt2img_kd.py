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

device_label_txt2img_kd, model_arch = detect_device()
device_txt2img_kd = torch.device(device_label_txt2img_kd)

# Gestion des modèles
model_path_txt2img_kd = "./models/Kandinsky/"
os.makedirs(model_path_txt2img_kd, exist_ok=True)

model_list_txt2img_kd = []

for filename in os.listdir(model_path_txt2img_kd):
    f = os.path.join(model_path_txt2img_kd, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors') or filename.endswith('.bin')):
        model_list_txt2img_kd.append(f)

model_list_txt2img_kd_builtin = [
#    "kandinsky-community/kandinsky-3",
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

def check_txt2img_kd(pipe, step_index, timestep, callback_kwargs) : 
    global stop_txt2img_kd
    if stop_txt2img_kd == False :
        return callback_kwargs
    elif stop_txt2img_kd == True :
        print(">>>[Kandinsky 🖼️ ]: generation canceled by user")
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

    print(">>>[Kandinsky 🖼️ ]: starting module")

    def check_model_type():
        txt2img_kd_model_type = model_arch if (modelid_txt2img_kd != "kandinsky-community/kandinsky-3") else torch.float16
        print(txt2img_kd_model_type)
        return txt2img_kd_model_type

    if (modelid_txt2img_kd == "kandinsky-community/kandinsky-3") :
        if modelid_txt2img_kd[0:9] == "./models/" :
            pipe_txt2img_kd = AutoPipelineForText2Image.from_single_file(
                modelid_txt2img_kd,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
        else :
            pipe_txt2img_kd = AutoPipelineForText2Image.from_pretrained(
                modelid_txt2img_kd,
                cache_dir=model_path_txt2img_kd,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )
    else :
        if modelid_txt2img_kd[0:9] == "./models/" :
            pipe_txt2img_kd = AutoPipelineForText2Image.from_single_file(
                modelid_txt2img_kd,
                torch_dtype=model_arch,
                use_safetensors=True,
            )
        else :
            pipe_txt2img_kd = AutoPipelineForText2Image.from_pretrained(
                modelid_txt2img_kd,
                cache_dir=model_path_txt2img_kd,
                torch_dtype=model_arch,
                use_safetensors=True,
                resume_download=True,
                local_files_only=True if offline_test() else None
            )

    pipe_txt2img_kd = get_scheduler(pipe=pipe_txt2img_kd, scheduler=sampler_txt2img_kd)
    pipe_txt2img_kd.enable_attention_slicing("max")
    if device_label_txt2img_kd == "cuda" :
        pipe_txt2img_kd.enable_sequential_cpu_offload()
    else :
        pipe_txt2img_kd = pipe_txt2img_kd.to(device_txt2img_kd)

    if seed_txt2img_kd == 0:
        random_seed = torch.randint(0, 10000000000, (1,))
        generator = torch.manual_seed(random_seed)
    else:
        generator = torch.manual_seed(seed_txt2img_kd)

    final_image = []
    for i in range (num_prompt_txt2img_kd):
        if (modelid_txt2img_kd == "kandinsky-community/kandinsky-3") :
            image = pipe_txt2img_kd(
                prompt=prompt_txt2img_kd,
                negative_prompt=negative_prompt_txt2img_kd,
                height=height_txt2img_kd,
                width=width_txt2img_kd,
                num_inference_steps=num_inference_step_txt2img_kd,
                guidance_scale=guidance_scale_txt2img_kd,
                num_images_per_prompt=num_images_per_prompt_txt2img_kd,
                generator = generator,
                callback=check_txt2img_kd,
            ).images
        else :
            image = pipe_txt2img_kd(
                prompt=prompt_txt2img_kd,
                negative_prompt=negative_prompt_txt2img_kd,
                height=height_txt2img_kd,
                width=width_txt2img_kd,
                num_inference_steps=num_inference_step_txt2img_kd,
                guidance_scale=guidance_scale_txt2img_kd,
                num_images_per_prompt=num_images_per_prompt_txt2img_kd,
                generator = generator,
                callback_on_step_end=check_txt2img_kd,
                callback_on_step_end_tensor_inputs=['latents'],
            ).images

        for j in range(len(image)):
            timestamp = time.time()
            savename = f"outputs/{timestamp}.png"
            if use_gfpgan_txt2img_kd == True :
                image[j] = image_gfpgan_mini(image[j])
            image[j].save(savename)
            final_image.append(image[j])

    print(f">>>[Kandinsky 🖼️ ]: generated {num_prompt_txt2img_kd} batch(es) of {num_images_per_prompt_txt2img_kd}")
    reporting_txt2img_kd = f">>>[Kandinsky 🖼️ ]: "+\
        f"Settings : Model={modelid_txt2img_kd} | "+\
        f"Sampler={sampler_txt2img_kd} | "+\
        f"Steps={num_inference_step_txt2img_kd} | "+\
        f"CFG scale={guidance_scale_txt2img_kd} | "+\
        f"Size={width_txt2img_kd}x{height_txt2img_kd} | "+\
        f"GFPGAN={use_gfpgan_txt2img_kd} | "+\
        f"Prompt={prompt_txt2img_kd} | "+\
        f"Negative prompt={negative_prompt_txt2img_kd}"
    print(reporting_txt2img_kd)

    del pipe_txt2img_kd, generator, image
    clean_ram()

    print(f">>>[Kandinsky 🖼️ ]: leaving module")
    return final_image, final_image
