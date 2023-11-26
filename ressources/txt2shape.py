# https://github.com/Woolverine94/biniou
# txt2shape.py
import gradio as gr
import os
from diffusers import ShapEPipeline
from diffusers.utils import export_to_gif, export_to_obj, export_to_ply
import torch
import time
import random
import trimesh
import numpy as np
from ressources.scheduler import *
from ressources.common import *

device_label_txt2shape, model_arch = detect_device()
device_txt2shape = torch.device(device_label_txt2shape)

# Gestion des modèles
model_path_txt2shape = "./models/Shap-E/"
model_path_txt2shape_safetychecker = "./models/Stable_Diffusion/"
os.makedirs(model_path_txt2shape, exist_ok=True)
model_list_txt2shape = []

for filename in os.listdir(model_path_txt2shape):
    f = os.path.join(model_path_txt2shape, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
        model_list_txt2shape.append(f)

model_list_txt2shape_builtin = [
    "openai/shap-e", 
]

for k in range(len(model_list_txt2shape_builtin)):
    model_list_txt2shape.append(model_list_txt2shape_builtin[k])

# Bouton Cancel
stop_txt2shape = False

def initiate_stop_txt2shape() :
    global stop_txt2shape
    stop_txt2shape = True

def check_txt2shape(step, timestep, latents) :
    global stop_txt2shape
    if stop_txt2shape == False :
#        result_preview = preview_image(step, timestep, latents, pipe_txt2shape)
        return
    elif stop_txt2shape == True :
        stop_txt2shape = False
        try:
            del ressources.txt2shape.pipe_txt2shape
        except NameError as e:
            raise Exception("Interrupting ...")
    return

@metrics_decoration
def image_txt2shape(
    modelid_txt2shape, 
    sampler_txt2shape,  
    prompt_txt2shape, 
    num_images_per_prompt_txt2shape, 
    num_prompt_txt2shape, 
    guidance_scale_txt2shape, 
    num_inference_step_txt2shape, 
    frame_size_txt2shape, 
    seed_txt2shape, 
    output_type_txt2shape, 
    nsfw_filter, 
    progress_txt2shape=gr.Progress(track_tqdm=True)
    ):

    print(">>>[Shap-E txt2shape 🧊]: starting module") 
    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_txt2shape_safetychecker, device_txt2shape, nsfw_filter)

    if modelid_txt2shape[0:9] == "./models/" :
        pipe_txt2shape = ShapEPipeline.from_single_file(
            modelid_txt2shape, 
            torch_dtype=model_arch,
#            use_safetensors=True,
            safety_checker=nsfw_filter_final, 
            feature_extractor=feat_ex,
        )
    else : 
        pipe_txt2shape = ShapEPipeline.from_pretrained(
            modelid_txt2shape, 
            cache_dir=model_path_txt2shape, 
            torch_dtype=model_arch,
#            use_safetensors=True,
            safety_checker=nsfw_filter_final, 
            feature_extractor=feat_ex,
            resume_download=True,
            local_files_only=True if offline_test() else None
        )
    
    pipe_txt2shape = get_scheduler(pipe=pipe_txt2shape, scheduler=sampler_txt2shape)
    pipe_txt2shape.enable_attention_slicing("max")
    if device_label_txt2shape == "cuda" :
        pipe_txt2shape.enable_sequential_cpu_offload()
    else : 
        pipe_txt2shape = pipe_txt2shape.to(device_txt2shape)
    
    if seed_txt2shape == 0:
        random_seed = random.randrange(0, 10000000000, 1)
        final_seed = random_seed
    else:
        final_seed = seed_txt2shape
    generator = []
    for k in range(num_prompt_txt2shape):
        generator.append([torch.Generator(device_txt2shape).manual_seed(final_seed + (k*num_images_per_prompt_txt2shape) + l ) for l in range(num_images_per_prompt_txt2shape)])

    prompt_txt2shape = str(prompt_txt2shape)
    if prompt_txt2shape == "None":
        prompt_txt2shape = ""
 
    savename_final = []
    final_seed = []
    for i in range (num_prompt_txt2shape):
        image = [] 
        image = pipe_txt2shape(
            prompt=prompt_txt2shape,
            frame_size=frame_size_txt2shape,
            num_images_per_prompt=num_images_per_prompt_txt2shape,
            num_inference_steps=num_inference_step_txt2shape,
            guidance_scale=guidance_scale_txt2shape,
            generator = generator[i],
            output_type="pil" if output_type_txt2shape=="gif" else "mesh",
        ).images
        
        seed_id = random_seed + i*num_images_per_prompt_txt2shape if (seed_txt2shape == 0) else seed_txt2shape + i*num_images_per_prompt_txt2shape
       
        if output_type_txt2shape=="gif" :
            timestamp = time.time()
            savename = f"outputs/{seed_id}_{timestamp}.gif"
            export_to_gif(image[0], savename)
            savename_final.append(savename)
            final_seed.append(seed_id)
       
        else : 
            timestamp = time.time()
            savename = f".tmp/{seed_id}_{timestamp}.ply"
            savename_glb = f"outputs/{seed_id}_{timestamp}.glb"
            savename_final = f".tmp/{seed_id}_output.glb" 
            ply_file = export_to_ply(image[0], savename) 
            mesh = trimesh.load(savename)
            mesh = mesh.scene()
            rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
            mesh = mesh.apply_transform(rot)
            rot = trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
            mesh = mesh.apply_transform(rot)
            mesh.export(savename_glb, file_type="glb")
            mesh.export(savename_final, file_type="glb") 
            final_seed.append(seed_id)

    print(f">>>[Shap-E txt2shape 🧊 ]: generated {num_prompt_txt2shape} batch(es) of {num_images_per_prompt_txt2shape}")
    reporting_txt2shape = f">>>[Shap-E txt2shape 🧊 ]: "+\
        f"Settings : Model={modelid_txt2shape} | "+\
        f"Sampler={sampler_txt2shape} | "+\
        f"Steps={num_inference_step_txt2shape} | "+\
        f"CFG scale={guidance_scale_txt2shape} | "+\
        f"Frame size={frame_size_txt2shape} | "+\
        f"Output type={output_type_txt2shape} | "+\
        f"nsfw_filter={bool(int(nsfw_filter))} | "+\
        f"Prompt={prompt_txt2shape} | "+\
        f"Seed List="+ ', '.join([f"{final_seed[m]}" for m in range(len(final_seed))])
    print(reporting_txt2shape) 
    
    del nsfw_filter_final, feat_ex, pipe_txt2shape, generator, image
    clean_ram()
    
    print(f">>>[Shap-E txt2shape 🧊 ]: leaving module")
    return savename_final, savename_final
