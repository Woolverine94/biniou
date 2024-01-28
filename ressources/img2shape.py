# https://github.com/Woolverine94/biniou
# img2shape.py
import gradio as gr
import os
import PIL
from diffusers import ShapEImg2ImgPipeline
from diffusers.utils import export_to_gif, export_to_ply
import torch
import random
import trimesh
import numpy as np
from ressources.common import *

device_label_img2shape, model_arch = detect_device()
device_img2shape = torch.device(device_label_img2shape)

# Gestion des modèles
model_path_img2shape = "./models/Shap-E/"
model_path_img2shape_safetychecker = "./models/Stable_Diffusion/"
os.makedirs(model_path_img2shape, exist_ok=True)
model_list_img2shape = []

for filename in os.listdir(model_path_img2shape):
    f = os.path.join(model_path_img2shape, filename)
    if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.safetensors')):
        model_list_img2shape.append(f)

model_list_img2shape_builtin = [
    "openai/shap-e-img2img", 
]

for k in range(len(model_list_img2shape_builtin)):
    model_list_img2shape.append(model_list_img2shape_builtin[k])

# Bouton Cancel
stop_img2shape = False

def initiate_stop_img2shape() :
    global stop_img2shape
    stop_img2shape = True

def check_img2shape(step, timestep, latents) :
    global stop_img2shape
    if stop_img2shape == False :
#        result_preview = preview_image(step, timestep, latents, pipe_img2shape)
        return
    elif stop_img2shape == True :
        stop_img2shape = False
        try:
            del ressources.img2shape.pipe_img2shape
        except NameError as e:
            raise Exception("Interrupting ...")
    return

@metrics_decoration
def image_img2shape(
    modelid_img2shape, 
    sampler_img2shape,  
    img_img2shape, 
    num_images_per_prompt_img2shape, 
    num_prompt_img2shape, 
    guidance_scale_img2shape, 
    num_inference_step_img2shape, 
    frame_size_img2shape, 
    seed_img2shape, 
    output_type_img2shape,     
    nsfw_filter, 
    progress_img2shape=gr.Progress(track_tqdm=True)
    ):
    
    print(">>>[Shap-E img2shape 🧊 ]: starting module") 
    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_img2shape_safetychecker, device_img2shape, nsfw_filter)
    
    if modelid_img2shape[0:9] == "./models/" :
        pipe_img2shape = ShapEImg2ImgPipeline.from_single_file(
            modelid_img2shape, 
            torch_dtype=model_arch,
#            use_safetensors=True, 
            load_safety_checker=False if (nsfw_filter_final == None) else True,
#            safety_checker=nsfw_filter_final, 
#            feature_extractor=feat_ex,
        )
    else : 
        pipe_img2shape = ShapEImg2ImgPipeline.from_pretrained(
            modelid_img2shape, 
            cache_dir=model_path_img2shape, 
            torch_dtype=model_arch,
#            use_safetensors=True, 
            safety_checker=nsfw_filter_final, 
            feature_extractor=feat_ex,
            resume_download=True,
            local_files_only=True if offline_test() else None
        )
    
    image_input = img_img2shape.convert("RGB") 
    dim_size = correct_size(image_input.size[0], image_input.size[1], frame_size_img2shape)
    image_input = image_input.resize((dim_size[0],dim_size[1]))
   
    pipe_img2shape = schedulerer(pipe_img2shape, sampler_img2shape)
    pipe_img2shape.enable_attention_slicing("max")
    if device_label_img2shape == "cuda" :
        pipe_img2shape.enable_sequential_cpu_offload()
    else : 
        pipe_img2shape = pipe_img2shape.to(device_img2shape)
    
    if seed_img2shape == 0:
        random_seed = random.randrange(0, 10000000000, 1)
        final_seed = random_seed
    else:
        final_seed = seed_img2shape
    generator = []
    for k in range(num_prompt_img2shape):
        generator.append([torch.Generator(device_img2shape).manual_seed(final_seed + (k*num_images_per_prompt_img2shape) + l ) for l in range(num_images_per_prompt_img2shape)])

    savename_final = []
    final_seed = []
    for i in range (num_prompt_img2shape):
        image = []
        image = pipe_img2shape(
            image=image_input,
            frame_size=frame_size_img2shape,
            num_images_per_prompt=num_images_per_prompt_img2shape,
            num_inference_steps=num_inference_step_img2shape,
            guidance_scale=guidance_scale_img2shape,
            generator=generator[i],
            output_type="pil" if output_type_img2shape=="gif" else "mesh",
        ).images

        seed_id = random_seed + i*num_images_per_prompt_img2shape if (seed_img2shape == 0) else seed_img2shape + i*num_images_per_prompt_img2shape            
           
        if output_type_img2shape=="gif" :
            savename = f"outputs/{seed_id}_{timestamper()}.gif"
            export_to_gif(image[0], savename)
            savename_final.append(savename)
            final_seed.append(seed_id)
        
        else : 
            timestamp = timestamper()
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

    print(f">>>[Shap-E img2shape 🧊 ]: generated {num_prompt_img2shape} batch(es) of {num_images_per_prompt_img2shape}")
    reporting_img2shape = f">>>[Shap-E img2shape 🧊 ]: "+\
        f"Settings : Model={modelid_img2shape} | "+\
        f"Sampler={sampler_img2shape} | "+\
        f"Steps={num_inference_step_img2shape} | "+\
        f"CFG scale={guidance_scale_img2shape} | "+\
        f"Frame size={frame_size_img2shape} | "+\
        f"Output type={output_type_img2shape} | "+\
        f"nsfw_filter={bool(int(nsfw_filter))} | "+\
        f"Seed List="+ ', '.join([f"{final_seed[m]}" for m in range(len(final_seed))])
    print(reporting_img2shape) 

    del nsfw_filter_final, feat_ex, pipe_img2shape, generator, image
    clean_ram()

    print(f">>>[Shap-E img2shape 🧊 ]: leaving module")
    return savename_final, savename_final
