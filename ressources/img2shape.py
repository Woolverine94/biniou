# https://github.com/Woolverine94/biniou
# img2shape.py
import gradio as gr
import os
import PIL
from diffusers import ShapEImg2ImgPipeline
from diffusers.utils import export_to_gif, export_to_ply
import torch
import time
import random
import trimesh
import numpy as np
from ressources.scheduler import *
from ressources.common import *

device_img2shape = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
    global pipe_img2shape
    nsfw_filter_final, feat_ex = safety_checker_sd(model_path_img2shape_safetychecker, device_img2shape, nsfw_filter)
    
    if modelid_img2shape[0:9] == "./models/" :
        pipe_img2shape = ShapEImg2ImgPipeline.from_single_file(
            modelid_img2shape, 
            torch_dtype=torch.float32, 
#            use_safetensors=True, 
            safety_checker=nsfw_filter_final, 
            feature_extractor=feat_ex,
        )
    else : 
        pipe_img2shape = ShapEImg2ImgPipeline.from_pretrained(
            modelid_img2shape, 
            cache_dir=model_path_img2shape, 
            torch_dtype=torch.float32, 
#            use_safetensors=True, 
            safety_checker=nsfw_filter_final, 
            feature_extractor=feat_ex,
            resume_download=True,
            local_files_only=True if offline_test() else None
        )
    
    image_input = img_img2shape.convert("RGB") 
    dim_size = correct_size(image_input.size[0], image_input.size[1], frame_size_img2shape)
    image_input = image_input.resize((dim_size[0],dim_size[1]))
   
    pipe_img2shape = get_scheduler(pipe=pipe_img2shape, scheduler=sampler_img2shape)
    pipe_img2shape = pipe_img2shape.to(device_img2shape)
    pipe_img2shape.enable_attention_slicing("max")
    
    if seed_img2shape == 0:
        random_seed = torch.randint(0, 10000000000, (1,))
        generator = torch.manual_seed(random_seed)
    else:
        generator = torch.manual_seed(seed_img2shape)

    final_image = []
    for i in range (num_prompt_img2shape):
        image = pipe_img2shape(
            image=image_input,
            frame_size=frame_size_img2shape,
            num_images_per_prompt=num_images_per_prompt_img2shape,
            num_inference_steps=num_inference_step_img2shape,
            guidance_scale=guidance_scale_img2shape,
            generator=generator,
            output_type="pil" if output_type_img2shape=="gif" else "mesh",
        ).images
       
    if output_type_img2shape=="gif" :
        for j in range(len(image)):
            timestamp = time.time()
            savename = f"outputs/{timestamp}.gif"
            export_to_gif(image[j], savename)
            final_image.append(savename)

        del nsfw_filter_final, feat_ex, pipe_img2shape, generator, image
        clean_ram()

        return final_image, final_image

    else : 
        timestamp = time.time()
        savename = f".tmp/{timestamp}.ply"
        savename_glb = f"outputs/{timestamp}.glb"
        savename_final = f".tmp/output.glb" 
        ply_file = export_to_ply(image[0], savename) 
        mesh = trimesh.load(savename)
        mesh = mesh.scene()
        rot = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        mesh = mesh.apply_transform(rot)
        rot = trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
        mesh = mesh.apply_transform(rot)
        mesh.export(savename_glb, file_type="glb")
        mesh.export(savename_final, file_type="glb") 

        del nsfw_filter_final, feat_ex, pipe_img2shape, generator, image
        clean_ram()

        return savename_final, savename_final
    return
